const TARGET_SAMPLE_RATE = 16000;
const FFT_LENGTH = 512;
const HOP_LENGTH = 160;
const WIN_LENGTH = 400;
const MEL_BINS = 80;
const MEL_FMIN = 20;
const MEL_FMAX = 7600;
const TARGET_FRAMES = 300;

const loadingStatusEl = document.getElementById('loadingStatus');
const perfWarningEl = document.getElementById('perfWarning');

// 实时录音相关元素
const recordBtn = document.getElementById('recordBtn');
const pipBtn = document.getElementById('pipBtn');
const maleBar = document.getElementById('maleBar');
const femaleBar = document.getElementById('femaleBar');
const malePercent = document.getElementById('malePercent');
const femalePercent = document.getElementById('femalePercent');
const chartCanvas = document.getElementById('genderChart');

// F0 相关元素
const f0Value = document.getElementById('f0Value');
const f0ChartCanvas = document.getElementById('f0Chart');

// 提词文本框相关元素
const promptText = document.getElementById('promptText');

let session = null;
let cachedMelWeightMatrix = null;

// 实时录音相关变量
let audioContext = null;
let microphone = null;
let scriptProcessor = null;
let mediaStream = null;
let isRecording = false;
let audioChunks = [];
let recordInterval = null;
const DEFAULT_ANALYSIS_INTERVAL = 600;
const PERF_WARNING_TRIGGER_COUNT = 4;
let perfOverrunCount = 0;

// 图表相关变量
let genderChart = null;
let f0Chart = null;
let pipGenderChart = null;
let pipWindow = null;
let f0TargetValue = parseInt(localStorage.getItem('f0TargetValue')) || 200; // F0目标值，从localStorage读取

// 性别置信度范围配置（从localStorage读取）
let genderMinPercent = parseFloat(localStorage.getItem('genderMinPercent')) || 85;
let genderMaxPercent = parseFloat(localStorage.getItem('genderMaxPercent')) || 99.8;
let genderTargetPercent = parseFloat(localStorage.getItem('genderTargetPercent')) || 98.2;

// 时间跨度配置（从localStorage读取，单位：秒）
let timeSpanSeconds = parseInt(localStorage.getItem('timeSpanSeconds')) || 30;
const intervalSeconds = 0.5; // 分析间隔（秒）

// 阈值报警配置（从localStorage读取）
let alertMinPercent = parseFloat(localStorage.getItem('alertMinPercent')) || 20;
let alertMaxPercent = parseFloat(localStorage.getItem('alertMaxPercent')) || 98.2;
let alertDurationSeconds = parseInt(localStorage.getItem('alertDurationSeconds')) || 5;
let alertEnabled = localStorage.getItem('alertEnabled') === 'true';
let alertConsecutiveCount = 0; // 连续落在报警区间的计数
let lastAlertTime = 0; // 上次报警时间（用于防止重复报警）
const ALERT_COOLDOWN = 10000; // 报警冷却时间（毫秒）

// 根据时间跨度计算最大数据点数
function getMaxDataPoints() {
  return Math.ceil(timeSpanSeconds / intervalSeconds);
}
let analysisCount = 0; // 总分析次数计数器
let chartData = {
  labels: [],
  male: [],
  female: []
};
let f0Data = {
  labels: [],
  f0: []
};
let recordingStartTime = null;
let windowStartIndex = 0;
let cachedF0Ranges = {
  length: 0,
  maleMax: [],
  maleMin: [],
  femaleMax: [],
  femaleMin: []
};

function setStatus(message) {
  if (!loadingStatusEl) return;

  if (!message) {
    loadingStatusEl.textContent = '';
    return;
  }

  if (message.includes('加载模型中')) {
    const isFirstVisit = !localStorage.getItem('modelLoadedBefore');
    loadingStatusEl.textContent = isFirstVisit ? '正在下载... (约70MB）' : '正在加载...';
    return;
  }

  loadingStatusEl.textContent = message;
}

function updatePerformanceWarning(hasVoice, durationMs, thresholdMs) {
  if (!perfWarningEl) return;
  if (perfWarningEl.classList.contains('visible')) return;
  if (recordingStartTime && Date.now() - recordingStartTime > 30000) {
    return;
  }
  if (!hasVoice) {
    perfOverrunCount = 0;
  } else if (durationMs > thresholdMs) {
    perfOverrunCount++;
  } else {
    perfOverrunCount = 0;
  }

  if (perfOverrunCount >= PERF_WARNING_TRIGGER_COUNT) {
    perfWarningEl.classList.add('visible');
    if (window.Sentry && typeof window.Sentry.captureMessage === 'function') {
      window.Sentry.captureMessage('performance_warning', 'warning');
    }
    if (typeof gtag === 'function') {
      gtag('event', 'performance_warning');
    }
  }
}

function updateRealtimeDisplay(maleProb, femaleProb) {
  const malePercentVal = (maleProb * 100).toFixed(1);
  const femalePercentVal = (femaleProb * 100).toFixed(1);

  maleBar.style.width = malePercentVal + '%';
  femaleBar.style.width = femalePercentVal + '%';

  malePercent.textContent = malePercentVal + '%';
  femalePercent.textContent = femalePercentVal + '%';
}

// 计算音频的 RMS（均方根）来检测静音
function calculateRMS(audioBuffer) {
  let sum = 0;
  for (let i = 0; i < audioBuffer.length; i++) {
    sum += audioBuffer[i] * audioBuffer[i];
  }
  return Math.sqrt(sum / audioBuffer.length);
}

// LOGIT 变换: logit(p) = ln(p / (1-p))
function logit(p) {
  // p 是 0-1 之间的概率值
  // 避免除以零和 log(0)
  p = Math.max(0.001, Math.min(0.999, p));
  return Math.log(p / (1 - p));
}

// Sigmoid 变换: sigmoid(x) = 1 / (1 + e^(-x))，用于从 logit 转回概率
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// 从 logit 值获取百分比显示
function logitToPercent(logitVal) {
  const prob = sigmoid(logitVal);
  return (prob * 100).toFixed(1) + '%';
}

// 检查阈值报警
function checkAlertThreshold(femaleProb) {
  if (!alertEnabled || femaleProb === null) return;

  const femalePercent = femaleProb * 100;

  // 检查是否在报警区间内
  if (femalePercent >= alertMinPercent && femalePercent <= alertMaxPercent) {
    alertConsecutiveCount++;
    const currentDuration = alertConsecutiveCount * intervalSeconds;

    // 检查是否达到报警持续时间阈值
    if (currentDuration >= alertDurationSeconds) {
      const now = Date.now();

      // 检查冷却时间，防止重复报警
      if (now - lastAlertTime >= ALERT_COOLDOWN) {
        triggerAlert(femalePercent, currentDuration);
        lastAlertTime = now;
      }
    }
  } else {
    // 不在报警区间内，重置计数
    alertConsecutiveCount = 0;
  }
}

// 触发报警通知
function triggerAlert(currentPercent, duration) {
  // 检查 Notification API 是否存在
  if (typeof Notification === 'undefined') {
    return;
  }
  // 请求通知权限
  if (Notification.permission === 'default') {
    Notification.requestPermission().then(permission => {
      if (permission === 'granted') {
        showNotification(currentPercent, duration);
      }
    });
  } else if (Notification.permission === 'granted') {
    showNotification(currentPercent, duration);
  }
}

// 显示通知
function showNotification(currentPercent, duration) {
  const notification = new Notification('声音性别警告', {
    body: `女性声音持续 ${duration.toFixed(1)} 秒在 ${alertMinPercent}%-${alertMaxPercent}% 区间内（当前: ${currentPercent.toFixed(1)}%）`,
    icon: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">⚠️</text></svg>',
    tag: 'voice-gender-alert',
    requireInteraction: false,
    silent: true
  });

  // 3秒后自动关闭
  setTimeout(() => {
    notification.close();
  }, 3000);
}

// F0 检测算法（改进的自相关法，减少倍频错误）
function detectF0(audioBuffer, sampleRate) {
  const minFreq = 60;
  const maxFreq = 400;
  const minPeriod = Math.floor(sampleRate / maxFreq);
  const maxPeriod = Math.ceil(sampleRate / minFreq);

  // 使用归一化自相关法
  const autocorr = new Float32Array(maxPeriod);
  for (let lag = 0; lag < maxPeriod; lag++) {
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < audioBuffer.length - lag; i++) {
      sum += audioBuffer[i] * audioBuffer[i + lag];
    }
    // 归一化
    for (let i = 0; i < audioBuffer.length - lag; i++) {
      sumSq += audioBuffer[i] * audioBuffer[i];
    }
    autocorr[lag] = sum / (sumSq + 1e-10);
  }

  // 找到第一个显著峰值（避免检测到倍频）
  let bestLag = minPeriod;
  let bestValue = 0;

  // 使用差分法检测峰值：只有当值同时大于前后值时才认为是峰值
  for (let lag = minPeriod + 1; lag < maxPeriod - 1; lag++) {
    if (autocorr[lag] > autocorr[lag - 1] &&
        autocorr[lag] > autocorr[lag + 1] &&
        autocorr[lag] > 0.3) {  // 峰值阈值，避免噪声干扰

      // 验证这是基频而不是倍频：检查 2*lag 位置是否有更大的峰值
      const doubleLag = lag * 2;
      if (doubleLag < maxPeriod) {
        // 如果 2*lag 位置的峰值明显更大，说明当前检测到的是倍频
        if (autocorr[doubleLag] > autocorr[lag] * 1.2) {
          continue; // 跳过这个峰值，寻找基频
        }
      }

      if (autocorr[lag] > bestValue) {
        bestValue = autocorr[lag];
        bestLag = lag;
      }
    }
  }

  // 抛物线插值提高精度
  if (bestValue > 0 && bestLag > minPeriod && bestLag < maxPeriod - 1) {
    const y1 = autocorr[bestLag - 1];
    const y2 = autocorr[bestLag];
    const y3 = autocorr[bestLag + 1];
    const refinedLag = bestLag + (y3 - y1) / (2 * (2 * y2 - y1 - y3));
    const f0 = sampleRate / refinedLag;

    // 限制在合理范围内
    if (f0 >= minFreq && f0 <= maxFreq) {
      return f0;
    }
  }
  return 0;
}

// 初始化图表
function initChart() {
  const { min, max, target } = getGenderLogitRange();
  const stepSize = calculateGenderStepSize(min, max);

  genderChart = new Chart(chartCanvas, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: '男性',
          data: [],
          borderColor: '#3b82f6',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 5],
          fill: false,
          tension: 0.4,
          pointRadius: 0
        },
        {
          label: '女性',
          data: [],
          borderColor: '#ec4899',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 5],
          fill: false,
          tension: 0.4,
          pointRadius: 0
        },
        {
          label: '男性平均值',
          data: [],
          borderColor: '#1d4ed8',
          backgroundColor: 'rgba(29, 78, 216, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 0
        },
        {
          label: '女性平均值',
          data: [],
          borderColor: '#be185d',
          backgroundColor: 'rgba(190, 24, 93, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 0
        },
        {
          label: '目标',
          // 初始化时设置两个点，确保目标线始终可见
          data: [target, target],
          borderColor: '#a855f7',
          borderWidth: 2,
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          borderDash: [],
          spanGaps: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 300
      },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: '时间 (秒)'
          }
        },
        y: {
          type: 'linear',
          display: true,
          min: min,
          max: max,
          title: {
            display: true,
            text: '置信度 (LOGIT 尺度)'
          },
          ticks: {
            stepSize: stepSize,
            callback: function(value, index, values) {
              // 将 logit 值转换回百分比显示
              return logitToPercent(value);
            }
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            font: {
              size: 12,
              weight: 'normal'
            }
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              const value = context.parsed.y;
              // 将 logit 值转换回百分比
              label += logitToPercent(value);
              return label;
            }
          }
        }
      }
    }
  });
}

// 初始化 F0 图表
function initF0Chart() {
  f0Chart = new Chart(f0ChartCanvas, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'F0 (Hz)',
          data: [],
          borderColor: '#10b981',
          backgroundColor: 'transparent',
          borderWidth: 2,
          fill: false,
          tension: 0.4,
          pointRadius: 0
        },
        {
          label: '男性范围',
          data: [],
          borderColor: 'transparent',
          backgroundColor: 'rgba(59, 130, 246, 0.08)',
          fill: false,
          pointRadius: 0,
          borderDash: [5, 5]
        },
        {
          label: '男性范围',
          data: [],
          borderColor: 'transparent',
          backgroundColor: 'rgba(59, 130, 246, 0.08)',
          fill: '-1',
          pointRadius: 0,
          borderDash: [5, 5]
        },
        {
          label: '女性范围',
          data: [],
          borderColor: 'transparent',
          backgroundColor: 'rgba(236, 72, 153, 0.08)',
          fill: false,
          pointRadius: 0,
          borderDash: [5, 5]
        },
        {
          label: '女性范围',
          data: [],
          borderColor: 'transparent',
          backgroundColor: 'rgba(236, 72, 153, 0.08)',
          fill: '-1',
          pointRadius: 0,
          borderDash: [5, 5]
        },
        {
          label: '目标',
          data: [f0TargetValue, f0TargetValue],
          borderColor: '#a855f7',
          borderWidth: 2,
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          borderDash: [],
          spanGaps: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 300
      },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: '时间 (秒)'
          }
        },
        y: {
          display: true,
          min: 50,
          max: 300,
          title: {
            display: true,
            text: '频率 (Hz)'
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            font: {
              size: 12,
              weight: 'normal'
            },
            // 过滤重复的图例项（只显示每个标签的第一个）
            filter: function(legendItem, chartData) {
              const label = legendItem.text;
              const index = legendItem.datasetIndex;
              // 检查是否是第一次出现这个标签
              for (let i = 0; i < index; i++) {
                if (chartData.datasets[i].label === label) {
                  return false;
                }
              }
              return true;
            }
          }
        }
      }
    }
  });
}

// 计算移动平均值
function calculateMovingAverage(data, windowSize) {
  const result = [];
  for (let i = 0; i < data.length; i++) {
    // 如果当前点是 null（静音），直接返回 null
    if (data[i] === null) {
      result.push(null);
      continue;
    }

    const start = Math.max(0, i - windowSize + 1);
    const slice = data.slice(start, i + 1);
    // 过滤掉 null 值
    const validValues = slice.filter(v => v !== null);
    if (validValues.length === 0) {
      result.push(null);
    } else {
      const sum = validValues.reduce((a, b) => a + parseFloat(b), 0);
      result.push((sum / validValues.length).toFixed(1));
    }
  }
  return result;
}

function getElapsedSeconds() {
  if (!recordingStartTime) {
    recordingStartTime = Date.now();
  }
  return (Date.now() - recordingStartTime) / 1000;
}

function getTimeIndex(elapsedSeconds) {
  return Math.floor(elapsedSeconds / intervalSeconds);
}

function getLabelForIndex(index) {
  return (index * intervalSeconds).toFixed(1) + 's';
}

function ensureWindowForIndex(timeIndex) {
  const maxDataPoints = getMaxDataPoints();

  if (chartData.labels.length === 0) {
    windowStartIndex = Math.max(0, timeIndex - maxDataPoints + 1);
    for (let i = 0; i < maxDataPoints; i++) {
      const labelIndex = windowStartIndex + i;
      chartData.labels.push(getLabelForIndex(labelIndex));
      chartData.male.push(null);
      chartData.female.push(null);
      f0Data.labels.push(getLabelForIndex(labelIndex));
      f0Data.f0.push(null);
    }
    return;
  }

  const windowEndIndex = windowStartIndex + maxDataPoints - 1;
  if (timeIndex <= windowEndIndex) {
    return;
  }

  const newStartIndex = timeIndex - maxDataPoints + 1;
  const shiftCount = newStartIndex - windowStartIndex;
  if (shiftCount <= 0) return;

  if (shiftCount >= maxDataPoints) {
    chartData.labels = [];
    chartData.male = [];
    chartData.female = [];
    f0Data.labels = [];
    f0Data.f0 = [];
    windowStartIndex = newStartIndex;
    for (let i = 0; i < maxDataPoints; i++) {
      const labelIndex = windowStartIndex + i;
      chartData.labels.push(getLabelForIndex(labelIndex));
      chartData.male.push(null);
      chartData.female.push(null);
      f0Data.labels.push(getLabelForIndex(labelIndex));
      f0Data.f0.push(null);
    }
    return;
  }

  chartData.labels.splice(0, shiftCount);
  chartData.male.splice(0, shiftCount);
  chartData.female.splice(0, shiftCount);
  f0Data.labels.splice(0, shiftCount);
  f0Data.f0.splice(0, shiftCount);

  windowStartIndex = newStartIndex;
  for (let i = 0; i < shiftCount; i++) {
    const labelIndex = windowEndIndex + 1 + i;
    chartData.labels.push(getLabelForIndex(labelIndex));
    chartData.male.push(null);
    chartData.female.push(null);
    f0Data.labels.push(getLabelForIndex(labelIndex));
    f0Data.f0.push(null);
  }
}

// 更新图表数据 - 使用模型预测结果
function updateChart(maleProb, femaleProb, timeIndexOverride) {
  const timeIndex =
    typeof timeIndexOverride === 'number'
      ? timeIndexOverride
      : getTimeIndex(getElapsedSeconds());
  ensureWindowForIndex(timeIndex);
  const offset = timeIndex - windowStartIndex;
  if (offset >= 0 && offset < chartData.labels.length) {
    chartData.male[offset] = maleProb;
    chartData.female[offset] = femaleProb;
  }

  // 先进行 LOGIT 变换（在移动平均之前）
  const logitTransform = (val) => {
    if (val === null) return null;
    return logit(parseFloat(val));
  };

  const maleLogitData = chartData.male.map(v => logitTransform(v));
  const femaleLogitData = chartData.female.map(v => logitTransform(v));

  // 获取移动平均窗口大小（秒）
  const avgWindowSeconds = 3;
  const windowSize = Math.max(1, Math.round(avgWindowSeconds / intervalSeconds));

  // 在 LOGIT 空间计算移动平均值
  const maleAvgLogit = calculateMovingAverage(maleLogitData, windowSize);
  const femaleAvgLogit = calculateMovingAverage(femaleLogitData, windowSize);

  // 更新图表
  genderChart.data.labels = chartData.labels;
  genderChart.data.datasets[0].data = maleLogitData;
  genderChart.data.datasets[1].data = femaleLogitData;
  genderChart.data.datasets[2].data = maleAvgLogit;
  genderChart.data.datasets[3].data = femaleAvgLogit;
  // 目标线：使用动态配置的目标值
  const { target } = getGenderLogitRange();
  const dataLength = Math.max(2, chartData.labels.length);
  genderChart.data.datasets[4].data = Array(dataLength).fill(target);
  genderChart.update('none');

  // 更新实时显示
  if (maleProb !== null && femaleProb !== null) {
    updateRealtimeDisplay(maleProb, femaleProb);
  }

  // 检查阈值报警
  checkAlertThreshold(femaleProb);

  // 同步画中画图表
  syncPiPCharts();
}

// 更新 F0 图表
function updateF0Chart(f0, timeIndexOverride) {
  if (!f0Chart || !f0Value) return;

  const timeIndex =
    typeof timeIndexOverride === 'number'
      ? timeIndexOverride
      : getTimeIndex(getElapsedSeconds());
  ensureWindowForIndex(timeIndex);
  const offset = timeIndex - windowStartIndex;
  if (offset >= 0 && offset < f0Data.labels.length) {
    f0Data.f0[offset] = f0 > 0 ? f0.toFixed(0) : null;
  }

  f0Chart.data.labels = f0Data.labels;
  f0Chart.data.datasets[0].data = f0Data.f0;

  const ranges = getF0RangeLines(f0Data.f0.length);
  f0Chart.data.datasets[1].data = ranges.maleMax;
  f0Chart.data.datasets[2].data = ranges.maleMin;
  f0Chart.data.datasets[3].data = ranges.femaleMax;
  f0Chart.data.datasets[4].data = ranges.femaleMin;

  // F0目标线（索引5）
  const dataLength = Math.max(2, f0Data.labels.length);
  f0Chart.data.datasets[5].data = Array(dataLength).fill(f0TargetValue);

  f0Chart.update('none');

  // 更新 F0 显示
  f0Value.textContent = f0 > 0 ? f0.toFixed(1) : '--';

  // 根据F0值设置颜色
  if (f0 > 0) {
    // 男性范围: 85-180 Hz
    // 女性范围: 165-255 Hz
    // 重合部分: 165-180 Hz
    if (f0 >= 165 && f0 <= 180) {
      // 重合区域：紫色
      f0Value.style.color = '#a855f7';
    } else if (f0 >= 85 && f0 < 165) {
      // 男性范围：蓝色
      f0Value.style.color = '#3b82f6';
    } else if (f0 > 180 && f0 <= 255) {
      // 女性范围：粉色
      f0Value.style.color = '#ec4899';
    } else {
      // 超出范围：绿色（默认）
      f0Value.style.color = '#10b981';
    }
  } else {
    f0Value.style.color = '#2563eb';
  }

  // 同步画中画图表
  syncPiPCharts();
}

// 重置图表数据
function resetChart() {
  analysisCount = 0; // 重置分析计数
  alertConsecutiveCount = 0; // 重置报警计数
  windowStartIndex = 0;
  chartData = {
    labels: [],
    male: [],
    female: []
  };
  f0Data = {
    labels: [],
    f0: []
  };
  recordingStartTime = null;
  genderChart.data.labels = [];
  genderChart.data.datasets[0].data = [];
  genderChart.data.datasets[1].data = [];
  genderChart.data.datasets[2].data = [];
  genderChart.data.datasets[3].data = [];
  genderChart.data.datasets[4].data = [];
  genderChart.update('none');

  if (f0Chart) {
    f0Chart.data.labels = [];
    f0Chart.data.datasets[0].data = [];
    f0Chart.data.datasets[1].data = [];
    f0Chart.data.datasets[2].data = [];
    f0Chart.data.datasets[3].data = [];
    f0Chart.data.datasets[4].data = [];
    f0Chart.data.datasets[5].data = [];
    f0Chart.update('none');
  }

  if (f0Value) f0Value.textContent = '--';

  // 重置画中画图表
  resetPiPCharts();
}

async function loadModel() {
  setStatus('加载模型中…');
  const modelParts = [
    'model.onnx.gz.part00',
    'model.onnx.gz.part01',
    'model.onnx.gz.part02'
  ];
  let totalLength = 0;
  const partBuffers = [];
  for (const partUrl of modelParts) {
    const response = await fetch(partUrl);
    if (!response.ok) {
      throw new Error(`模型分片下载失败: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    partBuffers.push(buffer);
    totalLength += buffer.byteLength;
  }

  if (!('DecompressionStream' in window)) {
    throw new Error('当前浏览器不支持模型解压');
  }

  const combined = new Uint8Array(totalLength);
  let offset = 0;
  for (const buffer of partBuffers) {
    combined.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  }

  const compressedBlob = new Blob([combined]);
  const decompressedStream = compressedBlob.stream().pipeThrough(new DecompressionStream('gzip'));
  const decompressedResponse = new Response(decompressedStream);
  const modelArrayBuffer = await decompressedResponse.arrayBuffer();
  session = await ort.InferenceSession.create(modelArrayBuffer);
  // 标记模型已加载过（用于后续访问时显示不同的加载提示）
  localStorage.setItem('modelLoadedBefore', 'true');
  setStatus('');
  recordBtn.disabled = false;
  // 模型加载后启用画中画按钮
  if (isPiPSupported()) {
    pipBtn.disabled = false;
    pipBtn.title = '';
  } else {
    pipBtn.disabled = false;
    pipBtn.title = '平台不支持画中画功能';
  }
}

// 从 AudioBuffer 转换为 Float32Array（重采样到 16kHz）
function audioBufferToFloat32(audioBuffer) {
  const offlineCtx = new OfflineAudioContext(
    1,
    Math.ceil(audioBuffer.duration * TARGET_SAMPLE_RATE),
    TARGET_SAMPLE_RATE
  );
  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start();
  return offlineCtx.startRendering().then(rendered => rendered.getChannelData(0));
}

// 实时分析录音数据
function padOrTrimFrames(melTensor) {
  const frameCount = melTensor.shape[1];
  if (frameCount === TARGET_FRAMES) {
    return melTensor;
  }
  if (frameCount > TARGET_FRAMES) {
    return melTensor.slice([0, 0], [MEL_BINS, TARGET_FRAMES]);
  }
  const padAmount = TARGET_FRAMES - frameCount;
  const padding = [[0, 0], [0, padAmount]];
  return melTensor.pad(padding);
}

// Helper function to create mel weight matrix
function createMelWeightMatrix() {
  const numSpectrogramBins = FFT_LENGTH / 2 + 1;

  // Convert to mel scale
  const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
  const melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);

  const melMin = hzToMel(MEL_FMIN);
  const melMax = hzToMel(MEL_FMAX);
  const melStep = (melMax - melMin) / (MEL_BINS + 1);

  const melEdges = [];
  for (let i = 0; i < MEL_BINS + 2; i++) {
    melEdges.push(melMin + i * melStep);
  }

  const hzEdges = melEdges.map(mel => melToHz(mel));
  const binEdges = hzEdges.map(hz => Math.floor((numSpectrogramBins - 1) * hz / (TARGET_SAMPLE_RATE / 2)));

  const matrix = [];
  for (let i = 0; i < MEL_BINS; i++) {
    const row = new Array(numSpectrogramBins).fill(0);
    const left = binEdges[i];
    const center = binEdges[i + 1];
    const right = binEdges[i + 2];

    for (let j = left; j <= right; j++) {
      if (j >= 0 && j < numSpectrogramBins) {
        if (j <= center) {
          row[j] = (j - left) / (center - left + 1e-6);
        } else {
          row[j] = (right - j) / (right - center + 1e-6);
        }
      }
    }
    matrix.push(row);
  }

  // Create transposed matrix directly (numSpectrogramBins x MEL_BINS)
  const transposedMatrix = [];
  for (let j = 0; j < numSpectrogramBins; j++) {
    const row = [];
    for (let i = 0; i < MEL_BINS; i++) {
      row.push(matrix[i][j]);
    }
    transposedMatrix.push(row);
  }

  return tf.tensor2d(transposedMatrix, [numSpectrogramBins, MEL_BINS]);
}

function getMelWeightMatrix() {
  if (!cachedMelWeightMatrix) {
    cachedMelWeightMatrix = tf.keep(createMelWeightMatrix());
  }
  return cachedMelWeightMatrix;
}

function getF0RangeLines(length) {
  if (cachedF0Ranges.length !== length) {
    cachedF0Ranges.length = length;
    cachedF0Ranges.maleMax = new Array(length).fill(180);
    cachedF0Ranges.maleMin = new Array(length).fill(85);
    cachedF0Ranges.femaleMax = new Array(length).fill(255);
    cachedF0Ranges.femaleMin = new Array(length).fill(165);
  }
  return cachedF0Ranges;
}

function buildMelSpectrogram(waveform) {
  return tf.tidy(() => {
    const waveformTensor = tf.tensor1d(waveform);
    const windowFn = tf.signal.hammingWindow;
    const stft = tf.signal.stft(waveformTensor, WIN_LENGTH, HOP_LENGTH, FFT_LENGTH, windowFn);
    const magnitude = tf.abs(stft).square();

    const melWeightMatrix = getMelWeightMatrix();

    const melSpectrogram = tf.matMul(magnitude, melWeightMatrix);
    const logMel = tf.log(melSpectrogram.add(1e-6));
    const transposed = logMel.transpose([1, 0]);
    const mean = transposed.mean(1, true);
    const normalized = transposed.sub(mean);
    return padOrTrimFrames(normalized).expandDims(0);
  });
}

async function analyzeAudioData(audioBuffer) {
  try {
    const waveform = await audioBufferToFloat32(audioBuffer);

    // 如果音频太短（少于3秒），填充零
    const requiredSamples = TARGET_SAMPLE_RATE * 3;
    let paddedWaveform = waveform;
    if (waveform.length < requiredSamples) {
      paddedWaveform = new Float32Array(requiredSamples);
      paddedWaveform.set(waveform);
    } else if (waveform.length > requiredSamples) {
      paddedWaveform = waveform.slice(0, requiredSamples);
    }

    const inputTensor = buildMelSpectrogram(paddedWaveform);
    const inputData = await inputTensor.data();
    inputTensor.dispose();

    const ortInput = new ort.Tensor('float32', inputData, [1, MEL_BINS, TARGET_FRAMES]);
    const outputs = await session.run({ mel: ortInput });
    const logits = outputs.logits.data;

    const logitsTensor = tf.tensor1d(logits);
    const probs = tf.softmax(logitsTensor);
    const probValues = await probs.data();
    logitsTensor.dispose();
    probs.dispose();

    return { male: probValues[0], female: probValues[1] };
  } catch (error) {
    if (window.Sentry && error instanceof Error) {
      window.Sentry.captureException(error);
    }
    console.error('Analysis error:', error);
    return null;
  }
}

// 开始/停止录音
async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  try {
    // 重置图表
    resetChart();

    // 预填充null数据以显示完整的时间跨度
    const maxDataPoints = getMaxDataPoints();
    windowStartIndex = 0;
    for (let i = 0; i < maxDataPoints; i++) {
      chartData.labels.push(getLabelForIndex(i));
      chartData.male.push(null);
      chartData.female.push(null);
      f0Data.labels.push(getLabelForIndex(i));
      f0Data.f0.push(null);
    }

    // 更新图表以显示预填充的数据
    const { target } = getGenderLogitRange();
    genderChart.data.labels = chartData.labels;
    genderChart.data.datasets[0].data = chartData.male.map(() => null);
    genderChart.data.datasets[1].data = chartData.female.map(() => null);
    genderChart.data.datasets[2].data = chartData.male.map(() => null);
    genderChart.data.datasets[3].data = chartData.female.map(() => null);
    genderChart.data.datasets[4].data = Array(chartData.labels.length).fill(target);
    genderChart.update('none');

    if (f0Chart) {
      f0Chart.data.labels = f0Data.labels;
      f0Chart.data.datasets[0].data = f0Data.f0;
      f0Chart.data.datasets[1].data = f0Data.f0.map(() => 180);
      f0Chart.data.datasets[2].data = f0Data.f0.map(() => 85);
      f0Chart.data.datasets[3].data = f0Data.f0.map(() => 255);
      f0Chart.data.datasets[4].data = f0Data.f0.map(() => 165);
      f0Chart.data.datasets[5].data = Array(f0Data.labels.length).fill(f0TargetValue);
      f0Chart.update('none');
    }

    // 重新启用图表动画
    if (genderChart) {
      genderChart.options.animation = { duration: 300 };
    }
    if (f0Chart) {
      f0Chart.options.animation = { duration: 300 };
    }

    // 获取用户设置的分析间隔
    const intervalSeconds = 0.5;
    const analysisInterval = Math.max(100, Math.min(5000, intervalSeconds * 1000)); // 限制在 0.1-5 秒之间

    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    microphone = audioContext.createMediaStreamSource(mediaStream);

    // 使用 ScriptProcessor 或 AudioWorklet 来收集音频数据
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

    audioChunks = [];
    const samplesNeeded = TARGET_SAMPLE_RATE * 3;
    const maxChunks = Math.ceil(samplesNeeded / 4096);

    scriptProcessor.onaudioprocess = (e) => {
      if (!isRecording) return;
      const inputData = e.inputBuffer.getChannelData(0);
      audioChunks.push(new Float32Array(inputData));
      if (audioChunks.length > maxChunks) {
        audioChunks.splice(0, audioChunks.length - maxChunks);
      }
    };

    microphone.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    isRecording = true;
    recordBtn.textContent = '停止录音';
    recordBtn.classList.add('recording');

    // 按用户设置的间隔进行分析
    recordInterval = setInterval(async () => {
      if (audioChunks.length === 0) return;

      const analysisStart = performance.now();
      // 合并最近的音频片段（最多3秒）
      let combined = new Float32Array(samplesNeeded);
      let offset = 0;
      let startIndex = Math.max(0, audioChunks.length - Math.ceil(samplesNeeded / 4096));

      for (let i = startIndex; i < audioChunks.length && offset < samplesNeeded; i++) {
        const chunk = audioChunks[i];
        const toCopy = Math.min(chunk.length, samplesNeeded - offset);
        combined.set(chunk.subarray(0, toCopy), offset);
        offset += toCopy;
      }

      // 创建 AudioBuffer 用于分析
      const audioBufferForAnalysis = audioContext.createBuffer(1, offset, TARGET_SAMPLE_RATE);
      audioBufferForAnalysis.getChannelData(0).set(combined.subarray(0, offset));

      // 静音检测（使用 RMS）
      const waveformForAnalysis = combined.subarray(0, offset);
      const rms = calculateRMS(waveformForAnalysis);
      const hasVoice = rms > 0.01; // 阈值可调

      // 只在有语音时进行模型预测和 F0 检测
      if (hasVoice) {
        const scores = await analyzeAudioData(audioBufferForAnalysis);
        const timeIndex = getTimeIndex(getElapsedSeconds());
        if (scores) {
          // 使用模型预测结果更新图表
          updateChart(scores.male, scores.female, timeIndex);
        }

        const f0 = detectF0(waveformForAnalysis, TARGET_SAMPLE_RATE);
        updateF0Chart(f0, timeIndex);
      } else {
        // 静音时添加空数据点
        const timeIndex = getTimeIndex(getElapsedSeconds());
        updateChart(null, null, timeIndex);
        updateF0Chart(0, timeIndex);
      }

      const analysisDuration = performance.now() - analysisStart;
      updatePerformanceWarning(hasVoice, analysisDuration, analysisInterval);

      // 两个图表都更新后，递增分析计数
      analysisCount++;
    }, analysisInterval);

  } catch (error) {
    if (window.Sentry && error instanceof Error) {
      window.Sentry.captureException(error);
    }
    const msg = error.message || String(error);
    if (msg.includes('Requested device not found')) {
      alert('未找到麦克风设备');
    } else if (msg.includes('NotAllowedError') || msg.includes('Permission denied')) {
      alert('麦克风权限被拒绝');
    } else {
      alert('启动录音失败: ' + msg);
    }
  }
}

function stopRecording() {
  isRecording = false;

  // 重置报警计数
  alertConsecutiveCount = 0;
  perfOverrunCount = 0;

  if (recordInterval) {
    clearInterval(recordInterval);
    recordInterval = null;
  }

  if (scriptProcessor) {
    scriptProcessor.disconnect();
    microphone.disconnect();
    scriptProcessor = null;
  }

  if (audioContext && audioContext.state !== 'closed') {
    audioContext.close();
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
  audioChunks = [];

  recordBtn.textContent = '开始录音';
  recordBtn.classList.remove('recording');

  // 停止后禁用图表动画，防止滚动效果
  if (genderChart) {
    genderChart.options.animation = false;
    genderChart.update();
  }
  if (f0Chart) {
    f0Chart.options.animation = false;
    f0Chart.update();
  }
}

// 录音按钮事件
recordBtn.addEventListener('click', () => {
  if (!isRecording && typeof gtag === 'function') {
    gtag('event', 'start_recording');
  }
  toggleRecording().catch((err) => {
    if (window.Sentry && err instanceof Error) {
      window.Sentry.captureException(err);
    }
    console.error(err);
  });
});

// 提词相关变量
const PROMPT_STORAGE_KEY = 'userPromptText';
let initialPromptValue = ''; // 记录初始加载时的值，用于判断是否真的改变了

// 加载提词内容（优先从 localStorage，否则从 JSON 随机加载）
async function loadPrompt() {
  const savedPrompt = localStorage.getItem(PROMPT_STORAGE_KEY);
  if (savedPrompt) {
    promptText.value = savedPrompt;
    initialPromptValue = savedPrompt;
  } else {
    await loadRandomStory();
  }
}

// 重置提词为随机内容
async function resetPrompt() {
  localStorage.removeItem(PROMPT_STORAGE_KEY);
  await loadRandomStory();
}

// 提词框失去焦点时检查是否改变并保存
promptText.addEventListener('blur', () => {
  const currentValue = promptText.value;
  // 如果内容确实改变了，保存到 localStorage
  if (currentValue !== initialPromptValue) {
    localStorage.setItem(PROMPT_STORAGE_KEY, currentValue);
    initialPromptValue = currentValue;
  }
});

// 重置提词按钮事件
const resetPromptBtn = document.getElementById('resetPromptBtn');
resetPromptBtn.addEventListener('click', () => {
  resetPrompt();
});

// 输入验证函数
function validateInput(value, rules) {
  const numValue = parseFloat(value);

  if (isNaN(numValue)) {
    return { valid: false, message: '请输入有效的数字' };
  }

  if (rules.min !== undefined && numValue < rules.min) {
    return { valid: false, message: `值不能小于 ${rules.min}` };
  }

  if (rules.max !== undefined && numValue > rules.max) {
    return { valid: false, message: `值不能大于 ${rules.max}` };
  }

  if (rules.minExclusive !== undefined && numValue <= rules.minExclusive) {
    return { valid: false, message: `值必须大于 ${rules.minExclusive}` };
  }

  if (rules.custom && !rules.custom(numValue)) {
    return { valid: false, message: rules.customMessage || '输入值无效' };
  }

  return { valid: true, value: numValue };
}

const inputStorageKeyMap = {
  f0TargetInput: 'f0TargetValue',
  genderMinInput: 'genderMinPercent',
  genderMaxInput: 'genderMaxPercent',
  genderTargetInput: 'genderTargetPercent',
  timeSpanInput: 'timeSpanSeconds',
  alertMinInput: 'alertMinPercent',
  alertMaxInput: 'alertMaxPercent',
  alertDurationInput: 'alertDurationSeconds'
};

// 显示错误提示
function showInputError(input, message) {
  input.classList.add('input-error');
  alert(message);
  input.classList.remove('input-error');
  // 恢复为localStorage中的有效值
  const storageKey = inputStorageKeyMap[input.id];
  input.value = (storageKey ? localStorage.getItem(storageKey) : null) ||
                input.defaultValue ||
                input.value;
}

// F0目标值输入框事件
const f0TargetInput = document.getElementById('f0TargetInput');
// 初始化输入框值为localStorage中的值
f0TargetInput.value = f0TargetValue;

f0TargetInput.addEventListener('change', (e) => {
  const validation = validateInput(e.target.value, {
    minExclusive: 0
  });

  if (!validation.valid) {
    showInputError(e.target, validation.message);
    return;
  }

  f0TargetValue = validation.value;
  localStorage.setItem('f0TargetValue', validation.value.toString());
  updateGenderChartScale();
});

// 性别置信度范围输入框事件
const genderMinInput = document.getElementById('genderMinInput');
const genderMaxInput = document.getElementById('genderMaxInput');
const genderTargetInput = document.getElementById('genderTargetInput');

// 初始化输入框值为localStorage中的值
genderMinInput.value = genderMinPercent;
genderMaxInput.value = genderMaxPercent;
genderTargetInput.value = genderTargetPercent;

// 时间跨度输入框事件
const timeSpanInput = document.getElementById('timeSpanInput');
// 初始化输入框值为localStorage中的值
timeSpanInput.value = timeSpanSeconds;

timeSpanInput.addEventListener('change', (e) => {
  const validation = validateInput(e.target.value, {
    minExclusive: 0
  });

  if (!validation.valid) {
    showInputError(e.target, validation.message);
    return;
  }

  const value = validation.value;
  // 检查数值是否真的改变了
  if (value !== timeSpanSeconds) {
    localStorage.setItem('timeSpanSeconds', value.toString());
  }
});

// 重置参数按钮事件
const resetParamsBtn = document.getElementById('resetParamsBtn');
resetParamsBtn.addEventListener('click', () => {
  // 默认值
  const defaults = {
    f0Target: 200,
    genderMin: 85,
    genderMax: 99.8,
    genderTarget: 98.2,
    timeSpan: 30,
    alertMin: 20,
    alertMax: 98.2,
    alertDuration: 5
  };

  // 清除localStorage（不重置报警启用状态）
  localStorage.removeItem('f0TargetValue');
  localStorage.removeItem('genderMinPercent');
  localStorage.removeItem('genderMaxPercent');
  localStorage.removeItem('genderTargetPercent');
  localStorage.removeItem('timeSpanSeconds');
  localStorage.removeItem('alertMinPercent');
  localStorage.removeItem('alertMaxPercent');
  localStorage.removeItem('alertDurationSeconds');
  // 重置变量（不重置报警启用状态）
  f0TargetValue = defaults.f0Target;
  genderMinPercent = defaults.genderMin;
  genderMaxPercent = defaults.genderMax;
  genderTargetPercent = defaults.genderTarget;
  // timeSpanSeconds = defaults.timeSpan;
  alertMinPercent = defaults.alertMin;
  alertMaxPercent = defaults.alertMax;
  alertDurationSeconds = defaults.alertDuration;

  // 更新输入框值
  f0TargetInput.value = f0TargetValue;
  genderMinInput.value = genderMinPercent;
  genderMaxInput.value = genderMaxPercent;
  genderTargetInput.value = genderTargetPercent;
  timeSpanInput.value = defaults.timeSpan;
  alertMinInput.value = alertMinPercent;
  alertMaxInput.value = alertMaxPercent;
  alertDurationInput.value = alertDurationSeconds;

  // 更新启用按钮状态
  updateAlertButton();

  // 更新所有图表
  updateGenderChartScale();
});

genderMinInput.addEventListener('change', (e) => {
  const validation = validateInput(e.target.value, {
    min: 50,
    max: 99.9
  });

  if (!validation.valid) {
    showInputError(e.target, validation.message);
    return;
  }

  genderMinPercent = validation.value;
  localStorage.setItem('genderMinPercent', validation.value.toString());
  updateGenderChartScale();
});

genderMaxInput.addEventListener('change', (e) => {
  const validation = validateInput(e.target.value, {
    min: 50,
    max: 99.9
  });

  if (!validation.valid) {
    showInputError(e.target, validation.message);
    return;
  }

  genderMaxPercent = validation.value;
  localStorage.setItem('genderMaxPercent', validation.value.toString());
  updateGenderChartScale();
});

genderTargetInput.addEventListener('change', (e) => {
  const value = parseFloat(e.target.value);

  if (isNaN(value)) {
    showInputError(e.target, '请输入有效的数字');
    return;
  }

  // 检查目标值是否在上下界内
  if (value < genderMinPercent || value > genderMaxPercent) {
    showInputError(e.target, `目标值必须在下界(${genderMinPercent}%)和上界(${genderMaxPercent}%)之间`);
    return;
  }

  genderTargetPercent = value;
  localStorage.setItem('genderTargetPercent', value.toString());
  updateGenderChartScale();
});

// 报警参数输入框事件
const alertMinInput = document.getElementById('alertMinInput');
const alertMaxInput = document.getElementById('alertMaxInput');
const alertDurationInput = document.getElementById('alertDurationInput');
const enableAlertBtn = document.getElementById('enableAlertBtn');

// 初始化输入框值为localStorage中的值
alertMinInput.value = alertMinPercent;
alertMaxInput.value = alertMaxPercent;
alertDurationInput.value = alertDurationSeconds;

// 更新启用按钮状态
function updateAlertButton() {
  if (alertEnabled) {
    enableAlertBtn.textContent = '已启用';
    enableAlertBtn.style.background = '#10b981';
    enableAlertBtn.style.color = 'white';
    enableAlertBtn.style.border = '1px solid #10b981';
  } else {
    enableAlertBtn.textContent = '启用';
    enableAlertBtn.style.background = '#f1f5f9';
    enableAlertBtn.style.color = 'inherit';
    enableAlertBtn.style.border = '1px solid #cbd5e1';
  }
}

// 初始化按钮状态
updateAlertButton();

// 启用/禁用报警按钮
enableAlertBtn.addEventListener('click', () => {
  if (!alertEnabled) {
    // 检查 Notification API 是否存在
    if (typeof Notification === 'undefined') {
      alert('无法请求通知权限\n\niOS 需要将网页添加至主屏幕才能启用通知');
      return;
    }
    // 请求通知权限
    if (Notification.permission === 'default') {
      Notification.requestPermission().then(permission => {
        if (permission === 'granted') {
          alertEnabled = true;
          localStorage.setItem('alertEnabled', 'true');
          updateAlertButton();
        } else {
          alert('请允许通知权限以使用报警功能');
        }
      }).catch(() => {
        alert('无法请求通知权限\n\niOS 需要将网页添加至主屏幕才能启用通知');
      });
    } else if (Notification.permission === 'granted') {
      alertEnabled = true;
      localStorage.setItem('alertEnabled', 'true');
      updateAlertButton();
    } else {
      alert('请允许通知权限以使用报警功能');
    }
  } else {
    alertEnabled = false;
    localStorage.setItem('alertEnabled', 'false');
    updateAlertButton();
  }
});

// 报警参数输入事件
alertMinInput.addEventListener('change', (e) => {
  const validation = validateInput(e.target.value, {
    min: 0,
    max: 99.9
  });

  if (!validation.valid) {
    showInputError(e.target, validation.message);
    return;
  }

  // 检查最小值是否小于最大值
  if (validation.value >= alertMaxPercent) {
    showInputError(e.target, `最小值必须小于最大值(${alertMaxPercent}%)`);
    return;
  }

  alertMinPercent = validation.value;
  localStorage.setItem('alertMinPercent', validation.value.toString());
  alertConsecutiveCount = 0; // 重置计数
});

alertMaxInput.addEventListener('change', (e) => {
  const validation = validateInput(e.target.value, {
    min: 0,
    max: 99.9
  });

  if (!validation.valid) {
    showInputError(e.target, validation.message);
    return;
  }

  // 检查最大值是否大于最小值
  if (validation.value <= alertMinPercent) {
    showInputError(e.target, `最大值必须大于最小值(${alertMinPercent}%)`);
    return;
  }

  alertMaxPercent = validation.value;
  localStorage.setItem('alertMaxPercent', validation.value.toString());
  alertConsecutiveCount = 0; // 重置计数
});

alertDurationInput.addEventListener('change', (e) => {
  const validation = validateInput(e.target.value, {
    minExclusive: 0
  });

  if (!validation.valid) {
    showInputError(e.target, validation.message);
    return;
  }

  alertDurationSeconds = validation.value;
  localStorage.setItem('alertDurationSeconds', validation.value.toString());
  alertConsecutiveCount = 0; // 重置计数
});

// 获取当前性别置信度的logit范围
function getGenderLogitRange() {
  return {
    min: logit(genderMinPercent / 100),
    max: logit(genderMaxPercent / 100),
    target: logit(genderTargetPercent / 100)
  };
}

// 计算刻度步长（生成5个均匀分布的刻度）
function calculateGenderStepSize(logitMin, logitMax) {
  return (logitMax - logitMin) / 4;
}

// 更新性别图表刻度
function updateGenderChartScale() {
  const { min, max, target } = getGenderLogitRange();
  const stepSize = calculateGenderStepSize(min, max);

  // 更新主图表
  if (genderChart) {
    genderChart.options.scales.y.min = min;
    genderChart.options.scales.y.max = max;
    genderChart.options.scales.y.ticks.stepSize = stepSize;
    // 更新目标线数据
    const dataLength = Math.max(2, chartData.labels.length);
    genderChart.data.datasets[4].data = Array(dataLength).fill(target);
    genderChart.update('none');
  }

  // 更新F0图表的目标线
  if (f0Chart) {
    const f0DataLength = Math.max(2, f0Data.labels.length);
    f0Chart.data.datasets[5].data = Array(f0DataLength).fill(f0TargetValue);
    f0Chart.update('none');
  }

  // 更新画中画图表
  if (pipGenderChart) {
    pipGenderChart.options.scales.y.min = min;
    pipGenderChart.options.scales.y.max = max;
    pipGenderChart.options.scales.y.ticks.stepSize = stepSize;
    // 更新目标线数据
    const pipDataLength = Math.max(2, pipGenderChart.data.labels.length);
    pipGenderChart.data.datasets[2].data = Array(pipDataLength).fill(target);
    // 更新F0轴刻度以保持目标线对齐
    updatePiPF0Scale();
  }
}

// 更新画中画F0轴刻度
function updatePiPF0Scale() {
  if (!pipWindow || pipWindow.closed || !pipGenderChart) return;

  const { min, max, target } = getGenderLogitRange();
  const relativePosition = (target - min) / (max - min);

  const f0Range = 250;
  const f0Min = f0TargetValue - relativePosition * f0Range;
  const f0Max = f0Min + f0Range;

  pipGenderChart.options.scales.y1.min = f0Min;
  pipGenderChart.options.scales.y1.max = f0Max;
  pipGenderChart.update('none');
}

// 加载故事文本
let cachedStoryText = null;

async function loadRandomStory() {
  try {
    if (!cachedStoryText) {
      const response = await fetch('story.txt');
      if (!response.ok) {
        throw new Error(`故事加载失败: ${response.status}`);
      }
      cachedStoryText = await response.text();
    }
    promptText.value = cachedStoryText;
    initialPromptValue = promptText.value;
  } catch (err) {
    if (window.Sentry && err instanceof Error) {
      window.Sentry.captureException(err);
    }
    console.error('故事加载失败:', err);
    promptText.value = `${err.message}`;
    initialPromptValue = promptText.value;
  }
}

loadPrompt().catch((err) => {
  if (window.Sentry && err instanceof Error) {
    window.Sentry.captureException(err);
  }
  console.error(err);
  setStatus(`提词加载失败: ${err?.message || String(err)}`);
});

try {
  // 立即初始化图表
  initChart();
} catch (err) {
  if (window.Sentry && err instanceof Error) {
    window.Sentry.captureException(err);
  }
  console.error(err);
  setStatus(`图表加载失败: ${err?.message || String(err)}`);
}

if (f0ChartCanvas) {
  try {
    initF0Chart();
  } catch (err) {
    if (window.Sentry && err instanceof Error) {
      window.Sentry.captureException(err);
    }
    console.error(err);
    setStatus(`图表加载失败: ${err?.message || String(err)}`);
  }
}

loadModel().catch((err) => {
  const errMsg = err?.message || String(err);
  if (errMsg.toLowerCase().includes('out of memory') || errMsg.toLowerCase().includes('oom')) {
    setStatus('内存不足');
    if (window.Sentry) {
      window.Sentry.captureMessage('oom', 'warning');
    }
  } else {
    if (window.Sentry && err instanceof Error) {
      window.Sentry.captureException(err);
    }
    console.error(err);
    setStatus(`模型加载失败: ${errMsg}`);
  }
});

// ========== 画中画功能 ==========

// 检查浏览器是否支持 Document Picture-in-Picture API
function isPiPSupported() {
  return 'documentPictureInPicture' in window && 'requestWindow' in documentPictureInPicture;
}

// 创建画中画窗口内容
function createPiPContent() {
  const container = document.createElement('div');
  container.className = 'pip-container';

  container.innerHTML = `
    <div class="pip-chart-wrapper pip-gender-chart">
      <canvas id="pipGenderChart"></canvas>
    </div>
  `;

  return container;
}

// 初始化画中画图表
function initPiPCharts() {
  const pipGenderCanvas = pipWindow.document.getElementById('pipGenderChart');

  // 注入辅助函数到画中画窗口
  const helperScript = pipWindow.document.createElement('script');
  helperScript.textContent = `
    // LOGIT 变换
    function logit(p) {
      p = Math.max(0.001, Math.min(0.999, p));
      return Math.log(p / (1 - p));
    }

    // Sigmoid 变换
    function sigmoid(x) {
      return 1 / (1 + Math.exp(-x));
    }

    // 从 logit 值获取百分比显示
    function logitToPercent(logitVal) {
      const prob = sigmoid(logitVal);
      return (prob * 100).toFixed(1) + '%';
    }
  `;
  pipWindow.document.head.appendChild(helperScript);

  // 复制 Chart.js 到画中画窗口
  const chartScript = pipWindow.document.createElement('script');
  chartScript.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js';
  pipWindow.document.head.appendChild(chartScript);

  chartScript.onload = () => {
    // 等待 Chart.js 加载后初始化图表
    setTimeout(() => {
      // 获取当前配置的logit范围
      const { min: genderMin, max: genderMax, target: genderTarget } = getGenderLogitRange();
      const relativePosition = (genderTarget - genderMin) / (genderMax - genderMin);

      // F0目标值设为f0TargetValue，需要计算y1轴的min/max
      // 使f0TargetValue在相同的相对位置
      // 保持y1轴的范围与原来相似（约250的范围）
      const f0Range = 250;
      const f0Min = f0TargetValue - relativePosition * f0Range;
      const f0Max = f0Min + f0Range;

      const stepSize = calculateGenderStepSize(genderMin, genderMax);

      // 初始化性别图表
      pipGenderChart = new pipWindow.Chart(pipGenderCanvas, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: '男性平均值',
              data: [],
              borderColor: '#1d4ed8',
              backgroundColor: 'rgba(29, 78, 216, 0.1)',
              fill: true,
              tension: 0.4,
              pointRadius: 0,
              yAxisID: 'y'
            },
            {
              label: '女性平均值',
              data: [],
              borderColor: '#be185d',
              backgroundColor: 'rgba(190, 24, 93, 0.1)',
              fill: true,
              tension: 0.4,
              pointRadius: 0,
              yAxisID: 'y'
            },
            {
              label: '目标',
              data: [genderTarget, genderTarget],
              borderColor: '#a855f7',
              borderWidth: 2,
              backgroundColor: 'transparent',
              fill: false,
              pointRadius: 0,
              borderDash: [],
              spanGaps: true,
              yAxisID: 'y'
            },
            {
              label: 'F0',
              data: [],
              borderColor: '#10b981',
              backgroundColor: 'transparent',
              fill: false,
              tension: 0.4,
              pointRadius: 0,
              borderWidth: 2,
              yAxisID: 'y1'
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: {
            duration: 300
          },
          scales: {
            x: {
              display: false
            },
            y: {
              type: 'linear',
              display: true,
              position: 'left',
              min: genderMin,
              max: genderMax,
              grid: {
                drawOnChartArea: true
              },
              ticks: {
                stepSize: stepSize,
                callback: function(value, index, values) {
                  return logitToPercent(value);
                },
                font: { size: 9 },
                color: '#64748b',
                padding: 0,
                mirror: true
              },
              border: {
                display: false
              }
            },
            y1: {
              type: 'linear',
              display: true,
              position: 'right',
              min: f0Min,
              max: f0Max,
              grid: {
                drawOnChartArea: false
              },
              ticks: {
                font: { size: 9 },
                color: '#10b981',
                padding: 0,
                mirror: true
              },
              border: {
                display: false
              }
            }
          },
          layout: {
            padding: 0
          },
          plugins: {
            legend: {
              display: false
            },
            tooltip: {
              enabled: false
            }
          }
        }
      });

      // 同步当前数据到画中画图表
      syncPiPCharts();
    }, 100);
  };

}

// 同步画中画图表数据
function syncPiPCharts() {
  if (!pipWindow || pipWindow.closed) return;

  // 同步性别图表
  if (pipGenderChart) {
    const logitTransform = (val) => {
      if (val === null) return null;
      return logit(parseFloat(val));
    };

    const maleLogitData = chartData.male.map(v => logitTransform(v));
    const femaleLogitData = chartData.female.map(v => logitTransform(v));

    const avgWindowSeconds = 3;
    const intervalSeconds = 0.5;
    const windowSize = Math.max(1, Math.round(avgWindowSeconds / intervalSeconds));

    const maleAvgLogit = calculateMovingAverage(maleLogitData, windowSize);
    const femaleAvgLogit = calculateMovingAverage(femaleLogitData, windowSize);

    pipGenderChart.data.labels = chartData.labels;
    // 数据集0: 男性平均值
    pipGenderChart.data.datasets[0].data = maleAvgLogit;
    // 数据集1: 女性平均值
    pipGenderChart.data.datasets[1].data = femaleAvgLogit;

    const { target } = getGenderLogitRange();
    const maxDataPoints = getMaxDataPoints();
    // 数据集2: 目标线，始终覆盖完整时间跨度
    pipGenderChart.data.datasets[2].data = Array(maxDataPoints).fill(target);

    // 数据集3: F0数据
    pipGenderChart.data.datasets[3].data = f0Data.f0;

    pipGenderChart.update('none');
  }
}

// 进入画中画模式
async function enterPiP() {
  try {
    pipWindow = await documentPictureInPicture.requestWindow({
      width: 360,
      height: 300
    });

    // 复制样式
    const styleLink = pipWindow.document.createElement('link');
    styleLink.rel = 'stylesheet';
    styleLink.href = 'style.css';
    pipWindow.document.head.appendChild(styleLink);

    // 添加内容
    const content = createPiPContent();
    pipWindow.document.body.appendChild(content);
    pipWindow.document.body.style.margin = '0';
    pipWindow.document.body.style.padding = '0';

    // 初始化图表
    initPiPCharts();

    // 监听画中画窗口关闭事件
    pipWindow.addEventListener('pagehide', () => {
      pipGenderChart = null;
      pipWindow = null;
      pipBtn.classList.remove('pip-active');
    });

    pipBtn.classList.add('pip-active');

  } catch (error) {
    if (window.Sentry && error instanceof Error) {
      window.Sentry.captureException(error);
    }
    console.error('Failed to enter PiP:', error);
    if (error.name === 'NotAllowedError') {
      alert('画中画请求被拒绝，请确保在用户手势（如点击）后调用。');
    } else {
      alert('进入画中画失败: ' + error.message);
    }
  }
}

// 退出画中画模式
function exitPiP() {
  if (pipWindow && !pipWindow.closed) {
    pipWindow.close();
  }
  pipGenderChart = null;
  pipWindow = null;
  pipBtn.classList.remove('pip-active');
}

// 画中画按钮点击事件
pipBtn.addEventListener('click', () => {
  if (!isPiPSupported()) {
    alert(pipBtn.title);
    return;
  }
  if (pipWindow && !pipWindow.closed) {
    exitPiP();
  } else {
    enterPiP();
  }
});

// 重置画中画图表
function resetPiPCharts() {
  if (pipWindow && !pipWindow.closed) {
    if (pipGenderChart) {
      pipGenderChart.data.labels = [];
      pipGenderChart.data.datasets.forEach(ds => ds.data = []);
      pipGenderChart.update('none');
    }
  }
}

// ========== 清空数据功能 ==========

// 清空所有站点数据
async function clearAllSiteData() {
  // 确认操作
  const confirmed = confirm('确定要清空所有站点数据吗？\n\n这将：\n• 清除所有本地存储及缓存\n• 下次访问需重新下载文件');
  if (!confirmed) return;

  try {
    // 1. 清空 localStorage
    localStorage.clear();

    // 2. 清空 sessionStorage
    sessionStorage.clear();

    // 3. 清除缓存
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      await Promise.all(cacheNames.map(name => caches.delete(name)));
    }

    // 4. 注销 Service Worker
    if ('serviceWorker' in navigator) {
      const registrations = await navigator.serviceWorker.getRegistrations();
      await Promise.all(registrations.map(reg => reg.unregister()));
    }

    // 5. 显示成功提示并刷新页面
    window.location.reload();

  } catch (error) {
    console.error('清空数据失败:', error);
    alert('清空数据时出错：' + error.message);
  }
}

// 清空数据按钮事件
const clearDataBtn = document.getElementById('clearDataBtn');
if (clearDataBtn) {
  clearDataBtn.addEventListener('click', clearAllSiteData);
}
