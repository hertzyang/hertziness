# Hertziness

Hertziness 是一款纯前端的声音性别分析工具，通过浏览器本地运行机器学习模型，实时分析声音的性别特征。

## 特性

- **实时分析** - 录音时实时显示声音性别置信度和基频（F0）
- **本地处理** - 使用 ONNX Runtime Web 在浏览器中运行模型，数据不上传
- **离线可用** - 支持 PWA，安装后可完全离线使用
- **画中画** - 支持 Document Picture-in-Picture API，浮窗显示分析结果
- **阈值报警** - 当声音在设定区间持续超过阈值时发送通知
- **提词功能** - 内置童话故事语料库，方便朗读练习

## 在线使用

访问 [voice.hertz.page](https://voice.hertz.page) 即可开始使用，无需安装。

## 部署说明

本项目使用 `SharedArrayBuffer` 实现多线程推理，部署时需要设置以下 HTTP 响应头：

```
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Opener-Policy: same-origin
```

### Cloudflare Pages

使用项目附带的 `_headers` 文件：

```
/*
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Opener-Policy: same-origin
```


## 其他

《AI 真的太好用了你知道吗》

项目代码每一行均由 AI 编写，绝无人工注水

## 许可证

项目自身代码使用 MIT 许可证进行授权。

## Reference

This project includes third-party software:

### JaesungHuh/voice-gender-classifier
- Repository: https://github.com/JaesungHuh/voice-gender-classifier
- Copyright (c) 2024 Jaesung Huh
- License: MIT


MIT License

Copyright (c) 2024 jaesunghuh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.