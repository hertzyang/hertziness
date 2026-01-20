const APP_CACHE = `app-cache`;

self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter((key) => key !== APP_CACHE)
        .map((key) => caches.delete(key))
    );
    await self.clients.claim();
  })());
});

async function staleWhileRevalidate(request, cacheName, options = {}) {
  const { skipRevalidateIfCached = false } = options;
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);

  if (cached) {
    if (skipRevalidateIfCached) {
      return cached;
    }
    const fetchPromise = fetch(request).then((response) => {
      if (response && (response.ok || response.type === 'opaque')) {
        cache.put(request, response.clone());
      }
      return response;
    });
    fetchPromise.catch(() => {});
    return cached;
  }

  const fetchPromise = fetch(request).then((response) => {
    if (response && (response.ok || response.type === 'opaque')) {
      cache.put(request, response.clone());
    }
    return response;
  });
  return fetchPromise;
}

self.addEventListener('fetch', (event) => {
  const { request } = event;

  if (request.method !== 'GET') return;
  if (!request.url.startsWith('http://') && !request.url.startsWith('https://')) return;

  const url = new URL(request.url);
  const skipRevalidateIfCached =
  //  (url.pathname.includes('.gz.') || url.pathname.endsWith('.gz') || url.pathname.endsWith('.wasm'));
  (url.pathname.includes('model.onnx.gz.'));

  event.respondWith(staleWhileRevalidate(request, APP_CACHE, { skipRevalidateIfCached }));
});
