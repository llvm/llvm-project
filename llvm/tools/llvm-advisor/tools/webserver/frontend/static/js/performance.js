// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

class PerformanceManager {
  constructor() {
    this.currentViewType = 'time-order';
    this.currentUnit = null;
    this.timeTraceData = null;
    this.runtimeTraceData = null;

    // Interactive state
    this.viewports = {
      timeTrace : {offsetX : 0, scaleX : 1, offsetY : 0, scaleY : 1},
      runtimeTrace : {offsetX : 0, scaleX : 1, offsetY : 0, scaleY : 1}
    };

    // Mouse/touch interaction state
    this.isDragging = false;
    this.lastMousePos = {x : 0, y : 0};
    this.isZooming = false;

    // Search
    this.searchQuery = '';
    this.searchResults = {timeTrace : [], runtimeTrace : []};

    // Constants
    this.FRAME_HEIGHT = 18;
    this.MINIMAP_HEIGHT = 50;
    this.MIN_FRAME_WIDTH_FOR_TEXT = 25;
    this.PADDING = 4;
    this.MIN_ZOOM = 0.1;
    this.MAX_ZOOM = 100;

    this.initializeEventListeners();
  }

  initializeEventListeners() {
    const viewTypeSelector = document.getElementById('view-type-selector-perf');
    if (viewTypeSelector) {
      viewTypeSelector.addEventListener('change', (e) => {
        this.currentViewType = e.target.value;
        this.renderBothViews();
      });
    }

    const refreshBtn = document.getElementById('refresh-performance-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click',
                                  () => { this.loadAllPerformanceData(); });
    }

    const unitSelector = document.getElementById('unit-selector');
    if (unitSelector) {
      unitSelector.addEventListener('change', (e) => {
        this.currentUnit = e.target.value;
        this.loadAllPerformanceData();
      });
    }

    this.initializeSearch();
  }

  initializeSearch() {
    const controlsDiv = document.querySelector(
        '#performance-content .flex.items-center.space-x-4');
    if (controlsDiv && !document.getElementById('perf-search')) {
      const searchDiv = document.createElement('div');
      searchDiv.className = 'flex items-center space-x-2';
      searchDiv.innerHTML = `
                <label for="perf-search" class="text-sm font-medium text-gray-700">Search:</label>
                <input id="perf-search" type="text" placeholder="Function name..." 
                       class="block w-48 px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-llvm-blue focus:border-llvm-blue sm:text-sm">
                <span id="search-results-count" class="text-sm text-gray-500"></span>
            `;
      controlsDiv.appendChild(searchDiv);

      const searchInput = document.getElementById('perf-search');
      searchInput.addEventListener('input', (e) => {
        this.searchQuery = e.target.value.toLowerCase();
        this.performSearch();
      });
    }
  }

  async loadAllPerformanceData() {
    try {
      const params = new URLSearchParams();
      if (this.currentUnit) {
        params.append('unit', this.currentUnit);
      }

      // Load both time-trace and runtime-trace data
      const [timeTraceResponse, runtimeTraceResponse] = await Promise.all([
        this.loadTraceData('time-trace', params),
        this.loadTraceData('runtime-trace', params)
      ]);

      this.timeTraceData = timeTraceResponse;
      this.runtimeTraceData = runtimeTraceResponse;

      // Reset viewports when new data is loaded
      this.resetViewports();

      this.renderBothViews();
      this.updateIndividualStats();
    } catch (error) {
      console.error('Failed to load performance data:', error);
      this.showError('Failed to load performance data: ' + error.message);
    }
  }

  async loadTraceData(traceType, params) {
    const endpoints = {
      'time-order' : `${traceType}/flamegraph`,
      'sandwich' : `${traceType}/sandwich`
    };

    const endpoint = `/api/${endpoints[this.currentViewType]}`;

    console.log(`Loading ${traceType} data from:`, endpoint,
                'params:', params.toString());

    try {
      const response = await fetch(`${endpoint}?${params}`);
      const result = await response.json();

      console.log(`${traceType} response:`, result);

      if (result.success) {
        // Add source marking to help distinguish data
        if (result.data && result.data.samples) {
          result.data.samples.forEach(sample => { sample.source = traceType; });
        }
        if (result.data && result.data.functions) {
          result.data.functions.forEach(func => { func.source = traceType; });
        }
        return result.data;
      } else {
        console.warn(`No ${traceType} data available:`,
                     result.message || result.error);
        return null;
      }
    } catch (error) {
      console.error(`Failed to load ${traceType} data:`, error);
      return null;
    }
  }

  renderBothViews() {
    switch (this.currentViewType) {
    case 'time-order':
      this.renderTimeOrderViews();
      break;
    case 'sandwich':
      this.renderSandwichViews();
      break;
    }
  }

  renderTimeOrderViews() {
    // Create dual time-order layout
    const container = document.getElementById('performance-visualization');
    container.innerHTML = `
            <div class="space-y-6">
                <!-- Time Trace Section -->
                <div class="bg-white rounded-lg shadow-lg overflow-hidden">
                    <div class="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4">
                        <h3 class="text-lg font-semibold text-white flex items-center">
                            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
                            </svg>
                            Compilation Time Trace
                            <div class="ml-auto text-sm font-normal flex items-center space-x-2">
                                <span id="time-trace-stats" class="bg-blue-800 bg-opacity-50 px-2 py-1 rounded">Loading...</span>
                                <span id="time-trace-debug" class="bg-blue-900 bg-opacity-30 px-2 py-1 rounded text-xs" title="Debug info">?</span>
                            </div>
                        </h3>
                    </div>
                    <div id="time-trace-container" class="relative">
                        ${this.createInteractiveCanvasContainer('time-trace')}
                    </div>
                </div>

                <!-- Runtime Trace Section -->
                <div class="bg-white rounded-lg shadow-lg overflow-hidden">
                    <div class="bg-gradient-to-r from-purple-600 to-purple-700 px-6 py-4">
                        <h3 class="text-lg font-semibold text-white flex items-center">
                            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                            </svg>
                            Runtime Offloading Trace
                            <div class="ml-auto text-sm font-normal flex items-center space-x-2">
                                <span id="runtime-trace-stats" class="bg-purple-800 bg-opacity-50 px-2 py-1 rounded">Loading...</span>
                                <span id="runtime-trace-debug" class="bg-purple-900 bg-opacity-30 px-2 py-1 rounded text-xs" title="Debug info">?</span>
                            </div>
                        </h3>
                    </div>
                    <div id="runtime-trace-container" class="relative">
                        ${
        this.createInteractiveCanvasContainer('runtime-trace')}
                    </div>
                </div>
            </div>
        `;

    this.setupInteractiveCanvases();
  }

  createInteractiveCanvasContainer(traceType) {
    return `
            <div class="w-full relative" style="min-height: 450px;">
                <!-- Controls -->
                <div class="absolute top-2 right-2 flex items-center space-x-2 z-10">
                    <button id="${
        traceType}-reset-btn" class="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded border" title="Reset zoom (Double-click frame to fit)">
                        Reset
                    </button>
                    <span id="${
        traceType}-zoom-level" class="text-xs text-gray-600 bg-white bg-opacity-90 px-2 py-1 rounded">100%</span>
                </div>
                
                <!-- Minimap -->
                <canvas id="${traceType}-minimap" width="800" height="${
        this.MINIMAP_HEIGHT}" 
                        class="w-full border-b border-gray-200 cursor-pointer bg-gray-50" title="Click and drag to navigate"></canvas>
                
                <!-- Main canvas -->
                <canvas id="${traceType}-main" width="800" height="350" 
                        class="w-full cursor-grab bg-white" title="Drag to pan, scroll to zoom, double-click frame to fit"></canvas>
                
                <!-- Tooltip -->
                <div id="${
        traceType}-tooltip" class="absolute pointer-events-none bg-gray-900 text-white px-3 py-2 rounded-lg text-sm z-20 hidden shadow-lg">
                    <div class="arrow-down absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                </div>
                
                <!-- Status -->
                <div id="${
        traceType}-status" class="absolute bottom-2 left-2 text-xs text-gray-500 bg-white bg-opacity-90 px-2 py-1 rounded">
                    Loading...
                </div>
                
                <!-- Keyboard shortcuts help -->
                <div class="absolute bottom-2 right-2 text-xs text-gray-400 bg-white bg-opacity-90 px-2 py-1 rounded" title="Keyboard: +/- to zoom, Arrow keys to pan, Escape to reset">
                    ? Keys
                </div>
            </div>
        `;
  }

  setupInteractiveCanvases() {
    ['time-trace', 'runtime-trace'].forEach(
        traceType => { this.setupSingleInteractiveCanvas(traceType); });

    // Add keyboard listeners
    document.addEventListener('keydown', this.handleKeyDown.bind(this));
    document.addEventListener('keyup', this.handleKeyUp.bind(this));
  }

  setupSingleInteractiveCanvas(traceType) {
    const mainCanvas = document.getElementById(`${traceType}-main`);
    const minimapCanvas = document.getElementById(`${traceType}-minimap`);
    const resetBtn = document.getElementById(`${traceType}-reset-btn`);

    if (!mainCanvas || !minimapCanvas)
      return;

    // Setup main canvas
    const container = mainCanvas.parentElement;
    const containerWidth = container.clientWidth;

    mainCanvas.width = containerWidth * window.devicePixelRatio;
    mainCanvas.height = 350 * window.devicePixelRatio;
    mainCanvas.style.width = containerWidth + 'px';
    mainCanvas.style.height = '350px';

    const mainCtx = mainCanvas.getContext('2d');
    mainCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Setup minimap
    minimapCanvas.width = containerWidth * window.devicePixelRatio;
    minimapCanvas.height = this.MINIMAP_HEIGHT * window.devicePixelRatio;
    minimapCanvas.style.width = containerWidth + 'px';
    minimapCanvas.style.height = this.MINIMAP_HEIGHT + 'px';

    const minimapCtx = minimapCanvas.getContext('2d');
    minimapCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Add interactive event listeners
    this.addCanvasInteractions(mainCanvas, minimapCanvas, traceType);

    // Reset button
    if (resetBtn) {
      resetBtn.addEventListener('click',
                                () => { this.resetViewport(traceType); });
    }

    // Render trace data
    const data =
        traceType === 'time-trace' ? this.timeTraceData : this.runtimeTraceData;
    this.renderSingleTimeOrder(traceType, data, mainCtx, minimapCtx, mainCanvas,
                               minimapCanvas);
  }

  renderSingleTimeOrder(traceType, data, mainCtx, minimapCtx, mainCanvas,
                        minimapCanvas) {
    const statusEl = document.getElementById(`${traceType}-status`);

    if (!data || !data.samples || data.samples.length === 0) {
      statusEl.textContent = 'No data available';
      statusEl.className =
          statusEl.className.replace('text-gray-500', 'text-red-500');

      const canvasWidth = mainCanvas.width / window.devicePixelRatio;
      const canvasHeight = mainCanvas.height / window.devicePixelRatio;

      mainCtx.clearRect(0, 0, canvasWidth, canvasHeight);
      mainCtx.fillStyle = '#f3f4f6';
      mainCtx.fillRect(0, 0, canvasWidth, canvasHeight);

      mainCtx.fillStyle = '#6b7280';
      mainCtx.font = '14px system-ui';
      mainCtx.textAlign = 'center';
      mainCtx.fillText('No trace data available', canvasWidth / 2,
                       canvasHeight / 2);

      minimapCtx.clearRect(0, 0, minimapCanvas.width / window.devicePixelRatio,
                           this.MINIMAP_HEIGHT);
      return;
    }

    const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
    const viewport = this.viewports[key];

    statusEl.textContent = `${data.samples.length} events • Zoom: ${
        Math.round(viewport.scaleX * 100)}%`;
    statusEl.className =
        statusEl.className.replace('text-red-500', 'text-gray-500');

    // Render flamechart with viewport
    this.renderFlamechart(data.samples, mainCtx, mainCanvas, traceType);
    this.renderMinimap(data.samples, minimapCtx, minimapCanvas, traceType);

    // Update zoom display
    this.updateZoomDisplay(traceType);
  }

  renderFlamechart(samples, ctx, canvas, traceType) {
    const canvasWidth = canvas.width / window.devicePixelRatio;
    const canvasHeight = canvas.height / window.devicePixelRatio;

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    if (!samples || samples.length === 0)
      return;

    // Get viewport for this trace
    const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
    const viewport = this.viewports[key];

    // Calculate time bounds
    const times =
        samples.map(s => [s.timestamp, s.timestamp + s.duration]).flat();
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const totalDuration = maxTime - minTime;

    if (totalDuration === 0)
      return;

    // Build layers to avoid overlaps
    const layers = this.buildNonOverlappingLayers(samples);

    // Apply viewport transformations
    const viewportLeft = viewport.offsetX * totalDuration;
    const viewportWidth = totalDuration / viewport.scaleX;
    const visibleTimeStart = minTime + viewportLeft;
    const visibleTimeEnd = visibleTimeStart + viewportWidth;

    // Calculate scaling with zoom
    const timeScale = canvasWidth / viewportWidth;
    const maxVisibleLayers = Math.floor(canvasHeight / this.FRAME_HEIGHT);

    // Apply vertical offset
    const layerOffset = Math.floor(viewport.offsetY * layers.length);
    const visibleLayers =
        layers.slice(layerOffset, layerOffset + maxVisibleLayers);

    // Render visible layers
    visibleLayers.forEach((layer, layerIndex) => {
      const y = layerIndex * this.FRAME_HEIGHT;

      layer.forEach(sample => {
        const sampleStart = sample.timestamp;
        const sampleEnd = sample.timestamp + sample.duration;

        // Only render if sample is visible in viewport
        if (sampleEnd >= visibleTimeStart && sampleStart <= visibleTimeEnd) {
          const x = (sampleStart - visibleTimeStart) * timeScale;
          const width = Math.max(1, sample.duration * timeScale);

          // Only draw if within canvas bounds
          if (x < canvasWidth && x + width > 0) {
            this.drawFrame(ctx, sample, x, y, width, this.FRAME_HEIGHT);
          }
        }
      });
    });

    // Render time axis for visible range
    this.renderTimeAxis(ctx, visibleTimeStart, viewportWidth, canvasWidth,
                        canvasHeight);

    // Store current viewport bounds for interaction
    canvas.dataset.visibleTimeStart = visibleTimeStart;
    canvas.dataset.visibleTimeEnd = visibleTimeEnd;
    canvas.dataset.timeScale = timeScale;
  }

  buildNonOverlappingLayers(samples) {
    const layers = [];
    const sortedSamples =
        [...samples ].sort((a, b) => a.timestamp - b.timestamp);

    for (const sample of sortedSamples) {
      let placed = false;

      // Try to place in existing layer
      for (let i = 0; i < layers.length; i++) {
        const layer = layers[i];
        const lastInLayer = layer[layer.length - 1];

        if (!lastInLayer ||
            lastInLayer.timestamp + lastInLayer.duration <= sample.timestamp) {
          layer.push(sample);
          placed = true;
          break;
        }
      }

      // Create new layer if needed
      if (!placed) {
        layers.push([ sample ]);
      }
    }

    return layers;
  }

  drawFrame(ctx, sample, x, y, width, height) {
    const isSearchMatch = this.searchQuery &&
                          sample.name.toLowerCase().includes(this.searchQuery);

    // Frame background
    let color = this.getCategoryColor(sample.category);
    if (this.searchQuery && !isSearchMatch) {
      color = this.fadeColor(color);
    }

    ctx.fillStyle = color;
    ctx.fillRect(x, y, width, height);

    // Frame border
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(x, y, width, height);

    // Search highlight
    if (isSearchMatch) {
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
    }

    // Frame text
    if (width > this.MIN_FRAME_WIDTH_FOR_TEXT) {
      ctx.fillStyle = this.getTextColor(color);
      ctx.font = '11px system-ui';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';

      const text = this.truncateText(sample.name, width - 2 * this.PADDING);
      ctx.fillText(text, x + this.PADDING, y + height / 2);
    }
  }

  renderTimeAxis(ctx, minTime, totalDuration, canvasWidth, canvasHeight) {
    if (totalDuration === 0)
      return;

    // Calculate intervals
    const targetInterval = totalDuration / 8;
    const magnitude = Math.pow(10, Math.floor(Math.log10(targetInterval)));
    let interval = magnitude;

    if (targetInterval / interval > 5)
      interval *= 5;
    else if (targetInterval / interval > 2)
      interval *= 2;

    // Draw axis
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#6b7280';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'center';

    const timeScale = canvasWidth / totalDuration;

    for (let time = Math.ceil(minTime / interval) * interval;
         time <= minTime + totalDuration; time += interval) {

      const x = (time - minTime) * timeScale;

      if (x >= 0 && x <= canvasWidth) {
        // Grid line
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvasHeight - 20);
        ctx.stroke();

        // Time label
        const timeMs = (time / 1000).toFixed(1);
        ctx.fillText(`${timeMs}ms`, x, canvasHeight - 5);
      }
    }
  }

  renderMinimap(samples, ctx, canvas, traceType) {
    const canvasWidth = canvas.width / window.devicePixelRatio;
    const canvasHeight = canvas.height / window.devicePixelRatio;

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    if (!samples || samples.length === 0)
      return;

    const times =
        samples.map(s => [s.timestamp, s.timestamp + s.duration]).flat();
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const totalDuration = maxTime - minTime;

    if (totalDuration === 0)
      return;

    const timeScale = canvasWidth / totalDuration;
    const barHeight = canvasHeight - 10;

    // Render simplified bars
    samples.forEach(sample => {
      const x = (sample.timestamp - minTime) * timeScale;
      const width = Math.max(1, sample.duration * timeScale);
      const y = 5;

      ctx.fillStyle = this.getCategoryColor(sample.category);
      ctx.fillRect(x, y, width, barHeight);
    });
  }

  renderSandwichViews() {
    const container = document.getElementById('performance-visualization');
    container.innerHTML = `
            <div class="flex h-full">
                <!-- Functions Table - Left side like speedscope -->
                <div class="w-2/5 bg-white border-r border-gray-200 flex flex-col">
                    <div class="bg-gray-50 px-4 py-3 border-b border-gray-200">
                        <div class="flex items-center justify-between">
                            <h3 class="text-sm font-semibold text-gray-900">Functions</h3>
                            <div class="flex items-center space-x-3">
                                <select id="sandwich-sort" class="text-xs border border-gray-300 rounded px-2 py-1">
                                    <option value="total">Total Time</option>
                                    <option value="self">Self Time</option>
                                    <option value="name">Name</option>
                                </select>
                                <span id="sandwich-count" class="text-xs text-gray-600">0 functions</span>
                            </div>
                        </div>
                    </div>
                    <div class="flex-1 overflow-hidden">
                        <div class="h-full overflow-y-auto">
                            <table class="min-w-full text-xs">
                                <thead class="bg-gray-50 sticky top-0">
                                    <tr class="border-b border-gray-200">
                                        <th class="px-3 py-2 text-left font-medium text-gray-700 cursor-pointer" data-sort="total">
                                            Total
                                            <svg class="w-3 h-3 inline ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"/>
                                            </svg>
                                        </th>
                                        <th class="px-3 py-2 text-left font-medium text-gray-700 cursor-pointer" data-sort="self">
                                            Self
                                            <svg class="w-3 h-3 inline ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"/>
                                            </svg>
                                        </th>
                                        <th class="px-3 py-2 text-left font-medium text-gray-700 cursor-pointer" data-sort="name">
                                            Function
                                            <svg class="w-3 h-3 inline ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"/>
                                            </svg>
                                        </th>
                                    </tr>
                                </thead>
                                <tbody id="sandwich-table-body" class="divide-y divide-gray-100">
                                    <!-- Dynamic content -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <!-- Flamegraph Details - Right side like speedscope -->
                <div class="flex-1 flex flex-col bg-white">
                    <div class="bg-gray-50 px-4 py-3 border-b border-gray-200">
                        <h3 class="text-sm font-semibold text-gray-900" id="sandwich-detail-title">Select a function to view details</h3>
                    </div>
                    <div class="flex-1 flex flex-col">
                        <!-- Callers Section -->
                        <div class="flex-1 border-b border-gray-200">
                            <div class="h-8 bg-gray-100 flex items-center px-4 border-b border-gray-200">
                                <span class="text-xs font-medium text-gray-700">Callers (functions that call this)</span>
                            </div>
                            <div id="callers-chart" class="h-full bg-gray-50 flex items-center justify-center text-gray-500 text-sm">
                                Select a function to see its callers
                            </div>
                        </div>
                        
                        <!-- Callees Section -->
                        <div class="flex-1">
                            <div class="h-8 bg-gray-100 flex items-center px-4 border-b border-gray-200">
                                <span class="text-xs font-medium text-gray-700">Callees (functions called by this)</span>
                            </div>
                            <div id="callees-chart" class="h-full bg-gray-50 flex items-center justify-center text-gray-500 text-sm">
                                Select a function to see its callees
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

    this.renderSpeedscopeSandwich();
  }

  renderSpeedscopeSandwich() {
    // Combine data from both traces
    const allFunctions = [];

    // Add time-trace functions
    if (this.timeTraceData && this.timeTraceData.functions) {
      this.timeTraceData.functions.forEach(func => {
        allFunctions.push({
          ...func,
          source : 'compilation',
          color : this.getCategoryColor(func.category)
        });
      });
    }

    // Add runtime-trace functions
    if (this.runtimeTraceData && this.runtimeTraceData.functions) {
      this.runtimeTraceData.functions.forEach(func => {
        allFunctions.push({
          ...func,
          source : 'runtime',
          color : this.getCategoryColor(func.category)
        });
      });
    }

    // Merge functions with same name
    const functionMap = new Map();
    allFunctions.forEach(func => {
      const key = func.name;
      if (functionMap.has(key)) {
        const existing = functionMap.get(key);
        existing.total_time += func.total_time;
        existing.call_count += func.call_count;
        existing.sources = existing.sources || [ existing.source ];
        if (!existing.sources.includes(func.source)) {
          existing.sources.push(func.source);
        }
      } else {
        functionMap.set(key, {
          ...func,
          self_time : func.total_time,
          sources : [ func.source ]
        });
      }
    });

    const mergedFunctions = Array.from(functionMap.values());
    const totalTime = mergedFunctions.reduce((sum, f) => sum + f.total_time, 0);

    // Sort by total time by default
    mergedFunctions.sort((a, b) => b.total_time - a.total_time);

    this.renderFunctionTable(mergedFunctions, totalTime);
    this.setupSandwichInteractions(mergedFunctions);

    // Update count
    const countEl = document.getElementById('sandwich-count');
    if (countEl) {
      countEl.textContent = `${mergedFunctions.length} functions`;
    }
  }

  renderFunctionTable(functions, totalTime) {
    const tbody = document.getElementById('sandwich-table-body');
    if (!tbody)
      return;

    tbody.innerHTML =
        functions
            .map((func, index) => {
              const totalPerc =
                  totalTime > 0 ? (func.total_time / totalTime * 100) : 0;
              const selfPerc =
                  totalTime > 0 ? (func.self_time / totalTime * 100) : 0;
              const isMatch =
                  this.searchQuery &&
                  func.name.toLowerCase().includes(this.searchQuery);

              return `
                <tr class="hover:bg-gray-50 cursor-pointer sandwich-row ${
                  isMatch ? 'bg-yellow-50' : ''}" data-index="${index}">
                    <td class="px-3 py-2">
                        <div class="flex items-center">
                            <div class="w-full bg-gray-200 rounded-full h-1 mr-2" style="width: 60px;">
                                <div class="h-1 rounded-full" style="width: ${
                  totalPerc.toFixed(
                      1)}%; background-color: ${func.color};"></div>
                            </div>
                            <span class="text-xs font-mono font-medium">${
                  (func.total_time / 1000).toFixed(2)}ms</span>
                            <span class="text-xs text-gray-500 ml-1">(${
                  totalPerc.toFixed(1)}%)</span>
                        </div>
                    </td>
                    <td class="px-3 py-2">
                        <div class="flex items-center">
                            <div class="w-full bg-gray-200 rounded-full h-1 mr-2" style="width: 60px;">
                                <div class="h-1 rounded-full" style="width: ${
                  selfPerc.toFixed(
                      1)}%; background-color: ${func.color};"></div>
                            </div>
                            <span class="text-xs font-mono font-medium">${
                  (func.self_time / 1000).toFixed(2)}ms</span>
                            <span class="text-xs text-gray-500 ml-1">(${
                  selfPerc.toFixed(1)}%)</span>
                        </div>
                    </td>
                    <td class="px-3 py-2">
                        <div class="flex items-center">
                            <div class="w-2 h-2 rounded mr-2" style="background-color: ${
                  func.color};"></div>
                            <div class="flex-1 min-w-0">
                                <div class="text-xs font-medium text-gray-900 truncate" title="${
                  func.name}">
                                    ${this.highlightSearchInText(func.name)}
                                </div>
                                <div class="text-xs text-gray-500">
                                    ${func.call_count.toLocaleString()} calls
                                    ${
                  func.sources ? ' • ' + func.sources.join(', ') : ''}
                                </div>
                            </div>
                        </div>
                    </td>
                </tr>
            `;
            })
            .join('');
  }

  setupSandwichInteractions(functions) {
    // Set up table row clicks
    const rows = document.querySelectorAll('.sandwich-row');
    rows.forEach(row => {
      row.addEventListener('click', () => {
        // Remove previous selection
        document.querySelectorAll('.sandwich-row')
            .forEach(r => r.classList.remove('bg-blue-50'));

        // Add selection to clicked row
        row.classList.add('bg-blue-50');

        const index = parseInt(row.dataset.index);
        const selectedFunction = functions[index];

        this.showFunctionDetails(selectedFunction);
      });
    });

    // Set up sorting
    const sortSelect = document.getElementById('sandwich-sort');
    if (sortSelect) {
      sortSelect.addEventListener(
          'change', () => { this.sortFunctions(functions, sortSelect.value); });
    }

    // Set up column header sorting
    const headers = document.querySelectorAll('th[data-sort]');
    headers.forEach(header => {
      header.addEventListener('click', () => {
        const sortBy = header.dataset.sort;
        this.sortFunctions(functions, sortBy);
      });
    });
  }

  sortFunctions(functions, sortBy) {
    switch (sortBy) {
    case 'total':
      functions.sort((a, b) => b.total_time - a.total_time);
      break;
    case 'self':
      functions.sort((a, b) => b.self_time - a.self_time);
      break;
    case 'name':
      functions.sort((a, b) => a.name.localeCompare(b.name));
      break;
    }

    const totalTime = functions.reduce((sum, f) => sum + f.total_time, 0);
    this.renderFunctionTable(functions, totalTime);
    this.setupSandwichInteractions(functions);
  }

  showFunctionDetails(func) {
    const titleEl = document.getElementById('sandwich-detail-title');
    const callersEl = document.getElementById('callers-chart');
    const calleesEl = document.getElementById('callees-chart');

    if (titleEl) {
      titleEl.innerHTML = `
                <div class="flex items-center">
                    <div class="w-3 h-3 rounded mr-2" style="background-color: ${
          func.color};"></div>
                    <span class="font-semibold">${func.name}</span>
                    <span class="ml-2 text-xs text-gray-600">${
          (func.total_time / 1000).toFixed(2)}ms total</span>
                </div>
            `;
    }

    // For now, show placeholder content in callers/callees
    if (callersEl) {
      callersEl.innerHTML = `
                <div class="p-4 text-center">
                    <div class="text-sm text-gray-600 mb-2">Callers for <strong>${
          func.name}</strong></div>
                    <div class="text-xs text-gray-500">Implementation would show flamegraph of functions that call this</div>
                    <div class="mt-2 text-xs">
                        <div class="bg-gray-200 rounded p-2">
                            Total calls: ${func.call_count.toLocaleString()}<br>
                            Sources: ${
          func.sources ? func.sources.join(', ') : 'unknown'}
                        </div>
                    </div>
                </div>
            `;
    }

    if (calleesEl) {
      calleesEl.innerHTML = `
                <div class="p-4 text-center">
                    <div class="text-sm text-gray-600 mb-2">Callees for <strong>${
          func.name}</strong></div>
                    <div class="text-xs text-gray-500">Implementation would show flamegraph of functions called by this</div>
                    <div class="mt-2 text-xs">
                        <div class="bg-gray-200 rounded p-2">
                            Self time: ${
          (func.self_time / 1000).toFixed(2)}ms<br>
                            Category: ${func.category || 'Unknown'}
                        </div>
                    </div>
                </div>
            `;
    }
  }

  performSearch() {
    if (this.currentViewType === 'sandwich') {
      this.renderSandwichTables();
    } else {
      this.renderTimeOrderViews();
    }
    this.updateSearchResultsDisplay();
  }

  updateSearchResultsDisplay() {
    const countEl = document.getElementById('search-results-count');
    if (!countEl)
      return;

    let totalResults = 0;

    if (this.timeTraceData && this.timeTraceData.functions) {
      totalResults +=
          this.timeTraceData.functions
              .filter(f => !this.searchQuery ||
                           f.name.toLowerCase().includes(this.searchQuery))
              .length;
    }

    if (this.runtimeTraceData && this.runtimeTraceData.functions) {
      totalResults +=
          this.runtimeTraceData.functions
              .filter(f => !this.searchQuery ||
                           f.name.toLowerCase().includes(this.searchQuery))
              .length;
    }

    countEl.textContent = this.searchQuery ? `${totalResults} results` : '';
  }

  updateIndividualStats() {
    // Update individual trace stats in headers
    this.updateTraceStats('time-trace', this.timeTraceData);
    this.updateTraceStats('runtime-trace', this.runtimeTraceData);

    // Update global stats if elements exist
    const totalEventsEl = document.getElementById('perf-total-events');
    const totalDurationEl = document.getElementById('perf-total-duration');
    const avgDurationEl = document.getElementById('perf-avg-duration');
    const viewModeEl = document.getElementById('perf-view-mode');

    if (totalEventsEl || totalDurationEl || avgDurationEl || viewModeEl) {
      let totalEvents = 0;
      let totalDuration = 0;

      // Combine stats from both traces
      [this.timeTraceData, this.runtimeTraceData].forEach(data => {
        if (data) {
          if (data.samples) {
            totalEvents += data.samples.length;
            totalDuration +=
                data.samples.reduce((sum, s) => sum + s.duration, 0);
          } else if (data.functions) {
            totalEvents +=
                data.functions.reduce((sum, f) => sum + f.call_count, 0);
            totalDuration +=
                data.functions.reduce((sum, f) => sum + f.total_time, 0);
          }
        }
      });

      if (totalEventsEl)
        totalEventsEl.textContent = totalEvents.toLocaleString();
      if (totalDurationEl)
        totalDurationEl.textContent = `${(totalDuration / 1000).toFixed(2)} ms`;
      if (avgDurationEl) {
        const avg = totalEvents > 0 ? totalDuration / totalEvents : 0;
        avgDurationEl.textContent = `${(avg / 1000).toFixed(2)} ms`;
      }
      if (viewModeEl) {
        viewModeEl.textContent = this.currentViewType.charAt(0).toUpperCase() +
                                 this.currentViewType.slice(1);
      }
    }
  }

  updateTraceStats(traceType, data) {
    const statsEl = document.getElementById(`${traceType}-stats`);
    if (!statsEl)
      return;

    if (!data) {
      statsEl.textContent = 'No data';
      return;
    }

    let events = 0;
    let duration = 0;
    let sources = new Set();

    if (data.samples) {
      events = data.samples.length;
      duration = data.samples.reduce((sum, s) => sum + (s.duration || 0), 0);
      data.samples.forEach(s => {
        if (s.source)
          sources.add(s.source);
      });
    } else if (data.functions) {
      events = data.functions.reduce((sum, f) => sum + f.call_count, 0);
      duration = data.functions.reduce((sum, f) => sum + f.total_time, 0);
      data.functions.forEach(f => {
        if (f.source)
          sources.add(f.source);
      });
    }

    const sourceInfo =
        sources.size > 0 ? ` • ${Array.from(sources).join(', ')}` : '';
    statsEl.textContent = `${events.toLocaleString()} events • ${
        (duration / 1000).toFixed(2)}ms${sourceInfo}`;

    // Update debug info in UI
    const debugEl = document.getElementById(`${traceType}-debug`);
    if (debugEl) {
      const sourceList = Array.from(sources);
      const debugInfo =
          sourceList.length > 0 ? sourceList.join(',') : 'unknown';
      debugEl.textContent = debugInfo;
      debugEl.title = `Data source: ${debugInfo}\nSamples: ${
          data.samples
              ? data.samples.length
              : 0}\nFunctions: ${data.functions ? data.functions.length : 0}`;
    }

    // Add debug info in console
    console.log(`${traceType} data:`, {
      events,
      duration,
      sources : Array.from(sources),
      sampleData : data.samples ? data.samples.slice(0, 3) : null,
      functionData : data.functions ? data.functions.slice(0, 3) : null
    });

    // Add visual indicator if data appears to be identical between traces
    if (traceType === 'runtime-trace' && window.timeTraceDataHash) {
      const currentHash = this.hashData(data);
      if (currentHash === window.timeTraceDataHash) {
        debugEl.style.backgroundColor = '#ef4444';
        debugEl.style.color = 'white';
        debugEl.textContent = '⚠ SAME';
        debugEl.title =
            'WARNING: This data appears identical to time-trace data';
      }
    } else if (traceType === 'time-trace') {
      window.timeTraceDataHash = this.hashData(data);
    }
  }

  // Utility methods
  getCategoryColor(category) {
    const colors = {
      'Source' : '#10b981',    // emerald-500
      'Frontend' : '#3b82f6',  // blue-500
      'Backend' : '#f59e0b',   // amber-500
      'CodeGen' : '#ef4444',   // red-500
      'Optimizer' : '#8b5cf6', // violet-500
      'Parse' : '#6b7280',     // gray-500
      'Runtime' : '#ec4899',   // pink-500
      '' : '#9ca3af'           // gray-400
    };
    return colors[category] || colors[''];
  }

  getTextColor(backgroundColor) {
    // Simple contrast calculation
    const hex = backgroundColor.replace('#', '');
    const r = parseInt(hex.substr(0, 2), 16);
    const g = parseInt(hex.substr(2, 2), 16);
    const b = parseInt(hex.substr(4, 2), 16);
    const brightness = (r * 299 + g * 587 + b * 114) / 1000;
    return brightness > 128 ? '#1f2937' : '#f9fafb';
  }

  fadeColor(color) {
    const hex = color.replace('#', '');
    const r = parseInt(hex.substr(0, 2), 16);
    const g = parseInt(hex.substr(2, 2), 16);
    const b = parseInt(hex.substr(4, 2), 16);
    return `rgba(${r}, ${g}, ${b}, 0.3)`;
  }

  truncateText(text, maxWidth) {
    if (text.length * 6 <= maxWidth)
      return text;
    const maxChars = Math.floor(maxWidth / 6) - 3;
    return text.substring(0, Math.max(0, maxChars)) + '...';
  }

  highlightSearchInText(text) {
    if (!this.searchQuery)
      return text;
    const regex = new RegExp(`(${this.searchQuery})`, 'gi');
    return text.replace(regex,
                        '<mark class="bg-yellow-200 px-1 rounded">$1</mark>');
  }

  showError(message) {
    const container = document.getElementById('performance-visualization');
    if (container) {
      container.innerHTML = `
                <div class="bg-white rounded-lg shadow-lg p-8">
                    <div class="text-center">
                        <svg class="mx-auto h-12 w-12 text-red-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                        </svg>
                        <h3 class="text-lg font-medium text-gray-900 mb-2">Error Loading Performance Data</h3>
                        <p class="text-gray-600">${message}</p>
                        <button onclick="window.performanceManager?.loadAllPerformanceData()" 
                                class="mt-4 px-4 py-2 bg-llvm-blue text-white rounded-md hover:bg-blue-700">
                            Retry
                        </button>
                    </div>
                </div>
            `;
    }
  }

  async initialize() {
    this.currentUnit = document.getElementById('unit-selector')?.value || null;
    // Auto-load data without requiring refresh button
    await this.loadAllPerformanceData();

    // Set up auto-refresh if unit changes
    const unitSelector = document.getElementById('unit-selector');
    if (unitSelector) {
      unitSelector.addEventListener('change', () => {
        this.currentUnit = unitSelector.value;
        this.loadAllPerformanceData();
      });
    }
  }

  // Interactive methods
  resetViewports() {
    this.viewports
        .timeTrace = {offsetX : 0, scaleX : 1, offsetY : 0, scaleY : 1};
    this.viewports
        .runtimeTrace = {offsetX : 0, scaleX : 1, offsetY : 0, scaleY : 1};
  }

  resetViewport(traceType) {
    const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
    this.viewports[key] = {offsetX : 0, scaleX : 1, offsetY : 0, scaleY : 1};
    this.redrawTrace(traceType);
    this.updateZoomDisplay(traceType);
  }

  updateZoomDisplay(traceType) {
    const zoomEl = document.getElementById(`${traceType}-zoom-level`);
    if (zoomEl) {
      const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
      const zoom = Math.round(this.viewports[key].scaleX * 100);
      zoomEl.textContent = `${zoom}%`;
    }
  }

  addCanvasInteractions(mainCanvas, minimapCanvas, traceType) {
    const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';

    // Main canvas interactions
    mainCanvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.handleWheel(e, traceType);
    });

    mainCanvas.addEventListener('mousedown',
                                (e) => { this.handleMouseDown(e, traceType); });

    mainCanvas.addEventListener('mousemove',
                                (e) => { this.handleMouseMove(e, traceType); });

    mainCanvas.addEventListener('mouseup',
                                (e) => { this.handleMouseUp(e, traceType); });

    mainCanvas.addEventListener(
        'dblclick', (e) => { this.handleDoubleClick(e, traceType); });

    // Minimap interactions
    minimapCanvas.addEventListener(
        'click', (e) => { this.handleMinimapClick(e, traceType); });

    minimapCanvas.addEventListener(
        'mousedown', (e) => { this.handleMinimapDrag(e, traceType); });
  }

  handleMinimapDrag(e, traceType) {
    let isDragging = false;
    const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
    const viewport = this.viewports[key];

    const startDrag = (startE) => {
      isDragging = true;

      const onDrag = (moveE) => {
        if (!isDragging)
          return;

        const rect = e.target.getBoundingClientRect();
        const currentX = (moveE.clientX - rect.left) / rect.width;

        const viewportWidthRatio = 1 / viewport.scaleX;
        viewport.offsetX = currentX - viewportWidthRatio / 2;

        // Clamp to valid range
        viewport.offsetX =
            Math.max(0, Math.min(1 - viewportWidthRatio, viewport.offsetX));

        this.redrawTrace(traceType);
      };

      const endDrag = () => {
        isDragging = false;
        document.removeEventListener('mousemove', onDrag);
        document.removeEventListener('mouseup', endDrag);
      };

      document.addEventListener('mousemove', onDrag);
      document.addEventListener('mouseup', endDrag);
    };

    startDrag(e);
  }

  handleWheel(e, traceType) {
    const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
    const viewport = this.viewports[key];

    const zoomSpeed = 0.1;
    const zoomFactor = e.deltaY < 0 ? (1 + zoomSpeed) : (1 - zoomSpeed);

    const newScale = Math.max(
        this.MIN_ZOOM, Math.min(this.MAX_ZOOM, viewport.scaleX * zoomFactor));

    if (newScale !== viewport.scaleX) {
      const rect = e.target.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;

      // Zoom towards mouse position
      const canvasWidth = rect.width;
      const normalizedMouseX = mouseX / canvasWidth;

      viewport.offsetX =
          normalizedMouseX -
          (normalizedMouseX - viewport.offsetX) * (newScale / viewport.scaleX);
      viewport.scaleX = newScale;

      this.redrawTrace(traceType);
      this.updateZoomDisplay(traceType);
    }
  }

  handleMouseDown(e, traceType) {
    this.isDragging = true;
    this.lastMousePos = {x : e.clientX, y : e.clientY};
    e.target.style.cursor = 'grabbing';
  }

  handleMouseMove(e, traceType) {
    if (this.isDragging) {
      const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
      const viewport = this.viewports[key];

      const rect = e.target.getBoundingClientRect();
      const deltaX = (e.clientX - this.lastMousePos.x) / rect.width;
      const deltaY = (e.clientY - this.lastMousePos.y) / rect.height;

      viewport.offsetX -= deltaX / viewport.scaleX;
      viewport.offsetY -= deltaY / viewport.scaleY;

      this.lastMousePos = {x : e.clientX, y : e.clientY};
      this.redrawTrace(traceType);
    } else {
      // Show tooltip
      this.showTooltip(e, traceType);
    }
  }

  handleMouseUp(e, traceType) {
    this.isDragging = false;
    e.target.style.cursor = 'grab';
  }

  handleDoubleClick(e, traceType) {
    // Fit frame functionality - find frame under cursor and zoom to it
    const frame = this.getFrameAtPosition(e, traceType);
    if (frame) {
      this.fitToFrame(frame, traceType);
    } else {
      this.resetViewport(traceType);
    }
  }

  handleMinimapClick(e, traceType) {
    const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
    const viewport = this.viewports[key];

    const rect = e.target.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) / rect.width;

    viewport.offsetX = clickX - 0.5 / viewport.scaleX;
    this.redrawTrace(traceType);
  }

  handleKeyDown(e) {
    let handled = false;

    switch (e.code) {
    case 'Equal':
    case 'NumpadAdd':
      if (e.ctrlKey || e.metaKey) {
        this.zoomIn();
        handled = true;
      }
      break;
    case 'Minus':
    case 'NumpadSubtract':
      if (e.ctrlKey || e.metaKey) {
        this.zoomOut();
        handled = true;
      }
      break;
    case 'Escape':
      this.resetViewports();
      this.renderTimeOrderViews();
      handled = true;
      break;
    case 'ArrowLeft':
      this.panLeft();
      handled = true;
      break;
    case 'ArrowRight':
      this.panRight();
      handled = true;
      break;
    }

    if (handled) {
      e.preventDefault();
    }
  }

  handleKeyUp(e) {
    // Handle key release if needed
  }

  zoomIn() {
    ['time-trace', 'runtime-trace'].forEach(traceType => {
      const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
      const viewport = this.viewports[key];
      viewport.scaleX = Math.min(this.MAX_ZOOM, viewport.scaleX * 1.2);
      this.redrawTrace(traceType);
      this.updateZoomDisplay(traceType);
    });
  }

  zoomOut() {
    ['time-trace', 'runtime-trace'].forEach(traceType => {
      const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
      const viewport = this.viewports[key];
      viewport.scaleX = Math.max(this.MIN_ZOOM, viewport.scaleX / 1.2);
      this.redrawTrace(traceType);
      this.updateZoomDisplay(traceType);
    });
  }

  panLeft() {
    ['time-trace', 'runtime-trace'].forEach(traceType => {
      const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
      this.viewports[key].offsetX -= 0.1;
      this.redrawTrace(traceType);
    });
  }

  panRight() {
    ['time-trace', 'runtime-trace'].forEach(traceType => {
      const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
      this.viewports[key].offsetX += 0.1;
      this.redrawTrace(traceType);
    });
  }

  redrawTrace(traceType) {
    const mainCanvas = document.getElementById(`${traceType}-main`);
    const minimapCanvas = document.getElementById(`${traceType}-minimap`);

    if (!mainCanvas || !minimapCanvas)
      return;

    const mainCtx = mainCanvas.getContext('2d');
    const minimapCtx = minimapCanvas.getContext('2d');

    const data =
        traceType === 'time-trace' ? this.timeTraceData : this.runtimeTraceData;
    this.renderSingleTimeOrder(traceType, data, mainCtx, minimapCtx, mainCanvas,
                               minimapCanvas);
  }

  getFrameAtPosition(e, traceType) {
    // Find frame under cursor for double-click to fit
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    return null;
  }

  fitToFrame(frame, traceType) {
    // Zoom and pan to fit specific frame
    const key = traceType === 'time-trace' ? 'timeTrace' : 'runtimeTrace';
    const viewport = this.viewports[key];

    // Calculate optimal zoom and offset for frame
    viewport.scaleX = 2.0;
    viewport.offsetX = 0;

    this.redrawTrace(traceType);
    this.updateZoomDisplay(traceType);
  }

  showTooltip(e, traceType) {
    const tooltip = document.getElementById(`${traceType}-tooltip`);
    if (!tooltip)
      return;

    // Find frame at mouse position and show tooltip
    const frame = this.getFrameAtPosition(e, traceType);

    if (frame) {
      tooltip.innerHTML = `
                <div class="font-semibold">${frame.name}</div>
                <div class="text-xs">Duration: ${
          (frame.duration / 1000).toFixed(2)}ms</div>
                <div class="text-xs">Category: ${
          frame.category || 'Unknown'}</div>
            `;

      tooltip.style.left = `${e.offsetX + 10}px`;
      tooltip.style.top = `${e.offsetY - 50}px`;
      tooltip.classList.remove('hidden');
    } else {
      tooltip.classList.add('hidden');
    }
  }

  hashData(data) {
    // Simple hash function to detect if data is identical
    if (!data)
      return null;

    let hashString = '';
    if (data.samples) {
      hashString =
          data.samples.map(s => `${s.name}:${s.duration}:${s.timestamp}`)
              .join('|');
    } else if (data.functions) {
      hashString =
          data.functions.map(f => `${f.name}:${f.total_time}:${f.call_count}`)
              .join('|');
    }

    // Simple hash
    let hash = 0;
    for (let i = 0; i < hashString.length; i++) {
      const char = hashString.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return hash;
  }

  cleanup() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }

    // Remove event listeners
    document.removeEventListener('keydown', this.handleKeyDown);
    document.removeEventListener('keyup', this.handleKeyUp);
  }
}

window.PerformanceManager = PerformanceManager;
window.performanceManager = null;
