// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Chart Components
 * Wrapper components for Chart.js with LLVM Advisor styling and functionality
 */

export class ChartComponents {
  constructor() {
    this.defaultColors = {
      primary : '#3b82f6',
      secondary : '#64748b',
      success : '#10b981',
      warning : '#f59e0b',
      error : '#ef4444',
      info : '#06b6d4'
    };

    this.colorPalettes = {
      blue : [
        '#dbeafe', '#bfdbfe', '#93c5fd', '#60a5fa', '#3b82f6', '#2563eb',
        '#1d4ed8', '#1e40af'
      ],
      green : [
        '#d1fae5', '#a7f3d0', '#6ee7b7', '#34d399', '#10b981', '#059669',
        '#047857', '#065f46'
      ],
      purple : [
        '#e9d5ff', '#d8b4fe', '#c084fc', '#a855f7', '#9333ea', '#7c3aed',
        '#6d28d9', '#5b21b6'
      ],
      orange : [
        '#fed7aa', '#fdba74', '#fb923c', '#f97316', '#ea580c', '#dc2626',
        '#b91c1c', '#991b1b'
      ],
      mixed : [
        '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4',
        '#84cc16', '#f97316'
      ]
    };
  }

  /**
   * Initialize Chart.js with global configurations
   */
  init() {
    if (typeof Chart === 'undefined') {
      console.warn('Chart.js not loaded. Charts will not be available.');
      return;
    }

    // Set global Chart.js defaults
    Chart.defaults.font.family = 'Inter, system-ui, -apple-system, sans-serif';
    Chart.defaults.font.size = 12;
    Chart.defaults.color = '#6b7280';
    Chart.defaults.borderColor = '#e5e7eb';
    Chart.defaults.backgroundColor = '#f9fafb';

    // Configure default responsive options
    Chart.defaults.responsive = true;
    Chart.defaults.maintainAspectRatio = false;

    // Configure default animation
    Chart.defaults.animation.duration = 400;
    Chart.defaults.animation.easing = 'easeInOutQuart';

    console.log('Chart components initialized');
  }

  /**
   * Generate color palette for charts
   */
  generateColors(count, palette = 'mixed', opacity = 1) {
    const colors = this.colorPalettes[palette] || this.colorPalettes.mixed;
    const result = [];

    for (let i = 0; i < count; i++) {
      const colorIndex = i % colors.length;
      const color = colors[colorIndex];

      if (opacity < 1) {
        // Convert hex to rgba
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        result.push(`rgba(${r}, ${g}, ${b}, ${opacity})`);
      } else {
        result.push(color);
      }
    }

    return result;
  }

  /**
   * Generate gradient colors for charts
   */
  generateGradient(ctx, color1, color2) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, color1);
    gradient.addColorStop(1, color2);
    return gradient;
  }

  /**
   * Get default chart options with LLVM Advisor styling
   */
  getDefaultOptions(type = 'default') {
    const baseOptions = {
      responsive : true,
      maintainAspectRatio : false,
      plugins : {
        legend : {
          display : true,
          position : 'top',
          align : 'start',
          labels : {
            padding : 20,
            usePointStyle : true,
            font : {size : 11, weight : '500'}
          }
        },
        tooltip : {
          backgroundColor : 'rgba(17, 24, 39, 0.95)',
          titleColor : '#f9fafb',
          bodyColor : '#f3f4f6',
          borderColor : '#374151',
          borderWidth : 1,
          cornerRadius : 8,
          displayColors : true,
          padding : 12,
          titleFont : {size : 12, weight : '600'},
          bodyFont : {size : 11}
        }
      },
      interaction : {intersect : false, mode : 'index'}
    };

    // Type-specific options
    switch (type) {
    case 'bar':
      return {
        ...baseOptions,
        scales : {
          x : {grid : {display : false}, border : {display : false}},
          y : {
            beginAtZero : true,
            grid : {color : '#f3f4f6', drawBorder : false},
            border : {display : false}
          }
        }
      };

    case 'line':
      return {
        ...baseOptions,
        elements : {
          point : {radius : 4, hoverRadius : 6, borderWidth : 2},
          line : {tension : 0.4, borderWidth : 2}
        },
        scales : {
          x : {grid : {display : false}, border : {display : false}},
          y : {
            beginAtZero : true,
            grid : {color : '#f3f4f6', drawBorder : false},
            border : {display : false}
          }
        }
      };

    case 'doughnut':
    case 'pie':
      return {
        ...baseOptions,
        cutout : type === 'doughnut' ? '60%' : 0,
        plugins : {
          ...baseOptions.plugins,
          legend : {...baseOptions.plugins.legend, position : 'bottom'}
        }
      };

    default:
      return baseOptions;
    }
  }

  /**
   * Create a loading placeholder for charts
   */
  createLoadingPlaceholder(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas)
      return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw loading spinner
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = 20;

    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    // Create animated spinner effect
    const drawSpinner = (rotation) => {
      ctx.clearRect(0, 0, width, height);

      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, rotation, rotation + Math.PI * 1.5);
      ctx.stroke();

      // Add loading text
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Loading chart...', centerX, centerY + radius + 20);
    };

    let rotation = 0;
    const interval = setInterval(() => {
      rotation += 0.1;
      drawSpinner(rotation);
    }, 50);

    // Store interval reference for cleanup
    canvas.dataset.loadingInterval = interval;
  }

  /**
   * Clear loading placeholder
   */
  clearLoadingPlaceholder(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas)
      return;

    const interval = canvas.dataset.loadingInterval;
    if (interval) {
      clearInterval(interval);
      delete canvas.dataset.loadingInterval;
    }

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  /**
   * Create a chart with error state
   */
  showChartError(canvasId, message = 'Failed to load chart data') {
    const canvas = document.getElementById(canvasId);
    if (!canvas)
      return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw error icon and message
    const centerX = width / 2;
    const centerY = height / 2;

    // Error icon (triangle with exclamation)
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - 15);
    ctx.lineTo(centerX - 12, centerY + 10);
    ctx.lineTo(centerX + 12, centerY + 10);
    ctx.closePath();
    ctx.fill();

    // Exclamation mark
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('!', centerX, centerY + 5);

    // Error message
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText(message, centerX, centerY + 35);
  }

  /**
   * Animate chart on data update
   */
  animateChart(chart, newData) {
    if (!chart || !newData)
      return;

    // Update data with animation
    chart.data = newData;
    chart.update('active');
  }

  /**
   * Export chart as image
   */
  exportChart(canvasId, filename = 'chart.png') {
    const canvas = document.getElementById(canvasId);
    if (!canvas)
      return;

    const link = document.createElement('a');
    link.download = filename;
    link.href = canvas.toDataURL();
    link.click();
  }

  /**
   * Get responsive font size based on container
   */
  getResponsiveFontSize(container) {
    const width = container.offsetWidth;
    if (width < 400)
      return 10;
    if (width < 600)
      return 11;
    return 12;
  }

  /**
   * Format numbers for chart labels
   */
  formatChartNumber(value, type = 'default') {
    switch (type) {
    case 'bytes':
      return this.formatBytes(value);
    case 'time':
      return this.formatTime(value);
    case 'percentage':
      return `${value.toFixed(1)}%`;
    case 'compact':
      if (value >= 1000000)
        return `${(value / 1000000).toFixed(1)}M`;
      if (value >= 1000)
        return `${(value / 1000).toFixed(1)}K`;
      return value.toString();
    default:
      return value.toLocaleString();
    }
  }

  formatBytes(bytes) {
    if (bytes === 0)
      return '0 B';
    const k = 1024;
    const sizes = [ 'B', 'KB', 'MB', 'GB' ];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  }

  formatTime(ms) {
    if (ms < 1000)
      return `${ms}ms`;
    if (ms < 60000)
      return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  }
}
