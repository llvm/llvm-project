// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Dashboard
 * Main dashboard logic and data visualization
 */

import {ChartComponents} from './chart-components.js';
import {Utils} from './utils.js';

export class Dashboard {
  constructor(apiClient) {
    this.apiClient = apiClient;
    this.chartComponents = new ChartComponents();
    this.charts = {};
    this.currentData = null;
    this.isInitialized = false;
  }

  /**
   * Initialize the dashboard
   */
  init() {
    console.log('📊 Initializing dashboard...');

    // Initialize chart components
    this.chartComponents.init();

    this.isInitialized = true;
    console.log('Dashboard initialized');
  }

  /**
   * Update dashboard with new data
   */
  async updateData(data) {
    if (!this.isInitialized) {
      console.warn('Dashboard not initialized, cannot update data');
      return;
    }

    this.currentData = data;

    try {
      // Update metrics cards
      this.updateMetricsCards(data);

      // Update charts
      await this.updateCharts(data);

      // Update insights and recommendations
      this.updateInsights(data);

      console.log('Dashboard updated successfully');

    } catch (error) {
      console.error('Failed to update dashboard:', error);
      this.showDashboardError('Failed to update dashboard data');
    }
  }

  /**
   * Update the key metrics cards
   */
  updateMetricsCards(data) {
    const summary = data.summary;
    const diagnostics = data.diagnostics;
    const remarks = data.remarks;
    const buildDependencies = data.buildDependencies;

    // Total Source Files (from dependencies - count unique source files for
    // current unit only)
    let totalSourceFiles = 0;

    // Get current unit name to filter data
    const currentUnitName = this.currentData?.unitDetail?.name ||
                            document.getElementById('unit-selector')?.value;

    console.log('Calculating total files for unit:', currentUnitName);

    if (buildDependencies && currentUnitName) {
      // Look for the specific unit's data
      if (buildDependencies.units && buildDependencies.units[currentUnitName]) {
        const unit = buildDependencies.units[currentUnitName];
        console.log(`Processing unit ${currentUnitName}:`, unit);

        // Look for summary_stats in the current unit
        if (unit.summary_stats && unit.summary_stats.unique_sources) {
          console.log(`Found unique_sources in summary_stats: ${
              unit.summary_stats.unique_sources}`);
          totalSourceFiles = unit.summary_stats.unique_sources;
        }
        // Fallback: look in metadata
        else if (unit.metadata && unit.metadata.unique_sources) {
          console.log(`Found unique_sources in metadata: ${
              unit.metadata.unique_sources}`);
          totalSourceFiles = unit.metadata.unique_sources;
        }
        // Check individual files and their metadata for unique_sources count
        else if (unit.files && Array.isArray(unit.files)) {
          console.log(`Checking files array for unit ${
              currentUnitName}, length: ${unit.files.length}`);
          const uniqueSourceFiles = new Set();

          unit.files.forEach(file => {
            // Check if this is a dependencies file and has metadata with
            // unique_sources
            if (file.metadata && file.metadata.unique_sources) {
              // For dependencies files, use the unique_sources count
              console.log(`Found unique_sources in file metadata: ${
                  file.metadata.unique_sources}`);
              totalSourceFiles =
                  Math.max(totalSourceFiles, file.metadata.unique_sources);
            }
            // Also count sources directory files if available
            else if (file.file_path && file.file_path.includes('/sources/')) {
              const fileName = file.file_path.split('/').pop();
              if (fileName && !fileName.startsWith('.')) {
                uniqueSourceFiles.add(fileName);
              }
            }
          });

          // If no unique_sources found in metadata, use the count from sources
          // directory
          if (totalSourceFiles === 0 && uniqueSourceFiles.size > 0) {
            totalSourceFiles = uniqueSourceFiles.size;
            console.log(
                `Counted ${uniqueSourceFiles.size} unique sources from files`);
          }
        }
      }
    }

    // Fallback: if no current unit or no data found, show 0
    if (!currentUnitName || totalSourceFiles === 0) {
      console.log(
          'No current unit selected or no source files found, showing 0');
      totalSourceFiles = 0;
    }

    console.log('Calculated source files:', totalSourceFiles);

    this.updateMetricCard('metric-total-files',
                          Utils.formatNumber(totalSourceFiles));

    // Success Rate
    const successRate = summary?.success_rate || 0;
    this.updateMetricCard('metric-success-rate', `${successRate.toFixed(1)}%`);

    // Total Errors
    const totalErrors = summary?.errors || 0;
    this.updateMetricCard('metric-total-errors',
                          Utils.formatNumber(totalErrors));

    // Compilation Phases (from compilation phases bindings)
    const compilationPhases =
        data.compilationPhasesBindings?.summary?.total_bindings || 0;
    this.updateMetricCard('metric-timing-phases',
                          Utils.formatNumber(compilationPhases));
  }

  /**
   * Update a single metric card
   */
  updateMetricCard(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
      // Add animation
      element.style.opacity = '0.6';
      setTimeout(() => {
        element.textContent = value;
        element.style.opacity = '1';
      }, 150);
    }
  }

  /**
   * Update all charts
   */
  async updateCharts(data) {
    const summary = data.summary;
    const diagnostics = data.diagnostics;
    const compilationPhases = data.compilationPhases;
    const binarySize = data.binarySize;
    const remarks = data.remarks;
    const remarksPasses = data.remarksPasses;
    const versionInfo = data.versionInfo;

    // Remarks Distribution Chart
    if (remarks || remarksPasses) {
      await this.updateRemarksDistributionChart(remarks, remarksPasses);
    }

    // Diagnostic Levels Chart
    if (diagnostics?.by_level) {
      await this.updateDiagnosticLevelsChart(diagnostics.by_level);
    }

    // Compilation Info Table
    if (compilationPhases || versionInfo) {
      this.updateCompilationInfoTable(compilationPhases, versionInfo);
    }

    // Binary Size Chart
    if (binarySize?.section_breakdown) {
      await this.updateBinarySizeChart(binarySize.section_breakdown);
    }
  }

  /**
   * Update remarks distribution chart
   */
  async updateRemarksDistributionChart(remarksData, remarksPassesData) {
    const canvas = document.getElementById('remarks-distribution-chart');
    if (!canvas)
      return;

    // Try to get remarks distribution by type from passes data
    let chartData = [];

    if (remarksPassesData && remarksPassesData.passes) {
      // Get top optimization passes by count
      const passesArray =
          Object.entries(remarksPassesData.passes)
              .sort(([, a ], [, b ]) => (b.count || 0) - (a.count || 0))
              .slice(0, 8); // Top 8 passes

      if (passesArray.length > 0) {
        const labels = passesArray.map(
            ([ name ]) =>
                name.length > 20 ? name.substring(0, 20) + '...' : name);
        const counts = passesArray.map(([, data ]) => data.count || 0);

        const colors = [
          '#3b82f6', '#1e40af', '#1d4ed8', '#2563eb', '#60a5fa', '#93c5fd',
          '#dbeafe', '#eff6ff'
        ];

        const config = {
          type : 'doughnut',
          data : {
            labels : labels,
            datasets : [ {
              data : counts,
              backgroundColor : colors.slice(0, labels.length),
              borderWidth : 2,
              borderColor : '#ffffff'
            } ]
          },
          options : {
            responsive : true,
            maintainAspectRatio : false,
            plugins : {
              legend : {
                position : 'bottom',
                labels :
                    {padding : 20, usePointStyle : true, font : {size : 11}}
              },
              tooltip : {
                callbacks : {
                  label : (context) => {
                    const label = context.label;
                    const value = context.parsed;
                    const total =
                        context.dataset.data.reduce((a, b) => a + b, 0);
                    const percentage = ((value / total) * 100).toFixed(1);
                    return `${label}: ${value} remarks (${percentage}%)`;
                  }
                }
              }
            }
          }
        };

        // Destroy existing chart if it exists
        if (this.charts.remarksDistribution) {
          this.charts.remarksDistribution.destroy();
        }

        this.charts.remarksDistribution = new Chart(canvas, config);
        return;
      }
    }

    // Fallback: Show placeholder when no remarks data is available
    this.showPlaceholderChart('remarks-distribution-chart',
                              'No Remarks Data Available');
  }

  /**
   * Update diagnostic levels chart
   */
  async updateDiagnosticLevelsChart(diagnosticData) {
    const canvas = document.getElementById('diagnostic-levels-chart');
    if (!canvas)
      return;

    // Extract diagnostic level counts
    const levels = [ 'error', 'warning', 'note', 'info' ];
    const colors = {
      'error' : '#ef4444',
      'warning' : '#f59e0b',
      'note' : '#3b82f6',
      'info' : '#10b981'
    };

    const data = levels.map(level => {
      if (diagnosticData.by_level && diagnosticData.by_level[level]) {
        return diagnosticData.by_level[level];
      }
      // Fallback: look for the data in different format
      return diagnosticData[level] || 0;
    });

    const config = {
      type : 'bar',
      data : {
        labels : levels.map(level => Utils.capitalize(level)),
        datasets : [ {
          label : 'Diagnostics',
          data : data,
          backgroundColor : levels.map(level => colors[level]),
          borderRadius : 4,
          borderWidth : 0
        } ]
      },
      options : {
        responsive : true,
        maintainAspectRatio : false,
        plugins : {
          legend : {display : false},
          tooltip : {
            callbacks : {
              label : (context) => {
                return `${context.label}: ${context.parsed.y} issues`;
              }
            }
          }
        },
        scales : {
          y : {beginAtZero : true, ticks : {precision : 0}},
          x : {grid : {display : false}}
        }
      }
    };

    if (this.charts.diagnosticLevels) {
      this.charts.diagnosticLevels.destroy();
    }

    this.charts.diagnosticLevels = new Chart(canvas, config);
  }

  /**
   * Update compilation info table
   */
  updateCompilationInfoTable(compilationData, versionInfo) {
    const container = document.getElementById('compilation-info-table');
    if (!container)
      return;

    // Check if we have valid compilation data
    if (!compilationData && !versionInfo) {
      container.innerHTML = `
                <div class="text-center py-4 text-gray-500">
                    <svg class="mx-auto h-8 w-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                    </svg>
                    No compilation information available
                </div>
            `;
      return;
    }

    // Extract compilation info from the actual parsed data
    let compilationInfo = [];

    // Get current unit name to filter data
    const currentUnitName = this.currentData?.unitDetail?.name ||
                            document.getElementById('unit-selector')?.value;

    console.log('Processing compilation info for unit:', currentUnitName);

    // Use Map to store unique info for current unit only
    const infoMap = new Map();

    const addOrAggregateInfo =
        (label, value, icon, shouldAggregate = false) => {
          if (!value)
            return;

          if (shouldAggregate && infoMap.has(label)) {
            // Aggregate numeric values (for timing info within this unit)
            const existing = infoMap.get(label);
            if (typeof value === 'string' && value.endsWith('s')) {
              // Parse timing values like "0.0171s"
              const currentTime = parseFloat(value.replace('s', ''));
              const existingTime = parseFloat(existing.value.replace('s', ''));
              if (!isNaN(currentTime) && !isNaN(existingTime)) {
                existing.value = `${(existingTime + currentTime).toFixed(4)}s`;
                return;
              }
            }
            // For non-timing aggregatable values, prefer the first one
            return;
          } else if (!shouldAggregate && infoMap.has(label)) {
            // For non-aggregatable values, just ignore duplicates
            return;
          }

          infoMap.set(label, {label, value, icon});
        };

    // Debug: log the structure to understand what we have
    console.log('FTime report data structure:', compilationData);
    console.log('Version info data structure:', versionInfo);

    // Process ftime-report data (compilation timing) - only for current unit
    if (compilationData && compilationData.units) {
      Object.entries(compilationData.units).forEach(([ unitName, unit ]) => {
        // Skip if not the current unit
        if (currentUnitName && unitName !== currentUnitName) {
          console.log(
              `Skipping unit ${unitName}, current unit is ${currentUnitName}`);
          return;
        }
        if (unit.files && Array.isArray(unit.files)) {
          unit.files.forEach(file => {
            console.log('Processing ftime file:', file);

            // Process ftime report file
            if (file.file_name?.includes('ftime') ||
                file.file_path?.includes('ftime-report')) {
              console.log('Found ftime file metadata:', file.metadata);

              // Extract compilation timing information
              if (file.metadata) {
                // Add timing information (aggregate timing values, keep first
                // for others)
                if (file.metadata.total_execution_time !== undefined) {
                  addOrAggregateInfo(
                      'Total Execution Time',
                      `${file.metadata.total_execution_time.toFixed(4)}s`,
                      'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z', true);
                }

                if (file.metadata.timing_entries_count !== undefined) {
                  addOrAggregateInfo(
                      'Timing Phases', file.metadata.timing_entries_count,
                      'M9 5H7a2 2 0 00-2 2v6a2 2 0 002 2h6a2 2 0 002-2V7a2 2 0 00-2-2h-2m-2 0V3a2 2 0 012-2h2a2 2 0 012 2v2m-6 0h6',
                      false);
                }

                if (file.metadata.top_time_consumer !== undefined &&
                    file.metadata.top_time_consumer) {
                  addOrAggregateInfo('Top Time Consumer',
                                     file.metadata.top_time_consumer,
                                     'M13 10V3L4 14h7v7l9-11h-7z', false);
                }
              }
            }
          });
        }
      });
    }

    // Process version-info data (clang version, target, etc.) - only for
    // current unit
    if (versionInfo && versionInfo.units) {
      Object.entries(versionInfo.units).forEach(([ unitName, unit ]) => {
        // Skip if not the current unit
        if (currentUnitName && unitName !== currentUnitName) {
          console.log(`Skipping version-info unit ${
              unitName}, current unit is ${currentUnitName}`);
          return;
        }
        if (unit.files && Array.isArray(unit.files)) {
          unit.files.forEach(file => {
            console.log('Processing version info file:', file);

            // Process version info file
            if (file.file_name?.includes('version') ||
                file.file_path?.includes('version-info')) {
              console.log('Found version info file metadata:', file.metadata);

              // Extract version information
              if (file.metadata || file.data) {
                const metadata = file.metadata || {};
                const data = file.data || {};

                // Extract unique compilation info (no aggregation needed for
                // compiler info)
                const clangVersion =
                    metadata.clang_version || data.clang_version;
                addOrAggregateInfo(
                    'Clang Version', clangVersion,
                    'M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 713.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 713.138-3.138z',
                    false);

                const target = metadata.target || data.target;
                addOrAggregateInfo(
                    'Target Architecture', target,
                    'M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z',
                    false);

                const threadModel = metadata.thread_model || data.thread_model;
                addOrAggregateInfo('Thread Model', threadModel,
                                   'M19 11H5m14-7l2 7-2 7M5 18l-2-7 2-7',
                                   false);
              }
            }
          });
        }
      });
    }

    // Add timestamp from unit detail if available
    if (this.currentData && this.currentData.unitDetail) {
      const unitDetail = this.currentData.unitDetail;
      console.log('Unit detail for timestamp:', unitDetail);

      if (unitDetail.timestamp) {
        const timestamp = new Date(unitDetail.timestamp);
        addOrAggregateInfo('Compilation Time', timestamp.toLocaleString(),
                           'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z',
                           false);
      }
    }

    // Convert map to array for display
    compilationInfo = Array.from(infoMap.values());

    console.log('Final compilation info for unit', currentUnitName, ':',
                compilationInfo);

    // Only show table if we have actual data
    if (compilationInfo.length === 0) {
      container.innerHTML = `
                <div class="text-center py-4 text-gray-500">
                    <svg class="mx-auto h-8 w-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                    </svg>
                    No compilation information available from parsed data
                </div>
            `;
    } else {
      // Create table HTML with actual parsed data
      container.innerHTML = `
                <div class="overflow-hidden">
                    <div class="grid grid-cols-1 gap-3">
                        ${
          compilationInfo
              .map(info => `
                            <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                <div class="flex items-center space-x-3">
                                    <svg class="h-5 w-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${
                       info.icon}"></path>
                                    </svg>
                                    <span class="text-sm font-medium text-gray-700">${
                       info.label}</span>
                                </div>
                                <span class="text-sm font-semibold text-gray-900">${
                       info.value}</span>
                            </div>
                        `)
              .join('')}
                    </div>
                </div>
            `;
    }
  }

  /**
   * Update binary size chart
   */
  async updateBinarySizeChart(sizeData) {
    const canvas = document.getElementById('binary-size-chart');
    if (!canvas)
      return;

    // Check if we have valid size data
    if (!sizeData || Object.keys(sizeData).length === 0) {
      this.showPlaceholderChart('binary-size-chart', 'No Binary Size Data');
      return;
    }

    // Get top sections by size
    const sections = Object.entries(sizeData)
                         .filter(([, size ]) => size > 0)
                         .sort(([, a ], [, b ]) => b - a)
                         .slice(0, 8);

    if (sections.length === 0) {
      this.showPlaceholderChart('binary-size-chart', 'No Binary Size Data');
      return;
    }

    const labels = sections.map(
        ([ name ]) =>
            Utils.formatSectionName ? Utils.formatSectionName(name) : name);
    const sizes = sections.map(([, size ]) => size);

    const colors =
        this.chartComponents.generateColors
            ? this.chartComponents.generateColors(labels.length, 'blue')
            : [
                '#3b82f6', '#1e40af', '#1d4ed8', '#2563eb', '#60a5fa',
                '#93c5fd', '#dbeafe', '#eff6ff'
              ];

    const config = {
      type : 'pie',
      data : {
        labels : labels,
        datasets : [ {
          data : sizes,
          backgroundColor : colors.slice(0, labels.length),
          borderWidth : 2,
          borderColor : '#ffffff'
        } ]
      },
      options : {
        responsive : true,
        maintainAspectRatio : false,
        plugins : {
          legend : {
            position : 'bottom',
            labels : {padding : 20, usePointStyle : true}
          },
          tooltip : {
            callbacks : {
              label : (context) => {
                const label = context.label;
                const value = context.parsed;
                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                const percentage = ((value / total) * 100).toFixed(1);
                const formattedSize = this.formatBytes ? this.formatBytes(value)
                                                       : `${value} bytes`;
                return `${label}: ${formattedSize} (${percentage}%)`;
              }
            }
          }
        }
      }
    };

    if (this.charts.binarySize) {
      this.charts.binarySize.destroy();
    }

    this.charts.binarySize = new Chart(canvas, config);
  }

  /**
   * Show placeholder chart for missing data
   */
  showPlaceholderChart(canvasId, message) {
    const canvas = document.getElementById(canvasId);
    if (!canvas)
      return;

    const chartKey = canvasId.replace('-chart', '')
                         .replace(/-([a-z])/g, (g) => g[1].toUpperCase());

    const config = {
      type : 'doughnut',
      data : {
        labels : [ message ],
        datasets :
            [ {data : [ 1 ], backgroundColor : [ '#f3f4f6' ], borderWidth : 0} ]
      },
      options : {
        responsive : true,
        maintainAspectRatio : false,
        plugins : {legend : {display : false}, tooltip : {enabled : false}}
      }
    };

    if (this.charts[chartKey]) {
      this.charts[chartKey].destroy();
    }

    this.charts[chartKey] = new Chart(canvas, config);
  }

  /**
   * Format bytes helper function
   */
  formatBytes(bytes) {
    if (bytes === 0)
      return '0 B';
    const k = 1024;
    const sizes = [ 'B', 'KB', 'MB', 'GB' ];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  /**
   * Update insights and recommendations
   */
  updateInsights(data) {
    this.updateRemarksSummary(data);
    this.updateOptimizationPasses(data);
  }

  /**
   * Update remarks summary
   */
  updateRemarksSummary(data) {
    const container = document.getElementById('remarks-summary-list');
    if (!container)
      return;

    const remarks = data.remarks;

    container.innerHTML = '';

    if (!remarks || !remarks.totals) {
      container.innerHTML = `
                <div class="text-center py-4 text-gray-500">
                    <svg class="mx-auto h-8 w-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                    </svg>
                    No optimization remarks available
                </div>
            `;
      return;
    }

    const summaryItems = [
      {
        title : 'Total Optimization Remarks',
        value : remarks.totals.remarks || 0,
        description : 'Total number of optimization opportunities found',
        icon : 'M13 10V3L4 14h7v7l9-11h-7z',
        color : 'bg-blue-50 border-blue-200'
      },
      {
        title : 'Unique Optimization Passes',
        value : remarks.totals.unique_passes || 0,
        description :
            'Number of different optimization passes that generated remarks',
        icon :
            'M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z',
        color : 'bg-green-50 border-green-200'
      },
      {
        title : 'Functions with Remarks',
        value : remarks.totals.unique_functions || 0,
        description : 'Functions that have optimization remarks',
        icon :
            'M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zM21 5a2 2 0 012 2v12a4 4 0 01-4 4h-4a2 2 0 01-2-2V5a2 2 0 012-2h4z',
        color : 'bg-purple-50 border-purple-200'
      }
    ];

    summaryItems.forEach(item => {
      const itemElement = document.createElement('div');
      itemElement.className =
          `flex items-start space-x-3 p-3 rounded-lg ${item.color}`;

      itemElement.innerHTML = `
                <div class="flex-shrink-0">
                    <svg class="h-5 w-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${
          item.icon}"></path>
                    </svg>
                </div>
                <div class="flex-1">
                    <h4 class="text-sm font-medium text-gray-900">${
          item.title}</h4>
                    <p class="text-sm text-gray-600 mt-1">${
          item.description}</p>
                    <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800 mt-2">${
          item.value}</span>
                </div>
            `;

      container.appendChild(itemElement);
    });
  }

  /**
   * Update optimization passes
   */
  updateOptimizationPasses(data) {
    const container = document.getElementById('optimization-passes-list');
    if (!container)
      return;

    const passesData = data.remarksPasses;

    container.innerHTML = '';

    if (!passesData || !passesData.passes) {
      container.innerHTML = `
                <div class="text-center py-4 text-gray-500">
                    <svg class="mx-auto h-8 w-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"></path>
                    </svg>
                    No optimization passes data available
                </div>
            `;
      return;
    }

    // Get top optimization passes by remark count
    const topPasses =
        Object.entries(passesData.passes)
            .sort(([, a ], [, b ]) => (b.count || 0) - (a.count || 0))
            .slice(0, 5); // Top 5 passes

    topPasses.forEach(([ passName, passInfo ]) => {
      const passElement = document.createElement('div');
      passElement.className =
          'flex items-start space-x-3 p-3 rounded-lg bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200';

      const description =
          passInfo.examples && passInfo.examples.length > 0
              ? passInfo.examples[0].message || 'Optimization pass execution'
              : 'Optimization pass with multiple improvements';

      passElement.innerHTML = `
                <div class="flex-shrink-0">
                    <svg class="h-5 w-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"></path>
                    </svg>
                </div>
                <div class="flex-1">
                    <h4 class="text-sm font-medium text-gray-900">${
          passName.replace(/-/g, ' ').replace(/\b\w/g,
                                              l => l.toUpperCase())}</h4>
                    <p class="text-sm text-gray-600 mt-1">${
          description.length > 80 ? description.substring(0, 80) + '...'
                                  : description}</p>
                    <div class="flex items-center space-x-4 mt-2">
                        <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">${
          passInfo.count || 0} remarks</span>
                        <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">${
          passInfo.unique_functions || 0} functions</span>
                    </div>
                </div>
            `;

      container.appendChild(passElement);
    });
  }

  /**
   * Generate top issues from data
   */
  generateTopIssues(data) {
    const issues = [];

    // Check for compilation errors
    if (data.summary?.errors > 0) {
      issues.push({
        severity : 'error',
        title : 'Compilation Errors',
        description : `${data.summary.errors} files failed to parse correctly`,
        count : data.summary.errors
      });
    }

    // Check for high error rate in diagnostics
    if (data.diagnostics?.by_level?.error > 10) {
      issues.push({
        severity : 'error',
        title : 'High Error Count',
        description : 'Large number of compilation errors detected',
        count : data.diagnostics.by_level.error
      });
    }

    // Check for many warnings
    if (data.diagnostics?.by_level?.warning > 50) {
      issues.push({
        severity : 'warning',
        title : 'Many Warnings',
        description :
            'High number of compiler warnings that should be addressed',
        count : data.diagnostics.by_level.warning
      });
    }

    return issues.slice(0, 5); // Return top 5 issues
  }

  /**
   * Generate optimization recommendations
   */
  generateOptimizationRecommendations(data) {
    const recommendations = [];

    // Optimization remarks suggestions
    if (data.remarks?.totals?.remarks > 0) {
      recommendations.push({
        title : 'Review Optimization Remarks',
        description : `${
            data.remarks.totals
                .remarks} optimization opportunities found in your code`,
        impact : 'Potential performance improvement'
      });
    }

    // Binary size optimization
    if (data.binarySize?.size_statistics?.total_size >
        10 * 1024 * 1024) { // > 10MB
      recommendations.push({
        title : 'Consider Binary Size Optimization',
        description :
            'Binary size is large, consider link-time optimization or unused code removal',
        impact : 'Reduce binary size'
      });
    }

    // Warning cleanup
    if (data.diagnostics?.by_level?.warning > 20) {
      recommendations.push({
        title : 'Clean Up Warnings',
        description :
            'Addressing compiler warnings can improve code quality and catch potential bugs',
        impact : 'Better code quality'
      });
    }

    return recommendations.slice(0, 4); // Return top 4 recommendations
  }

  /**
   * Get CSS class for issue severity
   */
  getIssueColorClass(severity) {
    switch (severity) {
    case 'error':
      return 'bg-red-50 border border-red-200';
    case 'warning':
      return 'bg-yellow-50 border border-yellow-200';
    case 'info':
      return 'bg-blue-50 border border-blue-200';
    default:
      return 'bg-gray-50 border border-gray-200';
    }
  }

  /**
   * Get icon for issue severity
   */
  getIssueIcon(severity) {
    const iconClass = severity === 'error'     ? 'text-red-500'
                      : severity === 'warning' ? 'text-yellow-500'
                                               : 'text-blue-500';

    const iconPath =
        severity === 'error'
            ? 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z'
            : 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z';

    return `
            <svg class="h-5 w-5 ${
        iconClass}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${
        iconPath}"></path>
            </svg>
        `;
  }

  /**
   * Show dashboard error
   */
  showDashboardError(message) {
    console.error('Dashboard error:', message);
  }

  /**
   * Clear all charts
   */
  clearCharts() {
    Object.values(this.charts).forEach(chart => {
      if (chart && typeof chart.destroy === 'function') {
        chart.destroy();
      }
    });
    this.charts = {};
  }

  /**
   * Refresh dashboard data
   */
  async refresh() {
    if (this.currentData) {
      await this.updateData(this.currentData);
    }
  }

  /**
   * Get dashboard statistics
   */
  getStats() {
    return {
      chartsActive : Object.keys(this.charts).length,
      lastUpdate : this.currentData ? new Date().toISOString() : null,
      isInitialized : this.isInitialized
    };
  }
}
