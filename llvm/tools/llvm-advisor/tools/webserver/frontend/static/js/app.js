// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Main Application Controller
 * Orchestrates the entire LLVM Advisor dashboard application
 */

import {ApiClient} from './api-client.js';
import {CompilationUnitManager} from './compilation-unit-manager.js';
import {Dashboard} from './dashboard.js';
import {Explorer} from './explorer.js';
import {TabManager} from './tab-manager.js';
import {Utils} from './utils.js';

class LLVMAdvisorApp {
  constructor() {
    this.apiClient = new ApiClient();
    this.tabManager = new TabManager();
    this.dashboard = new Dashboard(this.apiClient);
    this.explorer = new Explorer(this.apiClient, this);
    this.unitManager = new CompilationUnitManager(this.apiClient);
    this.performanceManager = null;

    this.currentUnit = null;
    this.appData = null;

    this.init();
  }

  async init() {
    try {
      console.log('Initializing LLVM Advisor Dashboard...');

      // Show loading screen
      this.showLoadingScreen();

      // Initialize components
      await this.initializeComponents();

      // Load initial data
      await this.loadInitialData();

      // Setup event listeners
      this.setupEventListeners();

      // Hide loading screen and show app
      this.hideLoadingScreen();

      console.log('Dashboard initialized successfully');

    } catch (error) {
      console.error('Failed to initialize dashboard:', error);
      this.showError(
          'Failed to initialize dashboard. Please check your connection and try again.');
    }
  }

  async initializeComponents() {
    // Initialize tab manager
    this.tabManager.init(
        {onTabChange : (tabId) => this.handleTabChange(tabId)});

    // Initialize compilation unit manager
    await this.unitManager.init(
        {onUnitChange : (unitName) => this.handleUnitChange(unitName)});

    // Initialize dashboard
    this.dashboard.init();

    // Initialize explorer
    this.explorer.init();

    // Initialize performance manager (lazy loaded)
    if (window.PerformanceManager) {
      this.performanceManager = new window.PerformanceManager();
      window.performanceManager = this.performanceManager;
    }
  }

  async loadInitialData() {
    try {
      // Check API health
      const healthStatus = await this.apiClient.getHealth();
      this.updateConnectionStatus(healthStatus.success);

      if (!healthStatus.success) {
        throw new Error('API server is not responding');
      }

      // Get available compilation units
      const unitsResponse = await this.apiClient.getUnits();
      if (unitsResponse.success) {
        this.unitManager.updateUnits(unitsResponse.data.units);

        // Select the most recent unit (first in list, as they're sorted by
        // recency)
        if (unitsResponse.data.units.length > 0) {
          this.currentUnit = unitsResponse.data.units[0].name;
          this.unitManager.selectUnit(this.currentUnit);

          // Load dashboard data for the selected unit
          await this.loadDashboardData();
        }
      }

    } catch (error) {
      console.error('Error loading initial data:', error);
      this.updateConnectionStatus(false);
      throw error;
    }
  }

  async loadDashboardData() {
    if (!this.currentUnit)
      return;

    try {
      console.log(`Loading dashboard data for unit: ${this.currentUnit}`);

      // Load summary data with error handling
      let summaryResponse;
      try {
        summaryResponse = await this.apiClient.getSummary();
        if (!summaryResponse.success) {
          console.warn('Summary API failed:', summaryResponse.error);
          summaryResponse = {success : false, data : null};
        }
      } catch (error) {
        console.warn('Summary API error:', error);
        summaryResponse = {success : false, data : null};
      }

      // Load unit detail to get compilation unit file count
      let unitDetail;
      try {
        unitDetail = await this.apiClient.getUnitDetail(this.currentUnit);
        if (!unitDetail.success) {
          console.warn('Unit detail API failed:', unitDetail.error);
          unitDetail = {success : false, data : null};
        }
      } catch (error) {
        console.warn('Unit detail API error:', error);
        unitDetail = {success : false, data : null};
      }

      // Load specialized data for dashboard with error handling
      const [remarksOverview, remarksPasses, diagnosticsOverview,
             compilationPhasesOverview, compilationPhasesBindings,
             binarySizeOverview, buildDependencies, versionInfo] =
          await Promise.allSettled([
            this.apiClient.getRemarksOverview().catch(
                e => ({success : false, error : e.message})),
            this.apiClient.getRemarksPasses().catch(
                e => ({success : false, error : e.message})),
            this.apiClient.getDiagnosticsOverview().catch(
                e => ({success : false, error : e.message})),
            this.apiClient.getFTimeReport().catch(
                e => ({success : false, error : e.message})),
            this.apiClient.getCompilationPhasesBindings().catch(
                e => ({success : false, error : e.message})),
            this.apiClient.getBinarySizeOverview().catch(
                e => ({success : false, error : e.message})),
            this.apiClient.getBuildDependencies().catch(
                e => ({success : false, error : e.message})),
            this.apiClient.getVersionInfo().catch(
                e => ({success : false, error : e.message}))
          ]);

      // Prepare dashboard data extraction
      const dashboardData = {
        summary : summaryResponse.success ? summaryResponse.data : null,
        unitDetail : unitDetail.success ? unitDetail.data : null,
        remarks : remarksOverview.status === 'fulfilled' &&
                          remarksOverview.value.success
                      ? remarksOverview.value.data
                      : null,
        remarksPasses :
            remarksPasses.status === 'fulfilled' && remarksPasses.value.success
                ? remarksPasses.value.data
                : null,
        diagnostics : diagnosticsOverview.status === 'fulfilled' &&
                              diagnosticsOverview.value.success
                          ? diagnosticsOverview.value.data
                          : null,
        compilationPhases : compilationPhasesOverview.status === 'fulfilled' &&
                                    compilationPhasesOverview.value.success
                                ? compilationPhasesOverview.value.data
                                : null,
        compilationPhasesBindings :
            compilationPhasesBindings.status === 'fulfilled' &&
                    compilationPhasesBindings.value.success
                ? compilationPhasesBindings.value.data
                : null,
        binarySize : binarySizeOverview.status === 'fulfilled' &&
                             binarySizeOverview.value.success
                         ? binarySizeOverview.value.data
                         : null,
        buildDependencies : buildDependencies.status === 'fulfilled' &&
                                    buildDependencies.value.success
                                ? buildDependencies.value.data
                                : null,
        versionInfo :
            versionInfo.status === 'fulfilled' && versionInfo.value.success
                ? versionInfo.value.data
                : null
      };

      // Update dashboard
      this.dashboard.updateData(dashboardData);

      this.appData = dashboardData;

    } catch (error) {
      console.error('Error loading dashboard data:', error);
      this.showError(
          'Failed to load dashboard data. Some sections may not be available.');
    }
  }

  setupEventListeners() {
    // Global error handler
    window.addEventListener('error', (event) => {
      console.error('Global error:', event.error);
      this.showError('An unexpected error occurred. Please refresh the page.');
    });

    // Handle connection issues
    window.addEventListener('online', () => {
      this.updateConnectionStatus(true);
      this.hideError();
    });

    window.addEventListener('offline', () => {
      this.updateConnectionStatus(false);
      this.showError(
          'You are currently offline. Some features may not work properly.');
    });

    // Auto-refresh data every 30 seconds if on dashboard tab
    setInterval(() => {
      if (this.tabManager.getCurrentTab() === 'dashboard' && this.currentUnit) {
        this.refreshCurrentData();
      }
    }, 30000);
  }

  async handleTabChange(tabId) {
    console.log(`📱 Switching to tab: ${tabId}`);

    try {
      switch (tabId) {
      case 'dashboard':
        if (this.currentUnit && !this.appData) {
          await this.loadDashboardData();
        }
        break;
      case 'explorer':
        await this.explorer.onActivate();
        break;
      case 'diagnostics':
        // Future: Load diagnostics-specific data
        break;
      case 'performance':
        if (this.performanceManager) {
          await this.performanceManager.initialize();
        }
        break;
      }
    } catch (error) {
      console.error(`Error switching to tab ${tabId}:`, error);
      this.showError(`Failed to load ${tabId} data.`);
    }
  }

  async handleUnitChange(unitName) {
    if (unitName === this.currentUnit)
      return;

    console.log(`Switching to compilation unit: ${unitName}`);
    this.currentUnit = unitName;

    // Clear existing data
    this.appData = null;

    // Reload data for new unit based on current tab
    const currentTab = this.tabManager.getCurrentTab();
    if (currentTab === 'dashboard') {
      await this.loadDashboardData();
    } else if (currentTab === 'explorer') {
      // Reload explorer files for new unit
      await this.explorer.loadAvailableFiles();
    }

    // If performance tab is active, notify performance manager
    if (currentTab === 'performance' && this.performanceManager) {
      this.performanceManager.onUnitChanged(unitName);
    }
  }

  async refreshCurrentData() {
    try {
      const currentTab = this.tabManager.getCurrentTab();

      if (currentTab === 'dashboard') {
        await this.loadDashboardData();
      }

      // Update last refresh time
      this.updateLastRefreshTime();

    } catch (error) {
      console.error('Error refreshing data:', error);
      // Don't show error for background refresh failures
    }
  }

  showLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    const app = document.getElementById('app');

    if (loadingScreen)
      loadingScreen.classList.remove('hidden');
    if (app)
      app.classList.add('hidden');
  }

  hideLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    const app = document.getElementById('app');

    if (loadingScreen)
      loadingScreen.classList.add('hidden');
    if (app)
      app.classList.remove('hidden');
  }

  showError(message) {
    const alertBanner = document.getElementById('alert-banner');
    const alertMessage = document.getElementById('alert-message');

    if (alertBanner && alertMessage) {
      alertMessage.textContent = message;
      alertBanner.classList.remove('hidden');
      alertBanner.classList.remove('bg-blue-50', 'border-blue-400');
      alertBanner.classList.add('bg-red-50', 'border-red-400');

      const messageElement = alertBanner.querySelector('p');
      if (messageElement) {
        messageElement.classList.remove('text-blue-700');
        messageElement.classList.add('text-red-700');
      }
    }
  }

  hideError() {
    const alertBanner = document.getElementById('alert-banner');
    if (alertBanner) {
      alertBanner.classList.add('hidden');
    }
  }

  showInfo(message) {
    const alertBanner = document.getElementById('alert-banner');
    const alertMessage = document.getElementById('alert-message');

    if (alertBanner && alertMessage) {
      alertMessage.textContent = message;
      alertBanner.classList.remove('hidden');
      alertBanner.classList.remove('bg-red-50', 'border-red-400');
      alertBanner.classList.add('bg-blue-50', 'border-blue-400');

      const messageElement = alertBanner.querySelector('p');
      if (messageElement) {
        messageElement.classList.remove('text-red-700');
        messageElement.classList.add('text-blue-700');
      }
    }
  }

  updateConnectionStatus(isConnected) {
    const statusIndicator = document.getElementById('status-indicator');
    if (!statusIndicator)
      return;

    const dot = statusIndicator.querySelector('div');
    const text = statusIndicator.querySelector('span');

    if (isConnected) {
      dot.className = 'h-2 w-2 bg-success rounded-full';
      text.textContent = 'Connected';
      text.className = 'text-sm text-gray-600';
    } else {
      dot.className = 'h-2 w-2 bg-error rounded-full';
      text.textContent = 'Disconnected';
      text.className = 'text-sm text-red-600';
    }
  }

  updateLastRefreshTime() {
    const now = new Date().toLocaleTimeString();
    console.log(`Data refreshed at ${now}`);
  }
}

// Initialize the application when DOM is loaded
document.addEventListener(
    'DOMContentLoaded',
    () => { window.llvmAdvisorApp = new LLVMAdvisorApp(); });

export {LLVMAdvisorApp};
