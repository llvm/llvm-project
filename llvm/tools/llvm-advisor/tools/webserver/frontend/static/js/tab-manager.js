// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Tab Manager
 * Handles tab switching and navigation in the LLVM Advisor dashboard
 */

export class TabManager {
  constructor() {
    this.currentTab = 'dashboard';
    this.tabs = new Map();
    this.onTabChangeCallback = null;
  }

  /**
   * Initialize the tab manager
   */
  init(options = {}) {
    this.onTabChangeCallback = options.onTabChange;

    // Register all tabs
    this.registerTabs();

    // Setup event listeners
    this.setupEventListeners();

    // Set initial tab state
    this.setActiveTab(this.currentTab);

    console.log('Tab manager initialized');
  }

  /**
   * Register all available tabs
   */
  registerTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
      const tabId = button.dataset.tab;
      const content = document.getElementById(`${tabId}-content`);

      if (content) {
        this.tabs.set(tabId, {
          button,
          content,
          isLoaded : tabId === 'dashboard', // Dashboard is loaded by default
          title : button.textContent.trim()
        });
      }
    });

    console.log(`📋 Registered ${this.tabs.size} tabs:`,
                Array.from(this.tabs.keys()));
  }

  /**
   * Setup event listeners for tab interactions
   */
  setupEventListeners() {
    // Handle tab button clicks
    document.addEventListener('click', (event) => {
      if (event.target.classList.contains('tab-button')) {
        event.preventDefault();
        const tabId = event.target.dataset.tab;
        if (tabId && this.tabs.has(tabId)) {
          this.switchTab(tabId);
        }
      }
    });

    // Handle keyboard navigation (Tab key to cycle through tabs)
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Tab' && event.ctrlKey) {
        event.preventDefault();
        this.switchToNextTab();
      }
    });

    // Handle URL hash changes for deep linking
    window.addEventListener('hashchange', () => { this.handleHashChange(); });

    // Set initial hash if none exists
    if (!window.location.hash && this.currentTab) {
      window.location.hash = `#${this.currentTab}`;
    }
  }

  /**
   * Switch to a specific tab
   */
  async switchTab(tabId) {
    if (!this.tabs.has(tabId) || tabId === this.currentTab) {
      return;
    }

    const previousTab = this.currentTab;

    try {
      // Update current tab
      this.currentTab = tabId;

      // Update UI
      this.setActiveTab(tabId);

      // Update URL hash
      window.location.hash = `#${tabId}`;

      // Call the tab change callback
      if (this.onTabChangeCallback) {
        await this.onTabChangeCallback(tabId, previousTab);
      }

      // Mark tab as loaded
      const tab = this.tabs.get(tabId);
      if (tab) {
        tab.isLoaded = true;
      }

      // Track tab switch for analytics
      this.trackTabSwitch(tabId, previousTab);

      console.log(`📱 Switched from ${previousTab} to ${tabId}`);

    } catch (error) {
      console.error(`Failed to switch to tab ${tabId}:`, error);

      // Revert to previous tab on error
      this.currentTab = previousTab;
      this.setActiveTab(previousTab);

      // Show error notification
      this.showTabSwitchError(tabId, error.message);
    }
  }

  /**
   * Set the visual active state for a tab
   */
  setActiveTab(tabId) {
    // Update all tab buttons
    this.tabs.forEach((tab, id) => {
      if (id === tabId) {
        // Activate current tab
        tab.button.classList.add('active');
        tab.button.classList.remove('text-gray-500', 'hover:text-gray-700',
                                    'hover:border-gray-300',
                                    'border-transparent');
        tab.button.classList.add('border-llvm-blue', 'text-llvm-blue');

        // Show current tab content
        tab.content.classList.remove('hidden');
        tab.content.classList.add('tab-transition');

      } else {
        // Deactivate other tabs
        tab.button.classList.remove('active', 'border-llvm-blue',
                                    'text-llvm-blue');
        tab.button.classList.add('border-transparent', 'text-gray-500',
                                 'hover:text-gray-700',
                                 'hover:border-gray-300');

        // Hide other tab contents
        tab.content.classList.add('hidden');
        tab.content.classList.remove('tab-transition');
      }
    });
  }

  /**
   * Switch to the next tab in sequence
   */
  switchToNextTab() {
    const tabIds = Array.from(this.tabs.keys());
    const currentIndex = tabIds.indexOf(this.currentTab);
    const nextIndex = (currentIndex + 1) % tabIds.length;
    const nextTabId = tabIds[nextIndex];

    this.switchTab(nextTabId);
  }

  /**
   * Switch to the previous tab in sequence
   */
  switchToPreviousTab() {
    const tabIds = Array.from(this.tabs.keys());
    const currentIndex = tabIds.indexOf(this.currentTab);
    const prevIndex = currentIndex === 0 ? tabIds.length - 1 : currentIndex - 1;
    const prevTabId = tabIds[prevIndex];

    this.switchTab(prevTabId);
  }

  /**
   * Handle URL hash changes for deep linking
   */
  handleHashChange() {
    const hash = window.location.hash.slice(1); // Remove the '#'

    if (hash && this.tabs.has(hash) && hash !== this.currentTab) {
      this.switchTab(hash);
    }
  }

  /**
   * Get the currently active tab
   */
  getCurrentTab() { return this.currentTab; }

  /**
   * Get information about a specific tab
   */
  getTabInfo(tabId) { return this.tabs.get(tabId); }

  /**
   * Get all registered tabs
   */
  getAllTabs() {
    const result = {};
    this.tabs.forEach((tab, id) => {
      result[id] = {
        title : tab.title,
        isLoaded : tab.isLoaded,
        isActive : id === this.currentTab
      };
    });
    return result;
  }

  /**
   * Check if a tab has been loaded
   */
  isTabLoaded(tabId) {
    const tab = this.tabs.get(tabId);
    return tab ? tab.isLoaded : false;
  }

  /**
   * Mark a tab as loaded
   */
  markTabAsLoaded(tabId) {
    const tab = this.tabs.get(tabId);
    if (tab) {
      tab.isLoaded = true;
    }
  }

  /**
   * Show loading state for a specific tab
   */
  showTabLoading(tabId) {
    const tab = this.tabs.get(tabId);
    if (tab && tab.content) {
      const loadingHtml = `
                <div class="flex items-center justify-center h-64">
                    <div class="text-center">
                        <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-llvm-blue"></div>
                        <p class="mt-2 text-gray-500">Loading ${
          tab.title}...</p>
                    </div>
                </div>
            `;

      // Store original content if not already stored
      if (!tab.originalContent) {
        tab.originalContent = tab.content.innerHTML;
      }

      tab.content.innerHTML = loadingHtml;
    }
  }

  /**
   * Hide loading state for a specific tab
   */
  hideTabLoading(tabId) {
    const tab = this.tabs.get(tabId);
    if (tab && tab.originalContent) {
      tab.content.innerHTML = tab.originalContent;
      delete tab.originalContent;
    }
  }

  /**
   * Show error state for tab switching
   */
  showTabSwitchError(tabId, errorMessage) {
    console.error(`Tab switch error for ${tabId}:`, errorMessage);

    const tab = this.tabs.get(tabId);
    if (tab) {
      alert(`Failed to switch to ${tab.title}: ${errorMessage}`);
    }
  }

  /**
   * Track tab switches for analytics/debugging
   */
  trackTabSwitch(newTab, previousTab) {
    const timestamp = new Date().toISOString();

    console.log(
        `Tab Analytics: ${previousTab} -> ${newTab} at ${timestamp}`);
  }

  /**
   * Enable/disable a specific tab
   */
  setTabEnabled(tabId, enabled) {
    const tab = this.tabs.get(tabId);
    if (tab) {
      if (enabled) {
        tab.button.removeAttribute('disabled');
        tab.button.classList.remove('opacity-50', 'cursor-not-allowed');
      } else {
        tab.button.setAttribute('disabled', 'true');
        tab.button.classList.add('opacity-50', 'cursor-not-allowed');

        // If this was the current tab, switch to another one
        if (tabId === this.currentTab) {
          const enabledTabs =
              Array.from(this.tabs.keys())
                  .filter(
                      id => id !== tabId &&
                            !this.tabs.get(id).button.hasAttribute('disabled'));

          if (enabledTabs.length > 0) {
            this.switchTab(enabledTabs[0]);
          }
        }
      }
    }
  }
}
