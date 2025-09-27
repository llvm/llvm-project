// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Compilation Unit Manager
 * Handles compilation unit selection and tracking
 */

export class CompilationUnitManager {
  constructor(apiClient) {
    this.apiClient = apiClient;
    this.units = [];
    this.currentUnit = null;
    this.onUnitChangeCallback = null;
    this.selector = null;
    this.metadata = new Map(); // Store metadata for each unit
  }

  /**
   * Initialize the compilation unit manager
   */
  async init(options = {}) {
    this.onUnitChangeCallback = options.onUnitChange;

    // Get the selector element
    this.selector = document.getElementById('unit-selector');
    if (!this.selector) {
      throw new Error('Unit selector element not found');
    }

    // Setup event listeners
    this.setupEventListeners();

    // Load compilation unit metadata
    await this.loadUnitMetadata();

    console.log('Compilation unit manager initialized');
  }

  /**
   * Setup event listeners
   */
  setupEventListeners() {
    if (this.selector) {
      this.selector.addEventListener('change', (event) => {
        const selectedUnit = event.target.value;
        if (selectedUnit && selectedUnit !== this.currentUnit) {
          this.selectUnit(selectedUnit);
        }
      });
    }
  }

  /**
   * Load compilation unit metadata from API
   */
  async loadUnitMetadata() {
    try {
      // Get units from API
      const response = await this.apiClient.getUnits();

      if (response.success && response.data.units) {
        this.units = response.data.units;

        // Sort units by most recent first
        this.units.sort((a, b) => {
          // If units have timestamps, sort by those
          if (a.timestamp && b.timestamp) {
            return new Date(b.timestamp) - new Date(a.timestamp);
          }
          // Otherwise, sort alphabetically
          return a.name.localeCompare(b.name);
        });

        // Store metadata for each unit
        this.units.forEach(unit => {
          this.metadata.set(unit.name, {
            ...unit,
            lastSelected : null,
            isRecent : this.isRecentUnit(unit)
          });
        });

        console.log(`📋 Loaded ${this.units.length} compilation units`);
        return true;
      } else {
        console.warn('No compilation units found');
        return false;
      }
    } catch (error) {
      console.error('Failed to load unit metadata:', error);
      return false;
    }
  }

  /**
   * Update the units list
   */
  updateUnits(units) {
    this.units = units || [];
    this.updateSelector();

    // Update metadata
    this.units.forEach(unit => {
      if (!this.metadata.has(unit.name)) {
        this.metadata.set(
            unit.name,
            {...unit, lastSelected : null, isRecent : this.isRecentUnit(unit)});
      }
    });
  }

  /**
   * Update the selector dropdown
   */
  updateSelector() {
    if (!this.selector)
      return;

    // Clear existing options
    this.selector.innerHTML = '';

    if (this.units.length === 0) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = 'No compilation units found';
      option.disabled = true;
      this.selector.appendChild(option);
      return;
    }

    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select a compilation unit...';
    defaultOption.disabled = true;
    this.selector.appendChild(defaultOption);

    // Add unit options
    this.units.forEach(unit => {
      const option = document.createElement('option');
      option.value = unit.name;

      // Create descriptive text
      let displayText = unit.name;

      // Add artifact count if available
      if (unit.total_files) {
        displayText += ` (${unit.total_files} artifacts)`;
      }

      // Add run timestamp if available
      if (unit.run_timestamp) {
        const formattedTime = this.formatTimestamp(unit.run_timestamp);
        displayText += ` - ${formattedTime}`;
      }

      // Add recent indicator
      if (this.isRecentUnit(unit)) {
        displayText += ' 🔥';
      }

      option.textContent = displayText;

      // Set tooltip with more information
      option.title = this.buildUnitTooltip(unit);

      this.selector.appendChild(option);
    });
  }

  /**
   * Select a specific compilation unit
   */
  async selectUnit(unitName) {
    if (!unitName || unitName === this.currentUnit) {
      return;
    }

    const unit = this.units.find(u => u.name === unitName);
    if (!unit) {
      console.error(`Unit not found: ${unitName}`);
      return;
    }

    try {
      // Update current unit
      const previousUnit = this.currentUnit;
      this.currentUnit = unitName;

      // Update selector
      if (this.selector) {
        this.selector.value = unitName;
      }

      // Update metadata
      const metadata = this.metadata.get(unitName);
      if (metadata) {
        metadata.lastSelected = new Date().toISOString();
      }

      // Call the change callback
      if (this.onUnitChangeCallback) {
        await this.onUnitChangeCallback(unitName, previousUnit);
      }

      // Update recent units tracking
      this.updateRecentUnits(unitName);

      console.log(`Selected compilation unit: ${unitName}`);

    } catch (error) {
      console.error(`Failed to select unit ${unitName}:`, error);

      // Revert selection on error
      this.currentUnit = this.currentUnit; // Keep previous selection
      if (this.selector) {
        this.selector.value = this.currentUnit || '';
      }

      throw error;
    }
  }

  /**
   * Get the currently selected unit
   */
  getCurrentUnit() { return this.currentUnit; }

  /**
   * Get information about the current unit
   */
  getCurrentUnitInfo() {
    if (!this.currentUnit)
      return null;

    const unit = this.units.find(u => u.name === this.currentUnit);
    const metadata = this.metadata.get(this.currentUnit);

    return {...unit, metadata};
  }

  /**
   * Get all available units
   */
  getAllUnits() {
    return this.units.map(
        unit => ({...unit, metadata : this.metadata.get(unit.name)}));
  }

  /**
   * Check if a unit is considered "recent"
   */
  isRecentUnit(unit) {
    // We are just considering the first 3 units as recent
    const index = this.units.findIndex(u => u.name === unit.name);
    return index < 3;
  }

  /**
   * Format timestamp from YYYYMMDD_HHMMSS format
   */
  formatTimestamp(timestamp) {
    if (!timestamp || timestamp.length !== 15) {
      return timestamp; // Return as-is if not in expected format
    }

    try {
      const year = timestamp.substring(0, 4);
      const month = timestamp.substring(4, 6);
      const day = timestamp.substring(6, 8);
      const hour = timestamp.substring(9, 11);
      const minute = timestamp.substring(11, 13);
      const second = timestamp.substring(13, 15);

      const date = new Date(year, month - 1, day, hour, minute, second);
      return date.toLocaleString();
    } catch (e) {
      return timestamp; // Fallback to original string
    }
  }

  /**
   * Build tooltip text for a unit
   */
  buildUnitTooltip(unit) {
    const parts = [];

    parts.push(`Unit: ${unit.name}`);

    if (unit.total_files) {
      parts.push(`Artifacts: ${unit.total_files}`);
    }

    if (unit.run_timestamp) {
      parts.push(`Run: ${this.formatTimestamp(unit.run_timestamp)}`);
    }

    if (unit.available_runs && unit.available_runs.length > 1) {
      parts.push(`Available runs: ${unit.available_runs.length}`);
    }

    if (unit.artifact_types && unit.artifact_types.length > 0) {
      parts.push(`Types: ${unit.artifact_types.join(', ')}`);
    }

    const metadata = this.metadata.get(unit.name);
    if (metadata && metadata.lastSelected) {
      const lastSelected = new Date(metadata.lastSelected);
      parts.push(`Last selected: ${lastSelected.toLocaleString()}`);
    }

    return parts.join('\n');
  }

  /**
   * Update recent units tracking
   */
  updateRecentUnits(unitName) {
    // Move selected unit to the front of recent units
    const unitIndex = this.units.findIndex(u => u.name === unitName);
    if (unitIndex > 0) {
      const unit = this.units.splice(unitIndex, 1)[0];
      this.units.unshift(unit);

      // Update the selector to reflect new order
      this.updateSelector();
    }

    // Store in localStorage for persistence
    try {
      const recentUnits =
          JSON.parse(localStorage.getItem('llvm_advisor_recent_units') || '[]');

      // Remove if already exists
      const filteredRecent = recentUnits.filter(name => name !== unitName);

      // Add to front
      filteredRecent.unshift(unitName);

      // Keep only last 5 recent units
      const updatedRecent = filteredRecent.slice(0, 5);

      localStorage.setItem('llvm_advisor_recent_units',
                           JSON.stringify(updatedRecent));
    } catch (error) {
      console.warn('Failed to update recent units in localStorage:', error);
    }
  }

  /**
   * Get recent units from localStorage
   */
  getRecentUnits() {
    try {
      return JSON.parse(localStorage.getItem('llvm_advisor_recent_units') ||
                        '[]');
    } catch (error) {
      console.warn('Failed to get recent units from localStorage:', error);
      return [];
    }
  }

  /**
   * Auto-select the most appropriate unit
   */
  autoSelectUnit() {
    if (this.units.length === 0)
      return null;

    // Try to select from recent units first
    const recentUnits = this.getRecentUnits();
    for (const recentUnit of recentUnits) {
      if (this.units.some(u => u.name === recentUnit)) {
        this.selectUnit(recentUnit);
        return recentUnit;
      }
    }

    // Otherwise, select the first unit (most recent by default)
    const firstUnit = this.units[0];
    if (firstUnit) {
      this.selectUnit(firstUnit.name);
      return firstUnit.name;
    }

    return null;
  }

  /**
   * Refresh units list from API
   */
  async refreshUnits() {
    const success = await this.loadUnitMetadata();
    if (success) {
      this.updateSelector();

      // If current unit no longer exists, auto-select a new one
      if (this.currentUnit &&
          !this.units.some(u => u.name === this.currentUnit)) {
        this.currentUnit = null;
        this.autoSelectUnit();
      }
    }
    return success;
  }

  /**
   * Get unit statistics
   */
  getUnitStats() {
    return {
      totalUnits : this.units.length,
      currentUnit : this.currentUnit,
      recentUnits : this.getRecentUnits(),
      unitsByFileCount :
          this.units.filter(u => u.total_files)
              .sort((a, b) => b.total_files - a.total_files)
              .slice(0, 5)
              .map(u => ({name : u.name, artifacts : u.total_files}))
    };
  }
}
