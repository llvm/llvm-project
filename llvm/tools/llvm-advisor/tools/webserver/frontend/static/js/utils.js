// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Utility Functions
 * Common helper functions used throughout the LLVM Advisor dashboard
 */

export class Utils {
  /**
   * Format numbers with proper separators and abbreviations
   */
  static formatNumber(num) {
    if (num === null || num === undefined)
      return '0';

    const number = parseInt(num);
    if (isNaN(number))
      return '0';

    // Use compact notation for large numbers
    if (number >= 1000000) {
      return `${(number / 1000000).toFixed(1)}M`;
    } else if (number >= 1000) {
      return `${(number / 1000).toFixed(1)}K`;
    }

    return number.toLocaleString();
  }

  /**
   * Format file type names for display
   */
  static formatFileType(type) {
    if (!type)
      return 'Unknown';

    const typeMap = {
      'opt_record' : 'Optimization Records',
      'opt_remarks' : 'Optimization Remarks',
      'time_trace' : 'Time Trace',
      'runtime_trace' : 'Runtime Trace',
      'binary_size' : 'Binary Size',
      'compilation_units' : 'Compilation Units',
      'diagnostics' : 'Diagnostics',
      'clang_diagnostics' : 'Clang Diagnostics',
      'coverage_report' : 'Coverage Report',
      'profile_data' : 'Profile Data',
      'ast_dump' : 'AST Dump',
      'ir_code' : 'IR Code',
      'assembly_code' : 'Assembly Code',
      'debug_info' : 'Debug Info',
      'static_analysis' : 'Static Analysis',
      'memory_usage' : 'Memory Usage',
      'compilation_commands' : 'Compilation Commands',
      'build_log' : 'Build Log',
      'link_map' : 'Link Map',
      'symbol_table' : 'Symbol Table'
    };

    return typeMap[type] || this.capitalize(type.replace(/_/g, ' '));
  }

  /**
   * Capitalize the first letter of a string
   */
  static capitalize(str) {
    if (!str)
      return '';
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
  }

  /**
   * Format compilation phase names for display
   */
  static formatPhaseName(name) {
    if (!name)
      return 'Unknown Phase';

    // Common LLVM phase name mappings
    const phaseMap = {
      'frontend' : 'Frontend',
      'backend' : 'Backend',
      'codegen' : 'Code Generation',
      'optimization' : 'Optimization',
      'linking' : 'Linking',
      'parsing' : 'Parsing',
      'semantic' : 'Semantic Analysis',
      'irgen' : 'IR Generation',
      'opt' : 'Optimization',
      'asm' : 'Assembly Generation',
      'obj' : 'Object Generation'
    };

    // Try direct mapping first
    if (phaseMap[name.toLowerCase()]) {
      return phaseMap[name.toLowerCase()];
    }

    // Format by replacing underscores and capitalizing
    return name.replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase())
        .replace(/\bIr\b/g, 'IR')
        .replace(/\bLlvm\b/g, 'LLVM')
        .replace(/\bCpp\b/g, 'C++')
        .replace(/\bAst\b/g, 'AST');
  }

  /**
   * Format time values (milliseconds) for display
   */
  static formatTime(timeMs) {
    if (timeMs === null || timeMs === undefined)
      return '0ms';

    const time = parseFloat(timeMs);
    if (isNaN(time))
      return '0ms';

    if (time < 1000) {
      return `${time.toFixed(0)}ms`;
    } else if (time < 60000) {
      return `${(time / 1000).toFixed(2)}s`;
    } else if (time < 3600000) {
      const minutes = Math.floor(time / 60000);
      const seconds = ((time % 60000) / 1000).toFixed(0);
      return `${minutes}m ${seconds}s`;
    } else {
      const hours = Math.floor(time / 3600000);
      const minutes = Math.floor((time % 3600000) / 60000);
      return `${hours}h ${minutes}m`;
    }
  }

  /**
   * Format byte sizes for display
   */
  static formatBytes(bytes) {
    if (bytes === null || bytes === undefined)
      return '0 B';

    const size = parseInt(bytes);
    if (isNaN(size) || size === 0)
      return '0 B';

    const units = [ 'B', 'KB', 'MB', 'GB', 'TB' ];
    const threshold = 1024;

    if (size < threshold)
      return `${size} B`;

    let unitIndex = 0;
    let value = size;

    while (value >= threshold && unitIndex < units.length - 1) {
      value /= threshold;
      unitIndex++;
    }

    return `${value.toFixed(1)} ${units[unitIndex]}`;
  }

  /**
   * Format binary section names for display
   */
  static formatSectionName(name) {
    if (!name)
      return 'Unknown Section';

    // Common binary section mappings
    const sectionMap = {
      '.text' : 'Code (.text)',
      '.data' : 'Data (.data)',
      '.bss' : 'BSS (.bss)',
      '.rodata' : 'Read-Only Data (.rodata)',
      '.debug' : 'Debug Info (.debug)',
      '.symtab' : 'Symbol Table (.symtab)',
      '.strtab' : 'String Table (.strtab)',
      '.rela' : 'Relocations (.rela)',
      '.dynamic' : 'Dynamic (.dynamic)',
      '.interp' : 'Interpreter (.interp)',
      '.note' : 'Notes (.note)',
      '.comment' : 'Comments (.comment)',
      '.plt' : 'PLT (.plt)',
      '.got' : 'GOT (.got)'
    };

    // Try direct mapping first
    if (sectionMap[name]) {
      return sectionMap[name];
    }

    // If it starts with a dot, assume it's a section name
    if (name.startsWith('.')) {
      return `${this.capitalize(name.slice(1))} (${name})`;
    }

    return this.capitalize(name);
  }

  /**
   * Format percentage values
   */
  static formatPercentage(value, decimals = 1) {
    if (value === null || value === undefined)
      return '0%';

    const num = parseFloat(value);
    if (isNaN(num))
      return '0%';

    return `${num.toFixed(decimals)}%`;
  }

  /**
   * Format diagnostic level names
   */
  static formatDiagnosticLevel(level) {
    const levelMap = {
      'error' : 'Error',
      'warning' : 'Warning',
      'note' : 'Note',
      'info' : 'Info',
      'fatal' : 'Fatal Error',
      'remark' : 'Remark'
    };

    return levelMap[level?.toLowerCase()] ||
           this.capitalize(level || 'unknown');
  }

  /**
   * Truncate text to specified length with ellipsis
   */
  static truncateText(text, maxLength = 50) {
    if (!text)
      return '';
    if (text.length <= maxLength)
      return text;

    return text.substring(0, maxLength - 3) + '...';
  }

  /**
   * Debounce function calls
   */
  static debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  /**
   * Throttle function calls
   */
  static throttle(func, limit) {
    let inThrottle;
    return function executedFunction(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }

  /**
   * Deep clone an object
   */
  static deepClone(obj) {
    if (obj === null || typeof obj !== 'object')
      return obj;
    if (obj instanceof Date)
      return new Date(obj.getTime());
    if (obj instanceof Array)
      return obj.map(item => this.deepClone(item));
    if (typeof obj === 'object') {
      const clonedObj = {};
      Object.keys(obj).forEach(
          key => { clonedObj[key] = this.deepClone(obj[key]); });
      return clonedObj;
    }
  }

  /**
   * Check if two objects are equal (deep comparison)
   */
  static isEqual(obj1, obj2) {
    if (obj1 === obj2)
      return true;
    if (obj1 == null || obj2 == null)
      return false;
    if (typeof obj1 !== typeof obj2)
      return false;

    if (typeof obj1 === 'object') {
      const keys1 = Object.keys(obj1);
      const keys2 = Object.keys(obj2);

      if (keys1.length !== keys2.length)
        return false;

      for (let key of keys1) {
        if (!keys2.includes(key))
          return false;
        if (!this.isEqual(obj1[key], obj2[key]))
          return false;
      }

      return true;
    }

    return obj1 === obj2;
  }

  /**
   * Generate a random color
   */
  static getRandomColor() {
    const colors = [
      '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4',
      '#84cc16', '#f97316', '#ec4899', '#6366f1', '#14b8a6', '#f59e0b'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  }

  /**
   * Get contrast color (black or white) for a given background color
   */
  static getContrastColor(hexColor) {
    // Remove # if present
    hexColor = hexColor.replace('#', '');

    // Convert to RGB
    const r = parseInt(hexColor.substr(0, 2), 16);
    const g = parseInt(hexColor.substr(2, 2), 16);
    const b = parseInt(hexColor.substr(4, 2), 16);

    // Calculate luminance
    const luminance = ((0.299 * r) + (0.587 * g) + (0.114 * b)) / 255;

    return luminance > 0.5 ? '#000000' : '#ffffff';
  }

  /**
   * Format date/time for display
   */
  static formatDateTime(date, options = {}) {
    if (!date)
      return 'Unknown';

    const dateObj = date instanceof Date ? date : new Date(date);

    const defaultOptions = {
      year : 'numeric',
      month : 'short',
      day : 'numeric',
      hour : '2-digit',
      minute : '2-digit'
    };

    return dateObj.toLocaleDateString('en-US', {...defaultOptions, ...options});
  }

  /**
   * Format relative time (e.g., "2 minutes ago")
   */
  static formatRelativeTime(date) {
    if (!date)
      return 'Unknown';

    const now = new Date();
    const dateObj = date instanceof Date ? date : new Date(date);
    const diffMs = now - dateObj;

    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffSecs < 60)
      return 'Just now';
    if (diffMins < 60)
      return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24)
      return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 7)
      return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;

    return this.formatDateTime(dateObj);
  }

  /**
   * Validate email address
   */
  static isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  /**
   * Escape HTML to prevent XSS
   */
  static escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Generate a simple hash from a string
   */
  static hashString(str) {
    let hash = 0;
    if (str.length === 0)
      return hash;

    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }

    return Math.abs(hash);
  }

  /**
   * Check if value is empty (null, undefined, empty string, empty array, etc.)
   */
  static isEmpty(value) {
    if (value == null)
      return true;
    if (typeof value === 'string')
      return value.trim().length === 0;
    if (Array.isArray(value))
      return value.length === 0;
    if (typeof value === 'object')
      return Object.keys(value).length === 0;
    return false;
  }

  /**
   * Sort array of objects by a property
   */
  static sortBy(array, property, ascending = true) {
    return array.sort((a, b) => {
      const aVal = a[property];
      const bVal = b[property];

      if (aVal < bVal)
        return ascending ? -1 : 1;
      if (aVal > bVal)
        return ascending ? 1 : -1;
      return 0;
    });
  }

  /**
   * Group array of objects by a property
   */
  static groupBy(array, property) {
    return array.reduce((groups, item) => {
      const key = item[property];
      if (!groups[key]) {
        groups[key] = [];
      }
      groups[key].push(item);
      return groups;
    }, {});
  }
}
