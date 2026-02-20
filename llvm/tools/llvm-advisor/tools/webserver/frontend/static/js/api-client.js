// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * API Client
 * Handles all communication with the LLVM Advisor API backend
 */

import {Utils} from './utils.js';

export class ApiClient {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
    this.cache = new Map();
    this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
  }

  /**
   * Generic HTTP request method with error handling and caching
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}/api/${endpoint}`;
    const cacheKey = `${url}${JSON.stringify(options)}`;

    // Check cache first for GET requests
    if (!options.method || options.method === 'GET') {
      const cached = this.cache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
        return cached.data;
      }
    }

    try {
      const response = await fetch(url, {
        method : 'GET',
        headers : {'Content-Type' : 'application/json', ...options.headers},
        ...options
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // Cache successful GET responses
      if (!options.method || options.method === 'GET') {
        this.cache.set(cacheKey, {data, timestamp : Date.now()});
      }

      return data;

    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      return {
        success : false,
        error : error.message,
        status : error.status || 500
      };
    }
  }

  /**
   * Clear all cached responses
   */
  clearCache() { this.cache.clear(); }

  // ============================================
  // Core API Endpoints
  // ============================================

  /**
   * Get system health status
   */
  async getHealth() { return await this.request('health'); }

  /**
   * Get all compilation units
   */
  async getUnits() { return await this.request('units'); }

  /**
   * Get detailed information for a specific unit
   */
  async getUnitDetail(unitName) {
    return await this.request(`units/${encodeURIComponent(unitName)}`);
  }

  /**
   * Get overall summary statistics
   */
  async getSummary() { return await this.request('summary'); }

  /**
   * Get available artifact types
   */
  async getArtifactTypes() { return await this.request('artifacts'); }

  /**
   * Get aggregated data for a specific file type
   */
  async getArtifactData(fileType) {
    return await this.request(`artifacts/${encodeURIComponent(fileType)}`);
  }

  /**
   * Get build dependencies data
   */
  async getBuildDependencies() {
    return await this.request('artifacts/dependencies');
  }

  /**
   * Get specific file content
   */
  async getFileContent(unitName, fileType, fileName, full = false) {
    const params = full ? '?full=true' : '';
    return await this.request(
        `file/${encodeURIComponent(unitName)}/${encodeURIComponent(fileType)}/${
            encodeURIComponent(fileName)}${params}`);
  }

  // ============================================
  // Specialized Endpoints - Remarks
  // ============================================

  /**
   * Get optimization remarks overview
   */
  async getRemarksOverview() { return await this.request('remarks/overview'); }

  /**
   * Get remarks analysis by optimization passes
   */
  async getRemarksPasses() { return await this.request('remarks/passes'); }

  /**
   * Get remarks analysis by functions
   */
  async getRemarksFunctions() {
    return await this.request('remarks/functions');
  }

  /**
   * Get optimization hotspots
   */
  async getRemarksHotspots() { return await this.request('remarks/hotspots'); }

  // ============================================
  // Specialized Endpoints - Diagnostics
  // ============================================

  /**
   * Get diagnostics overview
   */
  async getDiagnosticsOverview() {
    return await this.request('diagnostics/overview');
  }

  /**
   * Get diagnostics by level (error, warning, note)
   */
  async getDiagnosticsByLevel() {
    return await this.request('diagnostics/by-level');
  }

  /**
   * Get diagnostics by files
   */
  async getDiagnosticsFiles() {
    return await this.request('diagnostics/files');
  }

  /**
   * Get diagnostic patterns
   */
  async getDiagnosticsPatterns() {
    return await this.request('diagnostics/patterns');
  }

  // ============================================
  // Specialized Endpoints - Compilation Analysis
  // ============================================

  /**
   * Get ftime report data for compilation timing
   */
  async getFTimeReport() {
    return await this.request('artifacts/ftime-report');
  }

  /**
   * Get version info data (clang version, target, etc.)
   */
  async getVersionInfo() {
    return await this.request('artifacts/version-info');
  }

  /**
   * Get compilation phases bindings (from -ccc-print-bindings)
   */
  async getCompilationPhasesBindings() {
    return await this.request('compilation-phases/bindings');
  }

  // ============================================
  // Specialized Endpoints - Time Trace
  // ============================================

  /**
   * Get time trace overview
   */
  async getTimeTraceOverview() {
    return await this.request('time-trace/overview');
  }

  /**
   * Get time trace timeline (with optional limit)
   */
  async getTimeTraceTimeline(limit = 1000) {
    return await this.request(`time-trace/timeline?limit=${limit}`);
  }

  /**
   * Get time trace hotspots
   */
  async getTimeTraceHotspots() {
    return await this.request('time-trace/hotspots');
  }

  /**
   * Get time trace categories analysis
   */
  async getTimeTraceCategories() {
    return await this.request('time-trace/categories');
  }

  /**
   * Get parallelism analysis
   */
  async getTimeTraceParallelism() {
    return await this.request('time-trace/parallelism');
  }

  // ============================================
  // Specialized Endpoints - Binary Size
  // ============================================

  /**
   * Get binary size overview
   */
  async getBinarySizeOverview() {
    return await this.request('binary-size/overview');
  }

  /**
   * Get binary sections analysis
   */
  async getBinarySizeSections() {
    return await this.request('binary-size/sections');
  }

  /**
   * Get binary size optimization opportunities
   */
  async getBinarySizeOptimization() {
    return await this.request('binary-size/optimization');
  }

  /**
   * Get binary size comparison across units
   */
  async getBinarySizeComparison() {
    return await this.request('binary-size/comparison');
  }

  // ============================================
  // Specialized Endpoints - Runtime Trace
  // ============================================

  /**
   * Get runtime trace overview
   */
  async getRuntimeTraceOverview() {
    return await this.request('runtime-trace/overview');
  }

  /**
   * Get runtime trace timeline
   */
  async getRuntimeTraceTimeline(limit = 1000) {
    return await this.request(`runtime-trace/timeline?limit=${limit}`);
  }

  /**
   * Get runtime trace hotspots
   */
  async getRuntimeTraceHotspots() {
    return await this.request('runtime-trace/hotspots');
  }

  /**
   * Get runtime trace categories
   */
  async getRuntimeTraceCategories() {
    return await this.request('runtime-trace/categories');
  }

  /**
   * Get runtime parallelism analysis
   */
  async getRuntimeTraceParallelism() {
    return await this.request('runtime-trace/parallelism');
  }

  // ============================================
  // Specialized Endpoints - Code Explorer
  // ============================================

  /**
   * Get list of available source files for a specific unit
   */
  async getSourceFiles(unitName = null) {
    const params = unitName ? `?unit=${encodeURIComponent(unitName)}` : '';
    return await this.request(`explorer/files${params}`);
  }

  /**
   * Get source code for a specific file
   */
  async getSourceCode(filePath) {
    return await this.request(
        `explorer/source/${encodeURIComponent(filePath)}`);
  }

  /**
   * Get assembly output for a specific file
   */
  async getAssembly(filePath) {
    return await this.request(
        `explorer/assembly/${encodeURIComponent(filePath)}`);
  }

  /**
   * Get LLVM IR for a specific file
   */
  async getLLVMIR(filePath) {
    return await this.request(`explorer/ir/${encodeURIComponent(filePath)}`);
  }

  /**
   * Get optimized LLVM IR for a specific file
   */
  async getOptimizedIR(filePath) {
    return await this.request(
        `explorer/optimized-ir/${encodeURIComponent(filePath)}`);
  }

  /**
   * Get object code for a specific file
   */
  async getObjectCode(filePath) {
    return await this.request(
        `explorer/object/${encodeURIComponent(filePath)}`);
  }

  /**
   * Get AST JSON for a specific file
   */
  async getASTJSON(filePath) {
    return await this.request(
        `explorer/ast-json/${encodeURIComponent(filePath)}`);
  }

  /**
   * Get preprocessed source for a specific file
   */
  async getPreprocessed(filePath) {
    return await this.request(
        `explorer/preprocessed/${encodeURIComponent(filePath)}`);
  }

  /**
   * Get macro expansion for a specific file
   */
  async getMacroExpansion(filePath) {
    return await this.request(
        `explorer/macro-expansion/${encodeURIComponent(filePath)}`);
  }

  // ============================================
  // Utility Methods
  // ============================================

  /**
   * Check if the API is available
   */
  async isApiAvailable() {
    try {
      const health = await this.getHealth();
      return health.success && health.data.status === 'healthy';
    } catch (error) {
      return false;
    }
  }

  /**
   * Get cache statistics
   */
  getCacheStats() {
    return {size : this.cache.size, keys : Array.from(this.cache.keys())};
  }

  /**
   * Batch multiple API requests
   */
  async batchRequests(requests) {
    const promises =
        requests.map(req => typeof req === 'string'
                                ? this.request(req)
                                : this.request(req.endpoint, req.options));

    const results = await Promise.allSettled(promises);

    return results.map(
        (result, index) => ({
          request : requests[index],
          success : result.status === 'fulfilled' && result.value.success,
          data : result.status === 'fulfilled' ? result.value.data : null,
          error : result.status === 'rejected'
                      ? result.reason.message
                      : (result.value.success ? null : result.value.error)
        }));
  }
}
