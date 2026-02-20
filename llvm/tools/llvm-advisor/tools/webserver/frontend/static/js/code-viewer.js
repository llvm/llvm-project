// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Code Viewer Component
 * Read-only code editor with Prism.js syntax highlighting and selection
 */

import {Utils} from './utils.js';

export class CodeViewer {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      language : 'text',
      showLineNumbers : true,
      readOnly : true,
      ...options
    };

    this.content = '';
    this.inlineData = null;
    this.inlineDataVisible = {diagnostics : false, remarks : false};

    // References to DOM elements for scroll synchronization
    this.lineNumbersDiv = null;
    this.codeContentDiv = null;
  }

  render(content, language = null, inlineData = null) {
    if (!this.container)
      return;

    this.content = content || '';
    this.language = language || this.options.language;
    this.inlineData = inlineData;

    console.log('CodeViewer render:', {
      hasContent : !!this.content,
      language : this.language,
      hasInlineData : !!this.inlineData,
      contentLength : this.content.length
    });

    if (!this.content) {
      this.renderEmpty();
      return;
    }

    this.renderEditor();
  }

  renderEditor() {
    const lines = this.content.split('\n');
    const maxLineDigits = lines.length.toString().length;

    // Check if file is too large and truncate if necessary
    const MAX_LINES = 5000; // Reduced limit to prevent browser freeze
    const isLargeFile = lines.length > MAX_LINES;
    const displayLines = isLargeFile ? lines.slice(0, MAX_LINES) : lines;

    // Create the editor with proper structure for text selection
    const editorDiv = document.createElement('div');
    editorDiv.className =
        'code-editor h-full bg-white border border-gray-200 rounded-lg overflow-hidden';

    // Add warning for large files
    if (isLargeFile) {
      const warningDiv = document.createElement('div');
      warningDiv.className =
          'bg-yellow-50 border-b border-yellow-200 px-4 py-2 text-sm text-yellow-800';
      warningDiv.innerHTML = `⚠️ Large file detected. Showing first ${
          MAX_LINES} lines of ${lines.length} total lines.`;
      editorDiv.appendChild(warningDiv);
    }

    const flexContainer = document.createElement('div');
    flexContainer.className = 'flex h-full';

    // Line numbers column
    if (this.options.showLineNumbers) {
      const lineNumbersDiv = document.createElement('div');
      lineNumbersDiv.className =
          'line-numbers bg-gray-50 border-r border-gray-200 px-3 py-2 select-none flex-shrink-0 text-right font-mono text-sm text-gray-500';
      lineNumbersDiv.style.minWidth = `${maxLineDigits * 10 + 24}px`;
      lineNumbersDiv.style.cssText +=
          'font-size: 14px; line-height: 1.5; padding-top: 8px;';

      // Create line number entries that will match code lines
      displayLines.forEach((_, index) => {
        const lineNumber = index + 1;
        const lineInlineData = this.getInlineDataForLine(lineNumber);

        const lineNumWrapper = document.createElement('div');
        lineNumWrapper.className = 'line-number-wrapper';

        const lineNumDiv = document.createElement('div');
        lineNumDiv.className = 'line-number-content';
        lineNumDiv.style.cssText =
            'min-height: 21px; line-height: 1.5; padding-top: 0; padding-bottom: 0;';
        lineNumDiv.textContent = lineNumber.toString();

        lineNumWrapper.appendChild(lineNumDiv);

        // Add spacer for inline data if present and visible
        if (this.shouldShowInlineData(lineInlineData)) {
          const spacerDiv = document.createElement('div');
          spacerDiv.className = 'line-number-spacer';
          // This will be dynamically sized to match the inline data height
          lineNumWrapper.appendChild(spacerDiv);
        }

        lineNumbersDiv.appendChild(lineNumWrapper);
      });

      flexContainer.appendChild(lineNumbersDiv);

      // Store reference for later scroll synchronization
      this.lineNumbersDiv = lineNumbersDiv;
    }

    // Create code content using Prism.js
    const codeContentDiv = document.createElement('div');
    codeContentDiv.className = 'code-content flex-1 overflow-auto';

    // Create container for line-by-line rendering
    const linesContainer = document.createElement('div');
    linesContainer.className = 'font-mono text-sm';
    linesContainer.style.cssText =
        'padding: 8px; font-size: 14px; line-height: 1.5;';

    // Render each line individually with potential inline data
    displayLines.forEach((line, index) => {
      const lineNumber = index + 1;
      const lineInlineData = this.getInlineDataForLine(lineNumber);

      // Create line wrapper
      const lineWrapper = document.createElement('div');
      lineWrapper.className = 'code-line-wrapper';

      // Create the actual code line
      const codeLine = document.createElement('div');
      codeLine.className = 'code-line';
      codeLine.style.cssText = 'min-height: 21px; line-height: 1.5;';

      // Apply syntax highlighting to individual line
      const tempPre = document.createElement('pre');
      tempPre.className = 'language-' + this.getPrismLanguage(this.language);
      tempPre.style.cssText =
          'margin: 0; padding: 0; background: transparent; display: inline;';

      const tempCode = document.createElement('code');
      tempCode.className = 'language-' + this.getPrismLanguage(this.language);
      tempCode.textContent = line;

      tempPre.appendChild(tempCode);

      // Apply Prism highlighting to this line
      if (window.Prism) {
        try {
          window.Prism.highlightElement(tempCode);
        } catch (e) {
          // Fallback: just display the text
          tempCode.textContent = line;
        }
      }

      codeLine.appendChild(tempPre);
      lineWrapper.appendChild(codeLine);

      // Add inline diagnostics/remarks if visible and present
      if (this.shouldShowInlineData(lineInlineData)) {
        const inlineDataContainer = document.createElement('div');
        inlineDataContainer.className = 'inline-data-container';
        inlineDataContainer.innerHTML =
            this.renderInlineDataForLine(lineInlineData);
        lineWrapper.appendChild(inlineDataContainer);
      }

      linesContainer.appendChild(lineWrapper);
    });

    codeContentDiv.appendChild(linesContainer);

    // Synchronize line number spacer heights with inline data heights
    if (this.options.showLineNumbers) {
      this.synchronizeLineHeights();
    }

    // Set up scroll synchronization between line numbers and code content
    if (this.options.showLineNumbers && this.lineNumbersDiv) {
      codeContentDiv.addEventListener('scroll', () => {
        if (this.lineNumbersDiv) {
          this.lineNumbersDiv.scrollTop = codeContentDiv.scrollTop;
        }
      });
    }

    flexContainer.appendChild(codeContentDiv);
    editorDiv.appendChild(flexContainer);

    // Store reference to code content for external access
    this.codeContentDiv = codeContentDiv;

    // Replace container content
    this.container.innerHTML = '';
    this.container.appendChild(editorDiv);
  }

  renderEmpty() {
    this.container.innerHTML = `
            <div class="h-full flex items-center justify-center bg-gray-50 text-gray-500 border border-gray-200 rounded-lg">
                <div class="text-center">
                    <svg class="mx-auto h-12 w-12 mb-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                    </svg>
                    <p class="text-sm">No code to display</p>
                </div>
            </div>
        `;
  }

  toggleInlineData(type, visible) {
    console.log(`toggleInlineData: ${type} = ${visible}`);
    this.inlineDataVisible[type] = visible;

    // Re-render to show/hide inline data
    if (this.content) {
      console.log('Re-rendering editor after toggle');
      this.renderEditor();
    }
  }

  updateContent(content, language = null, inlineData = null) {
    this.render(content, language, inlineData);
  }

  setLanguage(language) {
    this.language = language;
    if (this.content) {
      this.renderEditor();
    }
  }

  clear() {
    this.content = '';
    this.inlineData = null;
    this.renderEmpty();
  }

  getPrismLanguage(language) {
    /** Map our language identifiers to Prism.js language classes */
    const languageMap = {
      'c' : 'c',
      'cpp' : 'cpp',
      'assembly' : 'nasm', // Use NASM for assembly
      'llvm-ir' : 'llvm',  // Prism has LLVM support
      'json' : 'json',
      'python' : 'python',
      'rust' : 'rust',
      // Probably this filetypes would be never used
      // But we never know hehe
      'javascript' : 'javascript',
      'go' : 'go',
      'java' : 'java'
    };
    return languageMap[language] || 'plaintext';
  }

  shouldShowInlineData(lineInlineData) {
    const hasDiagnostics = this.inlineDataVisible.diagnostics &&
                           lineInlineData.diagnostics.length > 0;
    const hasRemarks =
        this.inlineDataVisible.remarks && lineInlineData.remarks.length > 0;
    return hasDiagnostics || hasRemarks;
  }

  getInlineDataForLine(lineNumber) {
    if (!this.inlineData) {
      return {diagnostics : [], remarks : []};
    }

    // Handle the structured inline data format from the API
    let diagnostics = [];
    let remarks = [];

    // Check if diagnostics is an array of objects with line property
    if (Array.isArray(this.inlineData.diagnostics)) {
      diagnostics =
          this.inlineData.diagnostics.filter(d => d.line === lineNumber);
    }

    // Check if remarks is an array of objects with line property
    if (Array.isArray(this.inlineData.remarks)) {
      remarks = this.inlineData.remarks.filter(r => r.line === lineNumber);
    }

    return {diagnostics, remarks};
  }

  renderInlineDataForLine(lineInlineData) {
    let html = '';

    // Show diagnostics if enabled
    if (this.inlineDataVisible.diagnostics &&
        lineInlineData.diagnostics.length > 0) {
      lineInlineData.diagnostics.forEach(diagnostic => {
        const levelClass = this.getDiagnosticLevelClass(diagnostic.level);
        const icon = this.getDiagnosticIcon(diagnostic.level);

        html += `
                    <div class="inline-diagnostic ${
            levelClass} ml-8 mr-2 mt-1 mb-1 p-2 rounded text-xs border-l-4 transition-all duration-200">
                        <div class="flex items-start space-x-2">
                            <span class="flex-shrink-0">${icon}</span>
                            <div class="flex-1">
                                <div class="font-medium">${
            this.escapeHtml(diagnostic.message)}</div>
                                ${
            diagnostic.column ? `<div class="text-xs mt-1 opacity-75">Column ${
                                    diagnostic.column}</div>`
                              : ''}
                            </div>
                        </div>
                    </div>
                `;
      });
    }

    // Show remarks if enabled
    if (this.inlineDataVisible.remarks && lineInlineData.remarks.length > 0) {
      lineInlineData.remarks.forEach(remark => {
        html += `
                    <div class="inline-remark bg-blue-50 border-l-4 border-blue-400 ml-8 mr-2 mt-1 mb-1 p-2 rounded text-xs transition-all duration-200">
                        <div class="flex items-start space-x-2">
                            <span class="flex-shrink-0 text-blue-600">💡</span>
                            <div class="flex-1">
                                <div class="font-medium text-blue-800">${
            this.escapeHtml(remark.message)}</div>
                                ${
            remark.pass ? `<div class="text-blue-700 text-xs mt-1 italic">[${
                              this.escapeHtml(remark.pass)}]</div>`
                        : ''}
                                ${
            remark.column ? `<div class="text-xs mt-1 opacity-75">Column ${
                                remark.column}</div>`
                          : ''}
                            </div>
                        </div>
                    </div>
                `;
      });
    }

    return html;
  }

  getDiagnosticLevelClass(level) {
    const levelMap = {
      'error' : 'bg-red-50 border-red-400 text-red-800',
      'warning' : 'bg-yellow-50 border-yellow-400 text-yellow-800',
      'note' : 'bg-blue-50 border-blue-400 text-blue-800',
      'info' : 'bg-gray-50 border-gray-400 text-gray-700'
    };
    return levelMap[level] || levelMap['info'];
  }

  getDiagnosticIcon(level) {
    const iconMap =
        {'error' : '❌', 'warning' : '⚠️', 'note' : 'ℹ️', 'info' : '💬'};
    return iconMap[level] || iconMap['info'];
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  synchronizeLineHeights() {
    /**
     * Synchronize line number spacer heights with inline data container
     * heights
     */
    if (!this.lineNumbersDiv)
      return;

    // Use setTimeout to ensure DOM has been rendered
    setTimeout(() => {
      const lineWrappers =
          this.container.querySelectorAll('.code-line-wrapper');
      const lineNumberWrappers =
          this.lineNumbersDiv.querySelectorAll('.line-number-wrapper');

      lineWrappers.forEach((codeLineWrapper, index) => {
        const lineNumberWrapper = lineNumberWrappers[index];
        if (!lineNumberWrapper)
          return;

        const inlineDataContainer =
            codeLineWrapper.querySelector('.inline-data-container');
        const lineNumberSpacer =
            lineNumberWrapper.querySelector('.line-number-spacer');

        if (inlineDataContainer && lineNumberSpacer) {
          // Match the height of the inline data container
          const inlineDataHeight = inlineDataContainer.offsetHeight;
          lineNumberSpacer.style.height = `${inlineDataHeight}px`;
        } else if (lineNumberSpacer) {
          // If no inline data, remove the spacer height
          lineNumberSpacer.style.height = '0px';
        }
      });
    }, 0);
  }
}
