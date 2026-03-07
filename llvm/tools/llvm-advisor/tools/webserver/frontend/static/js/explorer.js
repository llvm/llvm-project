// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Explorer Module
 */

import {CodeViewer} from './code-viewer.js';
import {Utils} from './utils.js';

export class Explorer {
  constructor(apiClient, app) {
    this.apiClient = apiClient;
    this.app = app;
    this.currentFile = null;
    this.currentViewType = 'assembly';
    this.sourceContent = null;
    this.outputContent = null;
    this.availableFiles = [];
    this.isLoading = false;

    this.initializeElements();
  }

  initializeElements() {
    this.fileSelector = document.getElementById('file-selector');
    this.viewTypeSelector = document.getElementById('view-type-selector');
    this.sourceContainer = document.getElementById('source-code-container');
    this.outputContainer = document.getElementById('output-container');
    this.rightPanelTitle = document.getElementById('right-panel-title');
    this.copyBtn = document.getElementById('copy-output-btn');
    this.downloadBtn = document.getElementById('download-output-btn');

    // Inline data toggle buttons
    this.toggleDiagnosticsBtn =
        document.getElementById('toggle-diagnostics-btn');
    this.toggleRemarksBtn = document.getElementById('toggle-remarks-btn');

    // Initialize inline data state
    this.inlineDataVisible = {diagnostics : false, remarks : false};
    this.inlineData = null;

    // Initialize code viewers
    this.sourceViewer = null;
    this.outputViewer = null;
  }

  init() {
    this.setupEventListeners();
    console.log('🔍 Explorer initialized');
  }

  setupEventListeners() {
    // File selection change
    this.fileSelector?.addEventListener('change', (event) => {
      const selectedFile = event.target.value;
      if (selectedFile && selectedFile !== this.currentFile) {
        const fileData =
            this.availableFiles.find(f => (f.path || f.name) === selectedFile);
        this.updateViewTypeSelector(fileData);
        this.loadFile(selectedFile);
      }
    });

    // View type selection change
    this.viewTypeSelector?.addEventListener('change', async (event) => {
      const newViewType = event.target.value;
      if (newViewType !== this.currentViewType) {
        this.currentViewType = newViewType;
        this.updateRightPanelTitle();
        if (this.currentFile) {
          console.log(`Loading new view type: ${newViewType} for file: ${
              this.currentFile}`);
          const outputResponse =
              await this.loadOutput(this.currentFile, newViewType);
          if (outputResponse) {
            this.displayOutput(outputResponse);
          }
          this.updateActionButtons();
        }
      }
    });

    // Copy and download buttons
    this.copyBtn?.addEventListener('click', () => this.copyToClipboard());
    this.downloadBtn?.addEventListener('click', () => this.downloadOutput());

    // Inline data toggle buttons
    this.toggleDiagnosticsBtn?.addEventListener('click', (e) => {
      console.log('Diagnostics button clicked');
      e.preventDefault();
      this.toggleInlineData('diagnostics');
    });

    this.toggleRemarksBtn?.addEventListener('click', (e) => {
      console.log('Remarks button clicked');
      e.preventDefault();
      this.toggleInlineData('remarks');
    });
  }

  async loadAvailableFiles() {
    try {
      // Get current unit name
      const currentUnitName = this.app.currentUnit ||
                              document.getElementById('unit-selector')?.value;

      if (!currentUnitName) {
        console.log('No unit selected, cannot load files');
        this.availableFiles = [];
        this.updateFileSelector();
        return;
      }

      console.log('Loading files for unit:', currentUnitName);
      const response = await this.apiClient.getSourceFiles(currentUnitName);
      if (response.success && response.data) {
        this.availableFiles = response.data.files || [];
        console.log('Loaded files:', this.availableFiles);
        this.updateFileSelector();
      } else {
        console.warn('Failed to load source files:', response.error);
        this.showError('Failed to load available files');
      }
    } catch (error) {
      console.error('Error loading available files:', error);
      this.showError('Error loading available files');
    }
  }

  updateFileSelector() {
    if (!this.fileSelector)
      return;

    this.fileSelector.innerHTML = '<option value="">Select a file...</option>';

    this.availableFiles.forEach(file => {
      const option = document.createElement('option');
      option.value = file.path || file.name;
      option.textContent = file.display_name || file.name;
      this.fileSelector.appendChild(option);
    });

    // Auto-select first file if available
    if (this.availableFiles.length > 0) {
      const firstFile = this.availableFiles[0];
      this.fileSelector.value = firstFile.path || firstFile.name;
      this.updateViewTypeSelector(firstFile);
      this.loadFile(firstFile.path || firstFile.name);
    }
  }

  updateViewTypeSelector(selectedFile = null) {
    if (!this.viewTypeSelector)
      return;

    if (!selectedFile) {
      const selectedPath = this.fileSelector?.value;
      selectedFile =
          this.availableFiles.find(f => (f.path || f.name) === selectedPath);
    }

    this.viewTypeSelector.innerHTML = '';

    const availableArtifacts = selectedFile?.available_artifacts || [];

    const artifactDisplayNames = {
      'assembly' : 'Assembly',
      'ir' : 'LLVM IR',
      'ast-json' : 'AST JSON',
      'object' : 'Object Code',
      'preprocessed' : 'Preprocessed',
      'macro-expansion' : 'Macro Expansion'
    };

    availableArtifacts.forEach(artifact => {
      const option = document.createElement('option');
      option.value = artifact;
      option.textContent =
          artifactDisplayNames[artifact] || Utils.capitalize(artifact);
      this.viewTypeSelector.appendChild(option);
    });

    // Set default selection (prefer assembly, then ir, then other code
    // artifacts)
    if (availableArtifacts.length > 0) {
      const preferredOrder = [
        'assembly', 'ir', 'ast-json', 'object', 'preprocessed',
        'macro-expansion'
      ];
      let defaultType = availableArtifacts[0];

      for (const preferred of preferredOrder) {
        if (availableArtifacts.includes(preferred)) {
          defaultType = preferred;
          break;
        }
      }

      this.viewTypeSelector.value = defaultType;
      this.currentViewType = defaultType;
      this.updateRightPanelTitle();
    }
  }

  async loadFile(filePath) {
    if (this.isLoading)
      return;

    this.isLoading = true;
    this.currentFile = filePath;
    this.showLoadingStates();

    try {
      const [sourceResponse, outputResponse] = await Promise.all([
        this.loadSourceCode(filePath),
        this.loadOutput(filePath, this.currentViewType)
      ]);

      if (sourceResponse) {
        this.displaySourceCode(sourceResponse);
      }

      if (outputResponse) {
        this.displayOutput(outputResponse);
      }

      this.updateActionButtons();

    } catch (error) {
      console.error('Error loading file:', error);
      this.showError('Failed to load file content');
    } finally {
      this.isLoading = false;
    }
  }

  async loadSourceCode(filePath) {
    try {
      const response = await this.apiClient.getSourceCode(filePath);
      if (response.success) {
        this.sourceContent = response.data;
        console.log('Loaded source code with inline data:',
                    response.data.inline_data);
        return response.data;
      } else {
        console.warn('Failed to load source code:', response.error);
        return null;
      }
    } catch (error) {
      console.error('Error loading source code:', error);
      return null;
    }
  }

  async loadOutput(filePath, viewType) {
    try {
      const endpoint = `explorer/${encodeURIComponent(viewType)}/${
          encodeURIComponent(filePath)}`;
      const response = await this.apiClient.request(endpoint);
      console.log(`Response for ${viewType}:`, response);

      if (response?.success && response.data) {
        const content =
            response.data.content || response.data.data || response.data || '';

        // Debug: Check if content contains HTML tags
        if (typeof content === 'string' && content.includes('<span')) {
          console.warn(`Content appears to contain HTML tags: ${
              content.substring(0, 200)}...`);
        }

        // Ensure content is a string
        let finalContent = '';
        if (typeof content === 'string') {
          finalContent = content;
        } else if (typeof content === 'object') {
          finalContent = JSON.stringify(content, null, 2);
        } else {
          finalContent = String(content);
        }

        console.log(`Final content for ${viewType}: ${
            finalContent.length} chars, starts with: ${
            finalContent.substring(0, 100)}`);

        this.outputContent = {content : finalContent};
        return {content : finalContent};
      }
      console.warn(`Failed to load ${viewType}:`,
                   response?.error || 'No response data');
      return null;
    } catch (error) {
      console.error(`Error loading ${viewType}:`, error);
      return null;
    }
  }
  displaySourceCode(content) {
    if (!this.sourceContainer)
      return;

    if (!content || !content.source) {
      this.showSourceError('No source code available');
      return;
    }

    console.log('displaySourceCode called with:', {
      hasSource : !!content.source,
      language : content.language,
      hasInlineData : !!content.inline_data,
      inlineDataKeys : content.inline_data ? Object.keys(content.inline_data)
                                           : []
    });

    this.sourceContent = content;
    this.inlineData = content.inline_data;

    this.updateInlineDataButtons();

    // Initialize or update the source code viewer
    if (!this.sourceViewer) {
      this.sourceViewer = new CodeViewer(
          this.sourceContainer,
          {language : content.language || 'cpp', showLineNumbers : true});
    }

    this.sourceViewer.render(content.source, content.language || 'cpp',
                             this.inlineData);

    // Update inline data visibility
    this.sourceViewer.toggleInlineData('diagnostics',
                                       this.inlineDataVisible.diagnostics);
    this.sourceViewer.toggleInlineData('remarks',
                                       this.inlineDataVisible.remarks);
  }

  displayOutput(content) {
    if (!this.outputContainer)
      return;

    if (!content || (!content.content && !content.output)) {
      this.showOutputError(`No ${this.currentViewType} available`);
      return;
    }

    const outputText = content.content || content.output || '';

    // Check if content is empty or placeholder
    if (!outputText || outputText.trim() === '' ||
        outputText.trim() === '# No data available') {
      this.showOutputError(
          `No ${this.currentViewType} content found for this file`);
      return;
    }

    // Map current view type to language for syntax highlighting
    const languageMap = {
      'assembly' : 'assembly',
      'ir' : 'llvm-ir',
      'optimized-ir' : 'llvm-ir',
      'ast-json' : 'json',
      'object' : 'assembly',
      'preprocessed' : 'c',
      'macro-expansion' : 'c'
    };

    const language = languageMap[this.currentViewType] || 'text';

    // Initialize or update the output viewer
    if (!this.outputViewer) {
      this.outputViewer = new CodeViewer(
          this.outputContainer, {language : language, showLineNumbers : true});
    }

    this.outputViewer.render(outputText, language);
  }

  showLoadingStates() {
    if (this.sourceContainer) {
      this.sourceContainer.innerHTML =
          '<div class="flex items-center justify-center h-full text-gray-500">Loading source code...</div>';
    }

    if (this.outputContainer) {
      this.outputContainer.innerHTML =
          '<div class="flex items-center justify-center h-full text-gray-500">Loading output...</div>';
    }
  }

  showSourceError(message) {
    if (this.sourceContainer) {
      this.sourceContainer.innerHTML = `
                <div class="flex items-center justify-center h-full text-gray-500">
                    <div class="text-center">
                        <svg class="mx-auto h-8 w-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
                        </svg>
                        <p class="text-sm">${message}</p>
                    </div>
                </div>
            `;
    }
  }

  showOutputError(message) {
    if (this.outputContainer) {
      this.outputContainer.innerHTML = `
                <div class="flex items-center justify-center h-full text-gray-500">
                    <div class="text-center">
                        <svg class="mx-auto h-8 w-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
                        </svg>
                        <p class="text-sm">${message}</p>
                    </div>
                </div>
            `;
    }
  }

  showError(message) {
    console.error('Explorer error:', message);
    this.showSourceError('Failed to load content');
    this.showOutputError('Failed to load content');
  }

  updateRightPanelTitle() {
    if (!this.rightPanelTitle)
      return;

    const titles = {
      'assembly' : 'Assembly',
      'ir' : 'LLVM IR',
      'optimized-ir' : 'Optimized IR',
      'ast-json' : 'AST JSON',
      'object' : 'Object Code'
    };

    this.rightPanelTitle.textContent = titles[this.currentViewType] || 'Output';
  }

  updateActionButtons() {
    const hasContent = this.outputContent && (this.outputContent.content ||
                                              this.outputContent.output);

    if (this.copyBtn) {
      this.copyBtn.disabled = !hasContent;
    }

    if (this.downloadBtn) {
      this.downloadBtn.disabled = !hasContent;
    }
  }

  async copyToClipboard() {
    if (!this.outputContent)
      return;

    const text = this.outputContent.content || this.outputContent.output || '';

    try {
      await navigator.clipboard.writeText(text);

      const originalText = this.copyBtn?.textContent;
      if (this.copyBtn) {
        this.copyBtn.textContent = 'Copied!';
        setTimeout(() => { this.copyBtn.textContent = originalText; }, 2000);
      }
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  }

  downloadOutput() {
    if (!this.outputContent || !this.currentFile)
      return;

    const content =
        this.outputContent.content || this.outputContent.output || '';
    const filename =
        `${this.currentFile.split('/').pop()}.${this.currentViewType}`;

    const blob = new Blob([ content ], {type : 'text/plain'});
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();

    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }

  async refresh() {
    if (this.currentFile) {
      await this.loadFile(this.currentFile);
    } else {
      await this.loadAvailableFiles();
    }
  }

  async onActivate() {
    if (this.availableFiles.length === 0) {
      await this.loadAvailableFiles();
    }
  }

  onDeactivate() {
    // Clean up if needed
  }

  // Inline data methods
  updateInlineDataButtons() {
    let diagnosticsCount = 0;
    let remarksCount = 0;

    // Handle both array and string formats
    if (this.inlineData?.diagnostics) {
      if (Array.isArray(this.inlineData.diagnostics)) {
        diagnosticsCount = this.inlineData.diagnostics.length;
      } else if (typeof this.inlineData.diagnostics === 'string') {
        // Count diagnostic lines in string format
        diagnosticsCount = (this.inlineData.diagnostics.match(
                                /:\d+:\d+:\s+(warning|error|note|info):/g) ||
                            []).length;
      }
    }

    if (this.inlineData?.remarks) {
      if (Array.isArray(this.inlineData.remarks)) {
        remarksCount = this.inlineData.remarks.length;
      } else if (typeof this.inlineData.remarks === 'string') {
        // Count YAML blocks in string format
        remarksCount =
            (this.inlineData.remarks.match(/^---\s+!/gm) || []).length;
      }
    }

    if (this.toggleDiagnosticsBtn) {
      this.toggleDiagnosticsBtn.disabled = diagnosticsCount === 0;
      this.toggleDiagnosticsBtn.textContent =
          diagnosticsCount > 0 ? `Diagnostics (${diagnosticsCount})`
                               : 'Diagnostics';
    }

    if (this.toggleRemarksBtn) {
      this.toggleRemarksBtn.disabled = remarksCount === 0;
      this.toggleRemarksBtn.textContent =
          remarksCount > 0 ? `Remarks (${remarksCount})` : 'Remarks';
    }
  }

  toggleInlineData(type) {
    console.log(`Explorer.toggleInlineData called: ${type}`);
    console.log('Current state:', this.inlineDataVisible[type]);
    console.log('Available inline data:', this.inlineData);

    this.inlineDataVisible[type] = !this.inlineDataVisible[type];

    // Update button appearance
    const button = type === 'diagnostics' ? this.toggleDiagnosticsBtn
                                          : this.toggleRemarksBtn;
    console.log('Button found:', !!button);

    if (button) {
      if (this.inlineDataVisible[type]) {
        button.classList.add('bg-blue-600', 'text-white', 'border-blue-600');
        button.classList.remove('text-gray-600', 'border-gray-300',
                                'hover:bg-gray-200');
      } else {
        button.classList.remove('bg-blue-600', 'text-white', 'border-blue-600');
        button.classList.add('text-gray-600', 'border-gray-300',
                             'hover:bg-gray-200');
      }
    }

    // Update the source viewer inline data visibility
    if (this.sourceViewer) {
      console.log('Calling sourceViewer.toggleInlineData');
      this.sourceViewer.toggleInlineData(type, this.inlineDataVisible[type]);
    } else {
      console.log('No sourceViewer available');
    }
  }
}
