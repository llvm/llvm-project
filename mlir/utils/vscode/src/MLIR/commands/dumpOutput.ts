import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import {spawn} from 'child_process';

import {Command} from '../../command';
import {MLIRContext} from '../../mlirContext';
import * as config from '../../config';

/** One entry in `mlir.litSuites`. */
interface LitSuite {
  source: string;
  build: string;
}

export class DumpOutputCommand extends Command {
  constructor(context: MLIRContext) { super('mlir.dumpOutput', context); }

  /**
   * Walk up from startDir until a directory containing CMakeCache.txt is
   * found. Returns that directory, or null if the filesystem root is reached.
   */
  private findCmakeBuildRoot(startDir: string): string|null {
    let current = path.normalize(startDir);
    while (true) {
      if (fs.existsSync(path.join(current, 'CMakeCache.txt'))) {
        return current;
      }
      const parent = path.dirname(current);
      if (parent === current) {
        return null;  // reached filesystem root
      }
      current = parent;
    }
  }

  /**
   * True if filePath is inside (or equal to) dir.
   */
  private isPathUnderDirectory(filePath: string, dir: string): boolean {
    const normFile = path.normalize(filePath);
    const normDir = path.normalize(dir);
    const rel = path.relative(normDir, normFile);
    return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
  }

  /**
   * Parse all // RUN: lines from fileContent, handling backslash continuations.
   * Returns one logical command string per RUN directive.
   */
  private extractRunLines(fileContent: string): string[] {
    const lines = fileContent.split(/\r?\n/);
    const result: string[] = [];
    const runPrefix = /^\s*\/\/\s*RUN:\s*/;

    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      const match = runPrefix.exec(line);
      if (!match) {
        i++;
        continue;
      }

      // Strip the "// RUN:" prefix.
      let logical = line.slice(match[0].length);

      // Follow backslash continuations. In LLVM/MLIR convention each
      // continuation line also starts with "// RUN:", so strip that same
      // prefix from every continuation line.
      while (logical.endsWith('\\')) {
        logical = logical.slice(0, -1);  // strip trailing backslash
        i++;
        if (i >= lines.length) break;
        const next = lines[i];
        const contMatch = runPrefix.exec(next);
        logical += contMatch ? next.slice(contMatch[0].length) : next;
      }

      result.push(logical.trim());
      i++;
    }
    return result;
  }

  /**
   * Substitute %s and %S in a raw RUN line. Returns null if any unresolvable
   * substitution variable remains after replacement.
   */
  private applyLitSubstitutions(rawLine: string,
                                filePath: string): string|null {
    const fileDir = path.dirname(filePath);

    // Quote a path if it contains spaces.
    const q = (p: string) => p.includes(' ') ? `"${p}"` : p;

    let result = rawLine;
    result = result.replace(/%%/g, '\x00');  // temporarily hide %%
    result = result.replace(/%s/g, q(filePath));
    result = result.replace(/%S/g, q(fileDir));
    result = result.replace(/\x00/g, '%');  // restore %

    // Check for remaining unresolved substitutions.
    if (/%((\{[^}]*\})|[a-zA-Z])/.test(result)) {
      return null;
    }
    return result;
  }

  /**
   * Split a pipeline on | and drop all FileCheck stages.
   * Returns null if no non-FileCheck stages remain.
   */
  private stripFileCheckStages(pipeline: string): string|null {
    const stages = pipeline.split('|');
    const kept = stages.filter(stage => {
      const first = stage.trim().split(/\s+/)[0] ?? '';
      return path.basename(first) !== 'FileCheck';
    });
    if (kept.length === 0) return null;
    return kept.join('|').trim();
  }

  /**
   * Run a shell pipeline, returning captured stdout and the exit code.
   * stderr is forwarded live to outputChannel.
   */
  private runPipeline(pipeline: string, cmakeBinDir: string|null,
                      outputChannel: vscode.OutputChannel):
      Promise<{stdout: string, exitCode: number}> {
    return new Promise((resolve) => {
      const env = {...process.env};
      if (cmakeBinDir) {
        env['PATH'] = cmakeBinDir + path.delimiter + (env['PATH'] ?? '');
      }

      let stdout = '';
      const child = spawn(pipeline, [], {shell : true, env});

      child.stdout?.on('data', (data: Buffer) => { stdout += data.toString(); });
      child.stderr?.on(
          'data',
          (data: Buffer) => { outputChannel.append(data.toString()); });

      child.on('close', (code: number|null) => {
        resolve({stdout, exitCode : code ?? -1});
      });
      child.on('error', (err: Error) => {
        outputChannel.appendLine(
            `[DumpOutput] spawn error: ${err.message}`);
        resolve({stdout : '', exitCode : -1});
      });
    });
  }

  async execute() {
    let outputChannel = this.context.outputChannel;
    if (!outputChannel) {
      outputChannel = vscode.window.createOutputChannel('MLIR');
    }
    outputChannel.show(true);
    outputChannel.clear();
    outputChannel.appendLine('=== Dump Output ===');

    // Validate active editor.
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage('No active editor');
      return;
    }
    if (editor.document.languageId !== 'mlir') {
      vscode.window.showErrorMessage(
          'Current file is not an MLIR file. Please open a .mlir file first.');
      return;
    }
    if (editor.document.uri.scheme !== 'file') {
      vscode.window.showErrorMessage('File must be saved to disk');
      return;
    }

    const fileUri = editor.document.uri;
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(fileUri);
    if (!workspaceFolder) {
      vscode.window.showErrorMessage(
          'No workspace folder found. Please open a workspace.');
      return;
    }

    const filePath = fileUri.fsPath;
    const dumpPath = filePath + '.dump';

    // Resolve cmake bin dir from mlir.litSuites for PATH augmentation.
    let cmakeBinDir: string|null = null;
    const suites = config.get<LitSuite[]>('litSuites', workspaceFolder, []);
    if (Array.isArray(suites)) {
      for (const suite of suites) {
        if (!suite?.source || !suite?.build) continue;
        const workspacePath = workspaceFolder.uri.fsPath;
        const sourceResolved = path.isAbsolute(suite.source) ?
            suite.source :
            path.join(workspacePath, suite.source);
        if (!this.isPathUnderDirectory(filePath, sourceResolved)) continue;
        const buildResolved = path.isAbsolute(suite.build) ?
            suite.build :
            path.join(workspacePath, suite.build);
        const cmakeRoot = this.findCmakeBuildRoot(buildResolved);
        if (cmakeRoot) {
          cmakeBinDir = path.join(cmakeRoot, 'bin');
          outputChannel.appendLine(
              `[DumpOutput] cmake build root: ${cmakeRoot}`);
          outputChannel.appendLine(
              `[DumpOutput] prepending to PATH: ${cmakeBinDir}`);
        } else {
          outputChannel.appendLine(
              `[DumpOutput] WARN: CMakeCache.txt not found above ${
                  buildResolved}, falling back to system PATH`);
        }
        break;
      }
    }
    if (!cmakeBinDir) {
      outputChannel.appendLine(
          '[DumpOutput] No matching litSuite found; using system PATH');
    }

    // Parse and process RUN lines.
    const content = editor.document.getText();
    const rawLines = this.extractRunLines(content);
    if (rawLines.length === 0) {
      vscode.window.showInformationMessage('No RUN: lines found in this file.');
      return;
    }

    const runnableLines: string[] = [];
    for (const rawLine of rawLines) {
      const substituted = this.applyLitSubstitutions(rawLine, filePath);
      if (substituted === null) {
        outputChannel.appendLine(
            `[DumpOutput] WARN: skipping (unresolvable substitution): ${
                rawLine}`);
        continue;
      }
      const stripped = this.stripFileCheckStages(substituted);
      if (stripped === null) {
        outputChannel.appendLine(
            `[DumpOutput] WARN: skipping (all stages are FileCheck): ${
                substituted}`);
        continue;
      }
      runnableLines.push(stripped);
    }

    if (runnableLines.length === 0) {
      vscode.window.showInformationMessage(
          'All RUN: lines were skipped (unresolvable substitutions or FileCheck-only).');
      return;
    }

    // Execute pipelines and collect stdout.
    const outputChunks: string[] = [];
    const n = runnableLines.length;
    for (let i = 0; i < n; i++) {
      const pipeline = runnableLines[i];
      outputChannel.appendLine(
          `[DumpOutput] Running (${i + 1}/${n}): ${pipeline}`);
      const {stdout, exitCode} =
          await this.runPipeline(pipeline, cmakeBinDir, outputChannel);
      if (exitCode !== 0) {
        outputChannel.appendLine(
            `[DumpOutput] WARN: exit code ${exitCode} for pipeline ${i + 1}`);
      }
      const prefix = n > 1 ? `// --- RUN ${i + 1} ---\n` : '';
      outputChunks.push(prefix + stdout);
    }

    const combined = outputChunks.join('\n');

    // Write dump file.
    try {
      await fs.promises.writeFile(dumpPath, combined, 'utf8');
      outputChannel.appendLine(`[DumpOutput] Written: ${dumpPath}`);
    } catch (e: any) {
      vscode.window.showErrorMessage(
          `Failed to write dump file: ${e.message}`);
      return;
    }

    // Open dump file in editor with mlir language.
    try {
      const doc =
          await vscode.workspace.openTextDocument(vscode.Uri.file(dumpPath));
      await vscode.window.showTextDocument(doc, {preview : false});
      await vscode.languages.setTextDocumentLanguage(doc, 'mlir');
    } catch (e: any) {
      outputChannel.appendLine(
          `[DumpOutput] WARN: could not open dump file: ${e.message}`);
    }

    outputChannel.appendLine('[DumpOutput] Done.');
  }
}
