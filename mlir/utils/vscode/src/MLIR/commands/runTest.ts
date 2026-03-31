import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import {spawn} from 'child_process';

import {Command} from '../../command';
import {MLIRContext} from '../../mlirContext';
import * as config from '../../config';

/** One entry in `mlir.litSuites`: source dir on disk maps to lit cwd in build tree. */
interface LitSuite {
  source: string;
  build: string;
}

/**
 * A command that runs lit with IR dump on the current MLIR file.
 */
export class RunTestCommand extends Command {
  constructor(context: MLIRContext) {
    super('mlir.runTest', context);
  }

  /**
   * Check if a file is executable
   */
  private isExecutable(filePath: string): boolean {
    try {
      fs.accessSync(filePath, fs.constants.X_OK);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Resolve a command name to an absolute path using PATH (which / where).
   */
  private async resolveInPath(command: string): Promise<string | null> {
    const tool = process.platform === 'win32' ? 'where' : 'which';
    return new Promise((resolve) => {
      const childProcess = spawn(tool, [command], {shell: true});
      let out = '';
      childProcess.stdout?.on('data', (data) => {
        out += data.toString();
      });
      childProcess.on('close', (code) => {
        if (code !== 0) {
          resolve(null);
          return;
        }
        const firstLine = out.trim().split(/\r?\n/)[0]?.trim();
        resolve(firstLine && firstLine.length > 0 ? firstLine : null);
      });
      childProcess.on('error', () => {
        resolve(null);
      });
    });
  }

  /**
   * Resolve the lit or llvm-lit executable: workspace setting, then PATH, then user prompt.
   */
  private async resolveLitExecutablePath(
      workspaceFolder: vscode.WorkspaceFolder,
      outputChannel: vscode.OutputChannel
  ): Promise<string | null> {
    const workspacePath = workspaceFolder.uri.fsPath;
    const settingsLitPath = config.get<string>('litExecutablePath', workspaceFolder);
    const trimmedSetting = settingsLitPath?.trim() ?? '';

    if (trimmedSetting) {
      const resolved = path.isAbsolute(trimmedSetting) ?
          trimmedSetting :
          path.join(workspacePath, trimmedSetting);
      outputChannel.appendLine(
          `[RunTest] mlir.litExecutablePath (resolved): ${resolved}`);

      if (!fs.existsSync(resolved)) {
        const msg =
            `Lit executable does not exist: ${resolved} (from mlir.litExecutablePath)`;
        vscode.window.showErrorMessage(msg);
        outputChannel.appendLine(`[RunTest] ERROR: ${msg}`);
        return null;
      }
      if (fs.statSync(resolved).isDirectory()) {
        const msg =
            'mlir.litExecutablePath must be the full path to the lit or llvm-lit executable, not a directory.';
        vscode.window.showErrorMessage(msg);
        outputChannel.appendLine(`[RunTest] ERROR: ${msg}`);
        return null;
      }
      if (!this.isExecutable(resolved)) {
        const msg = `Lit executable is not executable: ${resolved}`;
        vscode.window.showErrorMessage(msg);
        outputChannel.appendLine(`[RunTest] ERROR: ${msg}`);
        return null;
      }
      return resolved;
    }

    for (const cmd of ['lit', 'llvm-lit']) {
      const found = await this.resolveInPath(cmd);
      if (found && fs.existsSync(found) && !fs.statSync(found).isDirectory() &&
          this.isExecutable(found)) {
        outputChannel.appendLine(`[RunTest] Using ${cmd} from PATH: ${found}`);
        return found;
      }
    }

    const input = await vscode.window.showInputBox({
      title: 'Lit or llvm-lit executable',
      prompt:
          'lit and llvm-lit were not found in PATH. Enter the full path to lit or llvm-lit.',
      ignoreFocusOut: true,
      validateInput: (value) => {
        const t = value.trim();
        if (!t) {
          return 'Enter a path, or cancel.';
        }
        const candidate =
            path.isAbsolute(t) ? t : path.join(workspacePath, t);
        if (!fs.existsSync(candidate)) {
          return 'Path does not exist.';
        }
        if (fs.statSync(candidate).isDirectory()) {
          return 'Path must be the executable file, not a directory.';
        }
        if (!this.isExecutable(candidate)) {
          return 'File is not executable.';
        }
        return null;
      },
    });

    if (!input?.trim()) {
      outputChannel.appendLine('[RunTest] No lit path: cancelled or empty input');
      return null;
    }

    const resolved = path.isAbsolute(input.trim()) ?
        input.trim() :
        path.join(workspacePath, input.trim());
    outputChannel.appendLine(`[RunTest] Using lit from user input: ${resolved}`);
    return resolved;
  }

  /**
   * Get lit executable and construct the command
   * @param workspaceFolder The workspace folder (for reading settings)
   * @param relativePath The relative path to the test file (for lit command)
   * @param outputChannel The output channel for command output
   * @returns The lit shell command, or null if setup failed
   */
  private async getLitSetup(
      workspaceFolder: vscode.WorkspaceFolder,
      relativePath: string,
      outputChannel: vscode.OutputChannel
  ): Promise<{litCommand: string} | null> {
    const litExecutable = await this.resolveLitExecutablePath(
        workspaceFolder, outputChannel);
    if (!litExecutable) {
      return null;
    }

    const quoted =
        litExecutable.includes(' ') ? JSON.stringify(litExecutable) : litExecutable;
    const litCommand = `${quoted} -vv -a ${relativePath}`;
    return {litCommand};
  }

  /**
   * Resolve a workspace-relative or absolute path string against the workspace root.
   */
  private resolveWorkspacePath(workspacePath: string, p: string): string {
    const t = p.trim();
    if (!t) {
      return '';
    }
    return path.isAbsolute(t) ? path.normalize(t) :
                                path.normalize(path.join(workspacePath, t));
  }

  /**
   * True if `filePath` is `sourceDir` or a file/directory under it.
   */
  private isPathUnderDirectory(filePath: string, sourceDir: string): boolean {
    const normFile = path.normalize(filePath);
    const normDir = path.normalize(sourceDir);
    const rel = path.relative(normDir, normFile);
    return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
  }

  /**
   * Validate resolved build directory exists and return it, or null with error.
   */
  private finishBuildDirectory(
      buildDir: string,
      outputChannel: vscode.OutputChannel
  ): string | null {
    if (!fs.existsSync(buildDir)) {
      vscode.window.showErrorMessage(
          `Build directory does not exist: ${buildDir}`);
      return null;
    }
    if (!fs.statSync(buildDir).isDirectory()) {
      vscode.window.showErrorMessage(
          `Build path is not a directory: ${buildDir}`);
      return null;
    }
    outputChannel.appendLine(`[RunTest] Using build directory: ${buildDir}`);
    return buildDir;
  }

  /**
   * Resolve lit cwd from the first matching `mlir.litSuites` entry for this file.
   * @param workspaceFolder The workspace folder
   * @param filePath Absolute path to the .mlir file being tested
   * @param outputChannel The output channel for logging
   * @returns The resolved build directory path, or null if not found or invalid
   */
  private async getBuildDirectory(
      workspaceFolder: vscode.WorkspaceFolder,
      filePath: string,
      outputChannel: vscode.OutputChannel
  ): Promise<string | null> {
    const workspacePath = workspaceFolder.uri.fsPath;
    const suites = config.get<LitSuite[]>('litSuites', workspaceFolder, []);

    if (!Array.isArray(suites) || suites.length === 0) {
      const errorMsg =
          'mlir.litSuites is empty or missing. Add at least one { "source", "build" } entry in workspace settings.';
      vscode.window.showErrorMessage(errorMsg);
      outputChannel.appendLine(`[RunTest] ERROR: ${errorMsg}`);
      return null;
    }

    for (const suite of suites) {
      if (!suite || typeof suite.source !== 'string' ||
          typeof suite.build !== 'string') {
        continue;
      }
      const sourceResolved = this.resolveWorkspacePath(
          workspacePath, suite.source);
      if (!sourceResolved) {
        continue;
      }
      if (!fs.existsSync(sourceResolved) ||
          !fs.statSync(sourceResolved).isDirectory()) {
        outputChannel.appendLine(
            `[RunTest] Skip suite (source missing or not a dir): ${sourceResolved}`);
        continue;
      }
      if (!this.isPathUnderDirectory(filePath, sourceResolved)) {
        continue;
      }
      const buildDir =
          this.resolveWorkspacePath(workspacePath, suite.build);
      outputChannel.appendLine(
          `[RunTest] Matched lit suite source="${suite.source}" -> build="${suite.build}"`);
      return this.finishBuildDirectory(buildDir, outputChannel);
    }

    const errorMsg =
        'No mlir.litSuites entry contains this file. Add or reorder a suite so its source directory includes the test path.';
    vscode.window.showErrorMessage(errorMsg);
    outputChannel.appendLine(`[RunTest] ERROR: ${errorMsg}`);
    return null;
  }

  /**
   * Run a command and return the result
   */
  private async runCommand(
      command: string,
      args: string[],
      cwd: string,
      outputChannel: vscode.OutputChannel,
      venvPath?: string
  ): Promise<{success: boolean, output: string}> {
    return new Promise((resolve) => {
      let output = '';
      let errorOutput = '';
      
      // If venv is provided, activate it first
      const env = {...process.env};
      if (venvPath) {
        const pythonPath = path.join(venvPath, 'bin', 'python');
        if (fs.existsSync(pythonPath)) {
          env.PATH = `${path.join(venvPath, 'bin')}:${env.PATH}`;
          env.VIRTUAL_ENV = venvPath;
        }
      }
      
      const childProcess = spawn(command, args, {
        cwd: cwd,
        shell: true,
        env: env,
      });
      
      childProcess.stdout?.on('data', (data) => {
        const text = data.toString();
        output += text;
        outputChannel.append(text);
      });
      
      childProcess.stderr?.on('data', (data) => {
        const text = data.toString();
        errorOutput += text;
        outputChannel.append(text);
      });
      
      childProcess.on('close', (code) => {
        resolve({
          success: code === 0,
          output: output + errorOutput,
        });
      });
      
      childProcess.on('error', (error) => {
        vscode.window.showErrorMessage(`[RunTest] Error running command: ${error.message}`);
        resolve({
          success: false,
          output: error.message,
        });
      });
    });
  }

  async execute() {
    // Ensure output channel exists and is shown
    let outputChannel = this.context.outputChannel;
    if (!outputChannel) {
      // Fallback: create a new output channel if context doesn't have one
      outputChannel = vscode.window.createOutputChannel('MLIR');
      console.warn('[RunTest] WARNING: Using fallback output channel');
    }
    // Show the output channel so messages are visible
    outputChannel.show(true);
    outputChannel.clear();
    outputChannel.appendLine('=== Run Lit with IR Dump ===');

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

    const fileUri = editor.document.uri;
    if (fileUri.scheme !== 'file') {
      vscode.window.showErrorMessage('File must be saved to disk');
      return;
    }

    const filePath = fileUri.fsPath;
    // Get workspace folder for later use
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(fileUri);
    if (!workspaceFolder) {
      vscode.window.showErrorMessage(
          'No workspace folder found. Please open a workspace.');
      return;
    }

    // Resolve build directory from mlir.litSuites (first matching source)
    const buildDir =
        await this.getBuildDirectory(workspaceFolder, filePath, outputChannel);
    if (!buildDir) {
      return;
    }

    const workspacePath = workspaceFolder.uri.fsPath;
    
    // Get the parent directory of the build directory (the project root)
    const buildParentDir = path.dirname(buildDir);
    outputChannel.appendLine(`[RunTest] Build parent directory: ${buildParentDir}`);
    
    // Get path relative to build parent directory
    let relativePath = path.relative(buildParentDir, filePath);
    
    // Normalize the path separators for the shell (use forward slashes)
    relativePath = relativePath.replace(/\\/g, '/');
    outputChannel.appendLine(`[RunTest] mlir file relative path: ${relativePath}`);

    // Get or setup lit executable and construct the command
    const litSetup = await this.getLitSetup(workspaceFolder, relativePath, outputChannel);
    if (!litSetup) {
      return;
    }

    const {litCommand} = litSetup;
    outputChannel.appendLine(`[RunTest] Lit command: ${litCommand}`);
    outputChannel.appendLine(`[RunTest] Build directory (cwd): ${buildDir}`);

    // Create a terminal and run the command from the build directory
    const terminal = vscode.window.createTerminal({
      name: 'Run MLIR Lit',
      cwd: buildDir,
    });

    terminal.show();
    terminal.sendText(litCommand);
    
    outputChannel.appendLine(`[RunTest] Terminal command sent successfully`);
    outputChannel.appendLine(`[RunTest] Running MLIR lit on: ${relativePath}`);
  }
}
