import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import {spawn} from 'child_process';

import {Command} from '../../command';
import {MLIRContext} from '../../mlirContext';
import * as config from '../../config';

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
   * Check if a command is available in the system PATH
   */
  private async checkCommandAvailable(command: string): Promise<boolean> {
    return new Promise((resolve) => {
      const childProcess = spawn('which', [command], {shell: true});
      childProcess.on('close', (code) => {
        resolve(code === 0);
      });
      childProcess.on('error', () => {
        resolve(false);
      });
    });
  }

  /**
   * Get or setup lit executable and construct the command
   * @param workspaceFolder The workspace folder (for reading settings)
   * @param relativePath The relative path to the test file (for lit command)
   * @param outputChannel The output channel for command output
   * @returns The lit command and activation string, or null if setup failed
   */
  private async getLitSetup(
      workspaceFolder: vscode.WorkspaceFolder,
      relativePath: string,
      outputChannel: vscode.OutputChannel
  ): Promise<{litCommand: string; pythonEnvActivate: string | null} | null> {
    let litExecutable: string | null = null;
    let pythonEnvActivate: string | null = null;

    // Get lit executable path from workspace settings only
    const settingsLitPath = config.get<string>('litExecutablePath', workspaceFolder);
    
    if (!settingsLitPath || settingsLitPath.trim() === '') {
      const errorMsg = 'Lit executable path not found in workspace settings. Please set "mlir.litExecutablePath" in .vscode/settings.json';
      vscode.window.showErrorMessage(errorMsg);
      outputChannel.appendLine(`[RunTest] ERROR: ${errorMsg}`);
      return null;
    }

    const workspacePath = workspaceFolder.uri.fsPath;
    const trimmedPath = settingsLitPath.trim();
    outputChannel.appendLine(`[RunTest] Using lit path from settings: ${trimmedPath}`);

    // Resolve the path - handle both absolute and relative paths
    let userPath: string;
    if (path.isAbsolute(trimmedPath)) {
      // Absolute path - use as is
      userPath = trimmedPath;
    } else {
      // Relative path - resolve relative to workspace directory
      userPath = path.join(workspacePath, trimmedPath);
    }

    outputChannel.appendLine(`[RunTest] Resolved path: ${userPath}`);

    // Validate that the path exists
    if (!fs.existsSync(userPath)) {
      const errorMsg = `Lit executable path does not exist: ${userPath} (resolved from: ${trimmedPath})`;
      vscode.window.showErrorMessage(errorMsg);
      outputChannel.appendLine(`[RunTest] ERROR: ${errorMsg}`);
      return null;
    }

    const stats = fs.statSync(userPath);

    if (!stats.isDirectory()) {
      const errorMsg = `Path is not a directory: ${userPath}`;
      vscode.window.showErrorMessage(errorMsg);
      outputChannel.appendLine(`[RunTest] ERROR: ${errorMsg}`);
      return null;
    }

    // It's a directory - check if it's a virtual env or contains lit executables
    outputChannel.appendLine(`[RunTest] lit executable directory: ${userPath}`);

    // First, check if it's a virtual env (has bin/activate)
    const activateScript = path.join(userPath, 'bin', 'activate');
    const litInBin = path.join(userPath, 'bin', 'lit');
    const llvmLitInBin = path.join(userPath, 'bin', 'llvm-lit');

    if (fs.existsSync(activateScript)) {
      // It's a virtual env - check for lit or llvm-lit in bin/
      if (fs.existsSync(llvmLitInBin) && this.isExecutable(llvmLitInBin)) {
        litExecutable = llvmLitInBin;
        outputChannel.appendLine(`[RunTest] Using llvm-lit from virtual env: ${litExecutable}`);
      } else if (fs.existsSync(litInBin) && this.isExecutable(litInBin)) {
        litExecutable = litInBin;
        outputChannel.appendLine(`[RunTest] Using lit from virtual env: ${litExecutable}`);
      } else {
        // Virtual env exists but lit not found - will activate and use lit from PATH
        pythonEnvActivate = `source ${activateScript} && `;
        outputChannel.appendLine(`[RunTest] Will activate virtual env: ${userPath}`);
      }
    } else {
      // Not a virtual env - check if directory contains llvm-lit or lit executables directly
      const litInDir = path.join(userPath, 'lit');
      const llvmLitInDir = path.join(userPath, 'llvm-lit');

      if (fs.existsSync(llvmLitInDir) && this.isExecutable(llvmLitInDir)) {
        litExecutable = llvmLitInDir;
        outputChannel.appendLine(`[RunTest] Using llvm-lit executable: ${litExecutable}`);
      } else if (fs.existsSync(litInDir) && this.isExecutable(litInDir)) {
        litExecutable = litInDir;
        outputChannel.appendLine(`[RunTest] Using lit executable: ${litExecutable}`);
      } else {
        const errorMsg = `Directory does not contain a valid lit/llvm-lit executable: ${userPath}`;
        vscode.window.showErrorMessage(errorMsg);
        outputChannel.appendLine(`[RunTest] ERROR: ${errorMsg}`);
        return null;
      }
    }

    // Construct the final lit command
    let litCommand: string;
    if (litExecutable) {
      litCommand = `${litExecutable} -vv -a ${relativePath}`;
    } else if (pythonEnvActivate) {
      litCommand = `lit -vv -a ${relativePath}`;
      outputChannel.appendLine(`[RunTest] Will activate Python environment before running lit`);
    } else {
      const errorMsg = 'Failed to determine lit executable or activation method';
      vscode.window.showErrorMessage(errorMsg);
      outputChannel.appendLine(`[RunTest] ERROR: ${errorMsg}`);
      return null;
    }

    return {
      litCommand,
      pythonEnvActivate,
    };
  }

  /**
   * Get the build directory from workspace settings
   * @param workspaceFolder The workspace folder
   * @param outputChannel The output channel for logging
   * @returns The resolved build directory path, or null if not found in settings or invalid
   */
  private async getBuildDirectory(
      workspaceFolder: vscode.WorkspaceFolder,
      outputChannel: vscode.OutputChannel
  ): Promise<string | null> {
    // workspacePath: /workspaces/TensorRT-Incubator/mlir-tensorrt
    const workspacePath = workspaceFolder.uri.fsPath;
    // Get build directory from workspace settings only
    const settingsBuildDir = config.get<string>('litBuildDirectory', workspaceFolder);
    
    if (!settingsBuildDir || settingsBuildDir.trim() === '') {
      const errorMsg = 'Build directory not found in workspace settings. Please set "mlir.litBuildDirectory" in .vscode/settings.json';
      vscode.window.showErrorMessage(errorMsg);
      outputChannel.appendLine(`[RunTest] ERROR: ${errorMsg}`);
      return null;
    }

    // Resolve the build directory path
    let buildDir: string;
    const trimmedPath = settingsBuildDir.trim();
    if (path.isAbsolute(trimmedPath)) {
      // Absolute path - use as is
      buildDir = trimmedPath;
    } else {
      // Relative path - resolve relative to workspace's directory
      const candidatePaths = [
        path.join(workspacePath, trimmedPath),
      ];
      
      // Check if any candidate exists
      let found = false;
      for (const candidate of candidatePaths) {
        if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) {
          buildDir = candidate;
          found = true;
          break;
        }
      }
      
      // If not found, use workspace path as default (will validate existence below)
      if (!found) {
        buildDir = path.join(workspacePath, trimmedPath);
      }
    }

    // Validate that the build directory exists
    if (!fs.existsSync(buildDir)) {
      vscode.window.showErrorMessage(
          `Build directory does not exist: ${buildDir}`);
      return null;
    }

    if (!fs.statSync(buildDir).isDirectory()) {
      vscode.window.showErrorMessage(
          `Build Path is not a directory: ${buildDir}`);
      return null;
    }

    outputChannel.appendLine(`[RunTest] Using build directory: ${buildDir}`);
    return buildDir;
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

    // Get build directory from user
    const buildDir = await this.getBuildDirectory(workspaceFolder, outputChannel);
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

    const {litCommand, pythonEnvActivate} = litSetup;
    outputChannel.appendLine(`[RunTest] Lit command: ${litCommand}`);
    outputChannel.appendLine(`[RunTest] Build directory (cwd): ${buildDir}`);

    // Create a terminal and run the command from the build directory
    const terminal = vscode.window.createTerminal({
      name: 'Run MLIR Lit',
      cwd: buildDir,
    });

    terminal.show();
    
    // Send command with environment activation if needed
    if (pythonEnvActivate) {
      terminal.sendText(`${pythonEnvActivate}${litCommand}`);
    } else {
      terminal.sendText(litCommand);
    }
    
    outputChannel.appendLine(`[RunTest] Terminal command sent successfully`);
    outputChannel.appendLine(`[RunTest] Running MLIR lit on: ${relativePath}`);
  }
}
