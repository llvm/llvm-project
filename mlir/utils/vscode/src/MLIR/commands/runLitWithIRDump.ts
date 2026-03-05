import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import {spawn} from 'child_process';

import {Command} from '../../command';
import {MLIRContext} from '../../mlirContext';

/**
 * A command that runs lit with IR dump on the current MLIR file.
 */
export class RunLitWithIRDumpCommand extends Command {
  constructor(context: MLIRContext) {
    super('mlir.runLitWithIRDump', context);
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
        vscode.window.showErrorMessage(`[RunLitWithIRDump] Error running command: ${error.message}`);
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
      console.warn('[RunLitWithIRDump] WARNING: Using fallback output channel');
    }
    
    vscode.window.showInformationMessage('[RunLitWithIRDump] Command started');

    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage('No active editor');
      return;
    }
    vscode.window.showInformationMessage(`[RunLitWithIRDump] Active editor found: ${editor.document.fileName}`);

    if (editor.document.languageId !== 'mlir') {
      vscode.window.showErrorMessage(
          'Current file is not an MLIR file. Please open a .mlir file first.');
      return;
    }
    vscode.window.showInformationMessage('[RunLitWithIRDump] File is MLIR format');

    const fileUri = editor.document.uri;
    if (fileUri.scheme !== 'file') {
      vscode.window.showErrorMessage('File must be saved to disk');
      return;
    }

    const filePath = fileUri.fsPath;
    vscode.window.showInformationMessage(`[RunLitWithIRDump] File path: ${filePath}`);

    // Try to find the build directory
    // First, check if we're in a workspace
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(fileUri);
    if (!workspaceFolder) {
      vscode.window.showErrorMessage(
          'No workspace folder found. Please open a workspace.');
      return;
    }

    // Look for build directory in common locations
    const workspacePath = workspaceFolder.uri.fsPath;
    vscode.window.showInformationMessage(`[RunLitWithIRDump] Workspace path: ${workspacePath}`);
    
    const possibleBuildDirs = [
      path.join(workspacePath, 'build'),
      path.join(workspacePath, '..', 'build'),
      path.join(workspacePath, '..', '..', 'build'),
    ];

    let buildDir = null;
    for (const buildPath of possibleBuildDirs) {
      if (fs.existsSync(buildPath) && fs.statSync(buildPath).isDirectory()) {
        buildDir = buildPath;
        vscode.window.showInformationMessage(`[RunLitWithIRDump] Found build directory: ${buildDir}`);
        break;
      }
    }

    if (!buildDir) {
      // Ask user for build directory
      const userBuildDir = await vscode.window.showInputBox({
        prompt: 'Enter the path to the build directory',
        placeHolder: 'e.g., /path/to/build or build',
        value: 'build',
      });

      if (!userBuildDir) {
        return;
      }

      // Resolve the path - try multiple locations for relative paths
      if (path.isAbsolute(userBuildDir)) {
        buildDir = userBuildDir;
        vscode.window.showInformationMessage(`[RunLitWithIRDump] User provided absolute path: ${buildDir}`);
      } else {
        // Try to find build directory by walking up from the file's directory
        // looking for a build/ directory or resolving relative to common locations
        let currentDir = path.dirname(filePath);
        const maxDepth = 10; // Prevent infinite loops
        let depth = 0;
        
        // First, try to find an existing build directory by walking up
        while (depth < maxDepth && currentDir !== path.dirname(currentDir)) {
          const buildPath = path.join(currentDir, userBuildDir);
          if (fs.existsSync(buildPath) && fs.statSync(buildPath).isDirectory()) {
            buildDir = buildPath;
            vscode.window.showInformationMessage(`[RunLitWithIRDump] Found build directory: ${buildDir}`);
            break;
          }
          currentDir = path.dirname(currentDir);
          depth++;
        }
        
        // If not found, try resolving relative path in common locations
        if (!buildDir) {
          const candidatePaths = [
            path.join(workspacePath, userBuildDir),
            path.join(path.dirname(filePath), userBuildDir),
          ];
          
          for (const candidate of candidatePaths) {
            if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) {
              buildDir = candidate;
              vscode.window.showInformationMessage(`[RunLitWithIRDump] Found build directory: ${buildDir}`);
              break;
            }
          }
          
          // If still not found, use workspace path as fallback (will error if doesn't exist)
          if (!buildDir) {
            buildDir = path.join(workspacePath, userBuildDir);
            vscode.window.showInformationMessage(`[RunLitWithIRDump] Using fallback path: ${buildDir}`);
          }
        }
      }

      if (!fs.existsSync(buildDir)) {
        vscode.window.showErrorMessage(
            `Build directory does not exist: ${buildDir}`);
        return;
      }
    }

    // Calculate relative path for lit command
    // Lit typically runs from the build directory
    // The path should be relative to the parent directory of the build directory
    // For example: if build is at <project>/build and file is at <project>/compiler/test/.../file.mlir
    // lit expects: compiler/test/.../file.mlir
    
    // Get the parent directory of the build directory (the project root)
    const buildParentDir = path.dirname(buildDir);
    vscode.window.showInformationMessage(`[RunLitWithIRDump] Build parent directory: ${buildParentDir}`);
    
    // Get path relative to build parent directory
    let relativePath = path.relative(buildParentDir, filePath);
    
    // Normalize the path separators for the shell (use forward slashes)
    relativePath = relativePath.replace(/\\/g, '/');
    vscode.window.showInformationMessage(`[RunLitWithIRDump] Relative path: ${relativePath}`);

    // Try to find or setup lit executable
    let litCommand: string;
    let pythonEnvActivate: string | null = null;
    
    // Step 1: Ask user for lit path or virtual env path (optional)
    // User can press Escape or leave empty to skip
    const userLitPath = await vscode.window.showInputBox({
      prompt: 'Enter path to llvm-lit executable or virtual env (optional - press Escape to auto-detect/create)',
      placeHolder: 'e.g., /path/to/llvm-lit or /path/to/.venv',
      ignoreFocusOut: false,
    });
    
    // If user cancelled (undefined), treat as empty string to proceed with auto-detection
    const userProvidedPath = userLitPath === undefined ? '' : userLitPath;
    // userProvidedPath is a llvm-lit or lit path:
    //   if it is a file, check if it is a llvm-lit or lit executable
    //   if it is a directory, check if */llvm-lit or */lit exists and it is a executable
    // userProvidedPath is a virtual env directory:
    //   check if */bin/lit exists and it is a executable

    let litExecutable: string | null = null;
    let venvPath: string | null = null;
    
    if (userProvidedPath && userProvidedPath.trim() !== '') {
      const userPath = userProvidedPath.trim();
      vscode.window.showInformationMessage(`[RunLitWithIRDump] User provided path: ${userPath}`);
      
      if (fs.existsSync(userPath)) {
        const stats = fs.statSync(userPath);
        
        if (stats.isFile()) {
          // It's a file - check if it's a llvm-lit or lit executable
          const fileName = path.basename(userPath);
          if ((fileName === 'lit' || fileName === 'llvm-lit') && this.isExecutable(userPath)) {
            litExecutable = userPath;
            vscode.window.showInformationMessage(`[RunLitWithIRDump] Using executable: ${litExecutable}`);
          } else {
            vscode.window.showWarningMessage(
                `[RunLitWithIRDump] File is not a valid lit/llvm-lit executable: ${userPath}`);
          }
        } else if (stats.isDirectory()) {
          // It's a directory - check if it's a virtual env or contains lit executables
          vscode.window.showInformationMessage(`[RunLitWithIRDump] User provided directory: ${userPath}`);
          
          // First, check if it's a virtual env (has bin/activate or bin/lit)
          // In virtual env, only check for lit (Python package), not llvm-lit
          const activateScript = path.join(userPath, 'bin', 'activate');
          const litInBin = path.join(userPath, 'bin', 'lit');
          
          if (fs.existsSync(activateScript)) {
            // It's a virtual env
            venvPath = userPath;
            if (fs.existsSync(litInBin) && this.isExecutable(litInBin)) {
              litExecutable = litInBin;
              vscode.window.showInformationMessage(`[RunLitWithIRDump] Using lit from virtual env: ${litExecutable}`);
            } else {
              // Virtual env exists but lit not installed yet - will activate and use lit from PATH
              pythonEnvActivate = `source ${activateScript} && `;
              vscode.window.showInformationMessage(`[RunLitWithIRDump] Will activate virtual env: ${venvPath}`);
            }
          } else {
            // Not a virtual env - check if directory contains llvm-lit or lit executables directly
            const litInDir = path.join(userPath, 'lit');
            const llvmLitInDir = path.join(userPath, 'llvm-lit');
            
            if (fs.existsSync(llvmLitInDir) && this.isExecutable(llvmLitInDir)) {
              litExecutable = llvmLitInDir;
              vscode.window.showInformationMessage(`[RunLitWithIRDump] Using llvm-lit from directory: ${litExecutable}`);
            } else if (fs.existsSync(litInDir) && this.isExecutable(litInDir)) {
              litExecutable = litInDir;
              vscode.window.showInformationMessage(`[RunLitWithIRDump] Using lit from directory: ${litExecutable}`);
            } else {
              vscode.window.showWarningMessage(
                  `[RunLitWithIRDump] Directory does not contain a valid lit/llvm-lit executable: ${userPath}`);
            }
          }
        }
      } else {
        vscode.window.showWarningMessage(`[RunLitWithIRDump] Path does not exist: ${userPath}. Will create venv and install lit.`);
      }
    }
    
    // Step 2: If no user-provided path, create uv venv and install lit
    if (!litExecutable && !pythonEnvActivate) {
      vscode.window.showInformationMessage(`[RunLitWithIRDump] No user-provided path. Will create uv venv and install lit.`);
      
      // Determine where to create the venv (prefer workspace root)
      venvPath = path.join(workspacePath, '.venv');
      vscode.window.showInformationMessage(`[RunLitWithIRDump] Target venv path: ${venvPath}`);
      
      // Check if uv is available
      const uvAvailable = await this.checkCommandAvailable('uv');
      if (!uvAvailable) {
        vscode.window.showErrorMessage(
            'uv is not available. Please install uv or provide a path to llvm-lit.',
            'Install uv'
        ).then(selection => {
          if (selection === 'Install uv') {
            vscode.env.openExternal(vscode.Uri.parse('https://github.com/astral-sh/uv'));
          }
        });
        return;
      }
      vscode.window.showInformationMessage(`[RunLitWithIRDump] uv is available`);
      
      // Create venv if it doesn't exist
      if (!fs.existsSync(venvPath)) {
        vscode.window.showInformationMessage(`[RunLitWithIRDump] Creating uv venv at: ${venvPath}`);
        const createVenvResult = await this.runCommand('uv', ['venv', venvPath], workspacePath, outputChannel);
        if (!createVenvResult.success) {
          vscode.window.showErrorMessage('Failed to create virtual environment');
          return;
        }
        vscode.window.showInformationMessage(`[RunLitWithIRDump] Successfully created venv`);
      } else {
        vscode.window.showInformationMessage(`[RunLitWithIRDump] Venv already exists at: ${venvPath}`);
      }
      
      // Check if lit or llvm-lit is already installed
      const litInVenv = path.join(venvPath, 'bin', 'lit');
      const llvmLitInVenv = path.join(venvPath, 'bin', 'llvm-lit');
      const activateScript = path.join(venvPath, 'bin', 'activate');
      
      if (fs.existsSync(activateScript)) {
        if (fs.existsSync(llvmLitInVenv)) {
          litExecutable = llvmLitInVenv;
          vscode.window.showInformationMessage(`[RunLitWithIRDump] Found existing llvm-lit in venv: ${litExecutable}`);
        } else if (fs.existsSync(litInVenv)) {
          litExecutable = litInVenv;
          vscode.window.showInformationMessage(`[RunLitWithIRDump] Found existing lit in venv: ${litExecutable}`);
        } else {
          // Install lit from testpypi (newer version >=22)
          vscode.window.showInformationMessage(`[RunLitWithIRDump] Installing lit>=22 from testpypi...`);
          const installLitResult = await this.runCommand(
            'uv',
            ['pip', 'install', '--index-url', 'https://test.pypi.org/simple/', '--extra-index-url', 'https://pypi.org/simple/', 'lit>=22'],
            workspacePath,
            outputChannel,
            venvPath
          );
          if (!installLitResult.success) {
            vscode.window.showErrorMessage('Failed to install lit from testpypi');
            return;
          }
          
          // Check if lit or llvm-lit is now available
          if (fs.existsSync(llvmLitInVenv)) {
            litExecutable = llvmLitInVenv;
            vscode.window.showInformationMessage(`[RunLitWithIRDump] Successfully installed llvm-lit: ${litExecutable}`);
          } else if (fs.existsSync(litInVenv)) {
            litExecutable = litInVenv;
            vscode.window.showInformationMessage(`[RunLitWithIRDump] Successfully installed lit: ${litExecutable}`);
          } else {
            // Fallback: use activation script
            pythonEnvActivate = `source ${activateScript} && `;
            vscode.window.showInformationMessage(`[RunLitWithIRDump] Lit installed but not found at expected path, will activate venv`);
          }
        }
      } else {
        vscode.window.showErrorMessage('Virtual environment is invalid');
        return;
      }
    }
    
    // Step 3: Construct the final lit command
    if (litExecutable) {
      litCommand = `${litExecutable} -vv -a ${relativePath}`;
    } else if (pythonEnvActivate) {
      // Try uv run first if in a uv project
      const possibleUvDirs = [
        path.dirname(filePath),
        workspacePath,
        path.join(workspacePath, '..'),
      ];
      
      let useUvRun = false;
      for (const dir of possibleUvDirs) {
        const pyprojectToml = path.join(dir, 'pyproject.toml');
        const uvLock = path.join(dir, 'uv.lock');
        if (fs.existsSync(pyprojectToml) || fs.existsSync(uvLock)) {
          useUvRun = true;
          vscode.window.showInformationMessage(`[RunLitWithIRDump] Detected uv project, will use 'uv run lit'`);
          break;
        }
      }
      
      if (useUvRun) {
        litCommand = `uv run lit -vv -a ${relativePath}`;
        pythonEnvActivate = null; // Don't activate if using uv run
      } else {
        litCommand = `lit -vv -a ${relativePath}`;
        vscode.window.showInformationMessage(`[RunLitWithIRDump] Will activate Python environment before running lit`);
      }
    } else {
      litCommand = `lit -vv -a ${relativePath}`;
      vscode.window.showInformationMessage(`[RunLitWithIRDump] Using system lit (may fail if not in PATH)`);
    }
    
    vscode.window.showInformationMessage(`[RunLitWithIRDump] Lit command: ${litCommand}`);
    vscode.window.showInformationMessage(`[RunLitWithIRDump] Build directory (cwd): ${buildDir}`);

    // Create a terminal and run the command from the build directory
    const terminal = vscode.window.createTerminal({
      name: 'MLIR Lit with IR Dump',
      cwd: buildDir,
    });

    terminal.show();
    
    // Send command with environment activation if needed
    if (pythonEnvActivate) {
      terminal.sendText(`${pythonEnvActivate}${litCommand}`);
    } else {
      terminal.sendText(litCommand);
    }
    
    vscode.window.showInformationMessage(`[RunLitWithIRDump] Terminal command sent successfully`);
    vscode.window.showInformationMessage(`Running lit with IR dump on: ${relativePath}`);
  }
}
