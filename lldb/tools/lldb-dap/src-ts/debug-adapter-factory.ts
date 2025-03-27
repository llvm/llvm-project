import * as path from "path";
import * as util from "util";
import * as vscode from "vscode";
import * as child_process from "child_process";
import * as fs from "node:fs/promises";

const exec = util.promisify(child_process.execFile);

export async function isExecutable(path: string): Promise<Boolean> {
  try {
    await fs.access(path, fs.constants.X_OK);
  } catch {
    return false;
  }
  return true;
}

async function findWithXcrun(executable: string): Promise<string | undefined> {
  if (process.platform === "darwin") {
    try {
      let { stdout, stderr } = await exec("/usr/bin/xcrun", [
        "-find",
        executable,
      ]);
      if (stdout) {
        return stdout.toString().trimEnd();
      }
    } catch (error) { }
  }
  return undefined;
}

async function findInPath(executable: string): Promise<string | undefined> {
  const env_path =
    process.platform === "win32" ? process.env["Path"] : process.env["PATH"];
  if (!env_path) {
    return undefined;
  }

  const paths = env_path.split(path.delimiter);
  for (const p of paths) {
    const exe_path = path.join(p, executable);
    if (await isExecutable(exe_path)) {
      return exe_path;
    }
  }
  return undefined;
}

async function findDAPExecutable(): Promise<string | undefined> {
  const executable = process.platform === "win32" ? "lldb-dap.exe" : "lldb-dap";

  // Prefer lldb-dap from Xcode on Darwin.
  const xcrun_dap = await findWithXcrun(executable);
  if (xcrun_dap) {
    return xcrun_dap;
  }

  // Find lldb-dap in the user's path.
  const path_dap = await findInPath(executable);
  if (path_dap) {
    return path_dap;
  }

  return undefined;
}

async function getDAPExecutable(
  session: vscode.DebugSession,
): Promise<string | undefined> {
  // Check if the executable was provided in the launch configuration.
  const launchConfigPath = session.configuration["debugAdapterExecutable"];
  if (typeof launchConfigPath === "string" && launchConfigPath.length !== 0) {
    return launchConfigPath;
  }

  // Check if the executable was provided in the extension's configuration.
  const config = vscode.workspace.getConfiguration(
    "lldb-dap",
    session.workspaceFolder,
  );
  const configPath = config.get<string>("executable-path");
  if (configPath && configPath.length !== 0) {
    return configPath;
  }

  // Try finding the lldb-dap binary.
  const foundPath = await findDAPExecutable();
  if (foundPath) {
    return foundPath;
  }

  return undefined;
}

async function isServerModeSupported(exe: string): Promise<boolean> {
  const { stdout } = await exec(exe, ['--help']);
  return /--connection/.test(stdout);
}

/**
 * This class defines a factory used to find the lldb-dap binary to use
 * depending on the session configuration.
 */
export class LLDBDapDescriptorFactory
  implements vscode.DebugAdapterDescriptorFactory, vscode.Disposable {
  private server?: Promise<{ process: child_process.ChildProcess, host: string, port: number }>;

  dispose() {
    this.server?.then(({ process }) => {
      process.kill();
    });
  }

  async createDebugAdapterDescriptor(
    session: vscode.DebugSession,
    executable: vscode.DebugAdapterExecutable | undefined,
  ): Promise<vscode.DebugAdapterDescriptor | undefined> {
    const config = vscode.workspace.getConfiguration(
      "lldb-dap",
      session.workspaceFolder,
    );

    const log_path = config.get<string>("log-path");
    let env: { [key: string]: string } = {};
    if (log_path) {
      env["LLDBDAP_LOG"] = log_path;
    }
    const configEnvironment =
      config.get<{ [key: string]: string }>("environment") || {};
    const dapPath = (await getDAPExecutable(session)) ?? executable?.command;

    if (!dapPath) {
      LLDBDapDescriptorFactory.showLLDBDapNotFoundMessage();
      return undefined;
    }

    if (!(await isExecutable(dapPath))) {
      LLDBDapDescriptorFactory.showLLDBDapNotFoundMessage(dapPath);
      return;
    }

    const dbgOptions = {
      env: {
        ...executable?.options?.env,
        ...configEnvironment,
        ...env,
      },
    };
    const dbgArgs = executable?.args ?? [];

    const serverMode = config.get<boolean>('serverMode', false);
    if (serverMode && await isServerModeSupported(dapPath)) {
      const { host, port } = await this.startServer(dapPath, dbgArgs, dbgOptions);
      return new vscode.DebugAdapterServer(port, host);
    }

    return new vscode.DebugAdapterExecutable(dapPath, dbgArgs, dbgOptions);
  }

  startServer(dapPath: string, args: string[], options: child_process.CommonSpawnOptions): Promise<{ host: string, port: number }> {
    if (this.server) return this.server;

    this.server = new Promise(resolve => {
      args.push(
        '--connection',
        'connect://localhost:0'
      );
      const server = child_process.spawn(dapPath, args, options);
      server.stdout!.setEncoding('utf8').once('data', (data: string) => {
        const connection = /connection:\/\/\[([^\]]+)\]:(\d+)/.exec(data);
        if (connection) {
          const host = connection[1];
          const port = Number(connection[2]);
          resolve({ process: server, host, port });
        }
      });
      server.on('exit', () => {
        this.server = undefined;
      })
    });
    return this.server;
  }

  /**
   * Shows a message box when the debug adapter's path is not found
   */
  static async showLLDBDapNotFoundMessage(path?: string) {
    const message =
      path
        ? `Debug adapter path: ${path} is not a valid file.`
        : "Unable to find the path to the LLDB debug adapter executable.";
    const openSettingsAction = "Open Settings";
    const callbackValue = await vscode.window.showErrorMessage(
      message,
      openSettingsAction,
    );

    if (openSettingsAction === callbackValue) {
      vscode.commands.executeCommand(
        "workbench.action.openSettings",
        "lldb-dap.executable-path",
      );
    }
  }
}
