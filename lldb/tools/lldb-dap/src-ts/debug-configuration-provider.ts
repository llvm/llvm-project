import * as vscode from "vscode";
import * as child_process from "child_process";
import * as util from "util";
import { LLDBDapServer } from "./lldb-dap-server";
import { createDebugAdapterExecutable } from "./debug-adapter-factory";
import { showErrorWithConfigureButton } from "./ui/error-messages";

const exec = util.promisify(child_process.execFile);

/**
 * Determines whether or not the given lldb-dap executable supports executing
 * in server mode.
 *
 * @param exe the path to the lldb-dap executable
 * @returns a boolean indicating whether or not lldb-dap supports server mode
 */
async function isServerModeSupported(exe: string): Promise<boolean> {
  const { stdout } = await exec(exe, ["--help"]);
  return /--connection/.test(stdout);
}

export class LLDBDapConfigurationProvider
  implements vscode.DebugConfigurationProvider
{
  constructor(private readonly server: LLDBDapServer) {}

  async resolveDebugConfiguration(
    folder: vscode.WorkspaceFolder | undefined,
    debugConfiguration: vscode.DebugConfiguration,
    _token?: vscode.CancellationToken,
  ): Promise<vscode.DebugConfiguration | null | undefined> {
    if (
      "debugAdapterHost" in debugConfiguration &&
      !("debugAdapterPort" in debugConfiguration)
    ) {
      return showErrorWithConfigureButton(
        "A debugAdapterPort must be provided when debugAdapterHost is set. Please update your launch configuration.",
      );
    }

    if (
      "debugAdapterPort" in debugConfiguration &&
      ("debugAdapterExecutable" in debugConfiguration ||
        "debugAdapterArgs" in debugConfiguration)
    ) {
      return showErrorWithConfigureButton(
        "The debugAdapterPort property is incompatible with debugAdapterExecutable and debugAdapterArgs. Please update your launch configuration.",
      );
    }

    // Server mode needs to be handled here since DebugAdapterDescriptorFactory
    // will show an unhelpful error if it returns undefined. We'd rather show a
    // nicer error message here and allow stopping the debug session gracefully.
    const config = vscode.workspace.getConfiguration("lldb-dap", folder);
    if (config.get<boolean>("serverMode", false)) {
      const executable = await createDebugAdapterExecutable(
        folder,
        debugConfiguration,
        /* userInteractive */ true,
      );
      if (!executable) {
        return undefined;
      }
      if (await isServerModeSupported(executable.command)) {
        const serverInfo = await this.server.start(
          executable.command,
          executable.args,
          executable.options,
        );
        if (!serverInfo) {
          return undefined;
        }
        // Use a debug adapter host and port combination rather than an executable
        // and list of arguments.
        delete debugConfiguration.debugAdapterExecutable;
        delete debugConfiguration.debugAdapterArgs;
        debugConfiguration.debugAdapterHost = serverInfo.host;
        debugConfiguration.debugAdapterPort = serverInfo.port;
      }
    }

    return debugConfiguration;
  }
}
