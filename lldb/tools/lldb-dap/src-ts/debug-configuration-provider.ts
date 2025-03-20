import * as vscode from "vscode";
import * as child_process from "child_process";
import * as util from "util";
import { LLDBDapServer } from "./lldb-dap-server";
import { createDebugAdapterExecutable } from "./debug-adapter-factory";
import { ConfigureButton, showErrorMessage } from "./ui/show-error-message";
import { ErrorWithNotification } from "./ui/error-with-notification";

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

  async resolveDebugConfigurationWithSubstitutedVariables(
    folder: vscode.WorkspaceFolder | undefined,
    debugConfiguration: vscode.DebugConfiguration,
    _token?: vscode.CancellationToken,
  ): Promise<vscode.DebugConfiguration | null | undefined> {
    try {
      if (
        "debugAdapterHost" in debugConfiguration &&
        !("debugAdapterPort" in debugConfiguration)
      ) {
        throw new ErrorWithNotification(
          "A debugAdapterPort must be provided when debugAdapterHost is set. Please update your launch configuration.",
          new ConfigureButton(),
        );
      }

      // Check if we're going to launch a debug session or use an existing process
      if ("debugAdapterPort" in debugConfiguration) {
        if (
          "debugAdapterExecutable" in debugConfiguration ||
          "debugAdapterArgs" in debugConfiguration
        ) {
          throw new ErrorWithNotification(
            "The debugAdapterPort property is incompatible with debugAdapterExecutable and debugAdapterArgs. Please update your launch configuration.",
            new ConfigureButton(),
          );
        }
      } else {
        // Always try to create the debug adapter executable as this will show the user errors
        // if there are any.
        const executable = await createDebugAdapterExecutable(
          folder,
          debugConfiguration,
        );
        if (!executable) {
          return undefined;
        }

        // Server mode needs to be handled here since DebugAdapterDescriptorFactory
        // will show an unhelpful error if it returns undefined. We'd rather show a
        // nicer error message here and allow stopping the debug session gracefully.
        const config = vscode.workspace.getConfiguration("lldb-dap", folder);
        if (
          config.get<boolean>("serverMode", false) &&
          (await isServerModeSupported(executable.command))
        ) {
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
    } catch (error) {
      // Show a better error message to the user if possible
      if (!(error instanceof ErrorWithNotification)) {
        throw error;
      }
      return await error.showNotification({
        modal: true,
        showConfigureButton: true,
      });
    }
  }
}
