import * as vscode from "vscode";
import { LLDBDapServer } from "./lldb-dap-server";
import { createDebugAdapterExecutable } from "./debug-adapter-factory";
import { showErrorWithConfigureButton } from "./ui/error-messages";

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

    // Check if we're going to launch a debug session or use an existing process
    if ("debugAdapterPort" in debugConfiguration) {
      if (
        "debugAdapterExecutable" in debugConfiguration ||
        "debugAdapterArgs" in debugConfiguration
      ) {
        return showErrorWithConfigureButton(
          "The debugAdapterPort property is incompatible with debugAdapterExecutable and debugAdapterArgs. Please update your launch configuration.",
        );
      }
    } else {
      // Always try to create the debug adapter executable as this will show the user errors
      // if there are any.
      const executable = await createDebugAdapterExecutable(
        folder,
        debugConfiguration,
        /* userInteractive */ true,
      );

      // Server mode needs to be handled here since DebugAdapterDescriptorFactory
      // will show an unhelpful error if it returns undefined. We'd rather show a
      // nicer error message here and allow stopping the debug session gracefully.
      const config = vscode.workspace.getConfiguration("lldb-dap", folder);
      if (config.get<boolean>("serverMode", false)) {
        if (!executable) {
          return undefined;
        }
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
