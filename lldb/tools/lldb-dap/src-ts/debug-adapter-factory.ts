import * as vscode from "vscode";
import { LLDBDapOptions } from "./types";

/**
 * This class defines a factory used to find the lldb-dap binary to use
 * depending on the session configuration.
 */
export class LLDBDapDescriptorFactory
  implements vscode.DebugAdapterDescriptorFactory
{
  private lldbDapOptions: LLDBDapOptions;

  constructor(lldbDapOptions: LLDBDapOptions) {
    this.lldbDapOptions = lldbDapOptions;
  }

  static async isValidDebugAdapterPath(
    pathUri: vscode.Uri,
  ): Promise<Boolean> {
    try {
      const fileStats = await vscode.workspace.fs.stat(pathUri);
      if (!(fileStats.type & vscode.FileType.File)) {
        return false;
      }
    } catch (err) {
      return false;
    }
    return true;
  }

  async createDebugAdapterDescriptor(
    session: vscode.DebugSession,
    executable: vscode.DebugAdapterExecutable | undefined,
  ): Promise<vscode.DebugAdapterDescriptor | undefined> {
    const config = vscode.workspace.getConfiguration(
      "lldb-dap",
      session.workspaceFolder,
    );
    const customPath = config.get<string>("executable-path");
    const path: string = customPath || executable!!.command;

    const fileUri = vscode.Uri.file(path);
    if (!(await LLDBDapDescriptorFactory.isValidDebugAdapterPath(fileUri))) {
      LLDBDapDescriptorFactory.showLLDBDapNotFoundMessage(fileUri.path);
    }
    return this.lldbDapOptions.createDapExecutableCommand(session, executable);
  }

  /**
   * Shows a message box when the debug adapter's path is not found
   */
  static async showLLDBDapNotFoundMessage(path: string) {
    const openSettingsAction = "Open Settings";
    const callbackValue = await vscode.window.showErrorMessage(
      `Debug adapter path: ${path} is not a valid file`,
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
