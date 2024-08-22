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

  public static async validateDebugAdapterPath(pathUri: vscode.Uri) {
    try {
      const fileStats = await vscode.workspace.fs.stat(pathUri);
      if (!(fileStats.type & vscode.FileType.File)) {
        this.showErrorMessage(pathUri.path);
      }
    } catch (err) {
      this.showErrorMessage(pathUri.path);
    }
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
    const path: string = customPath ? customPath : executable!!.command;

    await LLDBDapDescriptorFactory.validateDebugAdapterPath(
      vscode.Uri.file(path),
    );
    return this.lldbDapOptions.createDapExecutableCommand(session, executable);
  }

  /**
   * Shows a message box when the debug adapter's path is not found
   */
  private static showErrorMessage(path: string) {
    const openSettingsAction = "Open Settings";
    vscode.window
      .showErrorMessage(
        `Debug adapter path: ${path} is not a valid file`,
        { modal: false },
        openSettingsAction,
      )
      .then((callBackValue) => {
        if (openSettingsAction === callBackValue) {
          vscode.commands.executeCommand(
            "workbench.action.openSettings",
            "lldb-dap.executable-path",
          );
        }
      });
  }
}
