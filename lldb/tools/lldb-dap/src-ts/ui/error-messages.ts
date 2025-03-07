import * as vscode from "vscode";

/**
 * Shows a modal when the debug adapter's path is not found
 */
export async function showLLDBDapNotFoundMessage(path?: string) {
  const message =
    path !== undefined
      ? `Debug adapter path: ${path} is not a valid file`
      : "Unable to find the path to the LLDB debug adapter executable.";
  const openSettingsAction = "Open Settings";
  const callbackValue = await vscode.window.showErrorMessage(
    message,
    { modal: true },
    openSettingsAction,
  );

  if (openSettingsAction === callbackValue) {
    vscode.commands.executeCommand(
      "workbench.action.openSettings",
      "lldb-dap.executable-path",
    );
  }
}

/**
 * Shows an error message to the user that optionally allows them to open their
 * launch.json to configure it.
 *
 * Expected to be used in the context of a {@link vscode.DebugConfigurationProvider}.
 *
 * @param message The error message to display to the user
 * @returns `undefined` if the debug session should stop or `null` if the launch.json should be opened
 */
export async function showErrorWithConfigureButton(
  message: string,
): Promise<null | undefined> {
  const userSelection = await vscode.window.showErrorMessage(
    message,
    { modal: true },
    "Configure",
  );

  if (userSelection === "Configure") {
    return null; // Stops the debug session and opens the launch.json for editing
  }

  return undefined; // Only stops the debug session
}
