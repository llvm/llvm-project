import * as vscode from "vscode";

/**
 * Converts the given value to an integer if it isn't already.
 *
 * If the value cannot be converted then this function will return undefined.
 *
 * @param value the value to to be converted
 * @returns the integer value or undefined if unable to convert
 */
function convertToInteger(value: any): number | undefined {
  let result: number | undefined;
  switch (typeof value) {
    case "number":
      result = value;
      break;
    case "string":
      result = Number(value);
      break;
    default:
      return undefined;
  }
  if (!Number.isInteger(result)) {
    return undefined;
  }
  return result;
}

/**
 * A {@link vscode.DebugConfigurationProvider} used to resolve LLDB DAP debug configurations.
 *
 * Performs checks on the debug configuration before launching a debug session.
 */
export class LLDBDapConfigurationProvider
  implements vscode.DebugConfigurationProvider
{
  resolveDebugConfigurationWithSubstitutedVariables(
    _folder: vscode.WorkspaceFolder | undefined,
    debugConfiguration: vscode.DebugConfiguration,
  ): vscode.ProviderResult<vscode.DebugConfiguration> {
    // Convert the "pid" option to a number if it is a string
    if ("pid" in debugConfiguration) {
      const pid = convertToInteger(debugConfiguration.pid);
      if (pid === undefined) {
        vscode.window.showErrorMessage(
          "Invalid debug configuration: property 'pid' must either be an integer or a string containing an integer value.",
        );
        return null;
      }
      debugConfiguration.pid = pid;
    }
    return debugConfiguration;
  }
}
