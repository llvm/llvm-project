import * as vscode from "vscode";

/**
 * A button with a particular label that can perform an action when clicked.
 *
 * Used to add buttons to {@link showErrorMessage showErrorMessage()}.
 */
export interface NotificationButton<T, Result> {
  readonly label: T;
  action(): Promise<Result>;
}

/**
 * Represents a button that, when clicked, will open a particular VS Code setting.
 */
export class OpenSettingsButton
  implements NotificationButton<"Open Settings", undefined>
{
  readonly label = "Open Settings";

  constructor(private readonly settingId?: string) {}

  async action(): Promise<undefined> {
    await vscode.commands.executeCommand(
      "workbench.action.openSettings",
      this.settingId ?? "@ext:llvm-vs-code-extensions.lldb-dap ",
    );
  }
}

/**
 * Represents a button that, when clicked, will return `null`.
 *
 * Used by a {@link vscode.DebugConfigurationProvider} to indicate that VS Code should
 * cancel a debug session and open its launch configuration.
 *
 * **IMPORTANT**: this button will do nothing if the callee isn't a
 * {@link vscode.DebugConfigurationProvider}.
 */
export class ConfigureButton
  implements NotificationButton<"Configure", null | undefined>
{
  readonly label = "Configure";

  async action(): Promise<null | undefined> {
    return null; // Opens the launch.json if returned from a DebugConfigurationProvider
  }
}

/** Gets the Result type from a {@link NotificationButton} or string value. */
type ResultOf<T> = T extends string
  ? T
  : T extends NotificationButton<any, infer Result>
    ? Result
    : never;

/**
 * Shows an error message to the user with an optional array of buttons.
 *
 * This can be used with common buttons such as {@link OpenSettingsButton} or plain
 * strings as would normally be accepted by {@link vscode.window.showErrorMessage}.
 *
 * @param message The error message to display to the user
 * @param options Configures the behaviour of the message.
 * @param buttons An array of {@link NotificationButton buttons} or strings that the user can click on
 * @returns `undefined` or the result of a button's action
 */
export async function showErrorMessage<
  T extends string | NotificationButton<any, any>,
>(
  message: string,
  options: vscode.MessageOptions = {},
  ...buttons: T[]
): Promise<ResultOf<T> | undefined> {
  const userSelection = await vscode.window.showErrorMessage(
    message,
    options,
    ...buttons.map((button) => {
      if (typeof button === "string") {
        return button;
      }
      return button.label;
    }),
  );

  for (const button of buttons) {
    if (typeof button === "string") {
      if (userSelection === button) {
        // Type assertion is required to let TypeScript know that "button" isn't just any old string.
        return button as ResultOf<T>;
      }
    } else if (userSelection === button.label) {
      return await button.action();
    }
  }

  return undefined;
}
