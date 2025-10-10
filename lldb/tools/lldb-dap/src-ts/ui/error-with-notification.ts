import * as vscode from "vscode";
import {
  ConfigureButton,
  NotificationButton,
  showErrorMessage,
} from "./show-error-message";

/** Options used to configure {@link ErrorWithNotification.showNotification}. */
export interface ShowNotificationOptions extends vscode.MessageOptions {
  /**
   * Whether or not to show the configure launch configuration button.
   *
   * **IMPORTANT**: the configure launch configuration button will do nothing if the
   * callee isn't a {@link vscode.DebugConfigurationProvider}.
   */
  showConfigureButton?: boolean;
}

/**
 * An error that is able to be displayed to the user as a notification.
 *
 * Used in combination with {@link showErrorMessage showErrorMessage()} when whatever caused
 * the error was the result of a direct action by the user. E.g. launching a debug session.
 */
export class ErrorWithNotification extends Error {
  private readonly buttons: NotificationButton<any, null | undefined>[];

  constructor(
    message: string,
    ...buttons: NotificationButton<any, null | undefined>[]
  ) {
    super(message);
    this.buttons = buttons;
  }

  /**
   * Shows the notification to the user including the configure launch configuration button.
   *
   * **IMPORTANT**: the configure launch configuration button will do nothing if the
   * callee isn't a {@link vscode.DebugConfigurationProvider}.
   *
   * @param options Configure the behavior of the notification
   */
  showNotification(
    options: ShowNotificationOptions & { showConfigureButton: true },
  ): Promise<null | undefined>;

  /**
   * Shows the notification to the user.
   *
   * @param options Configure the behavior of the notification
   */
  showNotification(options?: ShowNotificationOptions): Promise<undefined>;

  // Actual implementation of showNotification()
  async showNotification(
    options: ShowNotificationOptions = {},
  ): Promise<null | undefined> {
    // Filter out the configure button unless explicitly requested
    let buttons = this.buttons;
    if (options.showConfigureButton !== true) {
      buttons = buttons.filter(
        (button) => !(button instanceof ConfigureButton),
      );
    }
    return showErrorMessage(this.message, options, ...buttons);
  }
}
