import * as vscode from "vscode";
import { AndroidConfigurationBuilder } from "./android-configuration-builder";
import { AndroidDebugSession } from "./android-debug-session";

/**
 * This class manages Android APK debugging.
 * It detects and resolves Android-specific debug configuration, and creates and
 * keeps track of any AndroidDebugSession associated with a VS Code debug session.
 */
export class AndroidPlatform {
  private static sessions = new WeakMap<
    vscode.DebugSession,
    AndroidDebugSession
  >();

  static async resolveDebugConfiguration(
    debugConfiguration: vscode.DebugConfiguration,
    logger: vscode.LogOutputChannel,
  ) {
    if (
      debugConfiguration.androidComponent &&
      debugConfiguration.request === "launch"
    ) {
      if (
        !debugConfiguration.launchCommands ||
        debugConfiguration.launchCommands.length === 0
      ) {
        if (!debugConfiguration.androidDeviceSerial) {
          debugConfiguration.androidDeviceSerial =
            await AndroidConfigurationBuilder.resolveDeviceSerial(
              debugConfiguration.androidDevice,
            );
        }
        logger.info(
          `Android device serial number: ${debugConfiguration.androidDeviceSerial}`,
        );
        if (!debugConfiguration.androidTargetArch) {
          debugConfiguration.androidTargetArch =
            await AndroidConfigurationBuilder.getTargetArch(
              debugConfiguration.androidDeviceSerial,
            );
        }
        logger.info(
          `Android target architecture: ${debugConfiguration.androidTargetArch}`,
        );
        if (!debugConfiguration.androidLldbServerPath) {
          if (!debugConfiguration.androidNDKPath) {
            debugConfiguration.androidNDKPath =
              await AndroidConfigurationBuilder.getDefaultNdkPath();
          }
          const ndkVersion =
            await AndroidConfigurationBuilder.checkNdkAndRetrieveVersion(
              debugConfiguration.androidNDKPath,
            );
          logger.info(`Android NDK path: ${debugConfiguration.androidNDKPath}`);
          logger.info(`Android NDK version: ${ndkVersion}`);
          debugConfiguration.androidLldbServerPath =
            await AndroidConfigurationBuilder.getLldbServerPath(
              debugConfiguration.androidNDKPath,
              debugConfiguration.androidTargetArch,
            );
        }
        debugConfiguration.launchCommands =
          AndroidConfigurationBuilder.getLldbLaunchCommands(
            debugConfiguration.androidDeviceSerial,
            debugConfiguration.androidComponent,
          );
      }
    }
  }

  static async createDebugSession(
    session: vscode.DebugSession,
  ): Promise<AndroidDebugSession | undefined> {
    if (
      session.configuration.androidComponent &&
      session.configuration.request === "launch"
    ) {
      const androidDebugSession = new AndroidDebugSession(session);
      AndroidPlatform.sessions.set(session, androidDebugSession);
      return androidDebugSession;
    }
    return undefined;
  }

  static getDebugSession(
    session: vscode.DebugSession,
  ): AndroidDebugSession | undefined {
    return AndroidPlatform.sessions.get(session);
  }
}
