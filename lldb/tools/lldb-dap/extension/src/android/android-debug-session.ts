import * as vscode from "vscode";
import { ApkDebugSession } from "./core/apk-debug-session";

/**
 * This class represents an Android remote debug session.
 * It includes everything needed to start and stop the Android activity
 * and lldb-server on the target device, and dismiss the "waiting for debugger"
 * dialog.
 * It assumes that the target device is connected (or the emulator is running),
 * the APK is installed, and the ADB daemon is running.
 * An AndroidDebugSession is always associated with a VS Code debug session.
 * It's created and tracked by the AndroidPlatform class.
 */
export class AndroidDebugSession {
  private apkDebugSession: ApkDebugSession;

  constructor(session: vscode.DebugSession) {
    const env = { lldbServerPath: session.configuration.androidLldbServerPath };
    const deviceSerial = session.configuration.androidDeviceSerial;
    const componentName = session.configuration.androidComponent;
    this.apkDebugSession = new ApkDebugSession(
      env,
      deviceSerial,
      componentName,
    );
  }

  async start() {
    // TODO: Do we want some exceptions to be reported as ErrorWithNotification?
    await this.apkDebugSession.start(true);
  }

  async stop() {
    await this.apkDebugSession.stop();
  }

  async dismissWaitingForDebuggerDialog() {
    await this.apkDebugSession.dismissWaitingForDebuggerDialog();
  }
}
