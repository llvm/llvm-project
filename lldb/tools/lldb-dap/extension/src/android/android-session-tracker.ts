import * as vscode from "vscode";
import { ApkDebugEnv, ApkDebugSession } from "./apk-debug-session";

/**
 * This class is for tracking the Android APK debug session associated with the
 * VS Code debug session.
 * It includes everything needed to start and stop the Android activity
 * and lldb-server on the target device, and dismiss the "waiting for debugger"
 * dialog.
 * It assumes that the target device is connected (or the emulator is running),
 * the APK is installed, and the ADB daemon is running.
 */
export class AndroidSessionTracker {

    private static catalog = new WeakMap<vscode.DebugSession, AndroidSessionTracker>();

    static getFromSession(session: vscode.DebugSession): AndroidSessionTracker | undefined {
        return AndroidSessionTracker.catalog.get(session);
    }

    private apkDebugSession: ApkDebugSession;

    constructor(session: vscode.DebugSession) {
        const env = { lldbServerPath: session.configuration.androidLldbServerPath };
        const deviceSerial = session.configuration.androidDeviceSerial;
        const componentName = session.configuration.androidComponent;
        this.apkDebugSession = new ApkDebugSession(env, deviceSerial, componentName);
        AndroidSessionTracker.catalog.set(session, this);
    }

    async startDebugSession() {
        await this.apkDebugSession.start(true);
    }

    async stopDebugSession() {
        await this.apkDebugSession.stop();
    }

    async dismissWaitingForDebuggerDialog() {
        await this.apkDebugSession.dismissWaitingForDebuggerDialog();
    }
}
