import * as vscode from "vscode";
import { ApkDebugSession } from "./apk-debug-session";

/**
 * Tracks an Android component associated with a VS Code debug session.
 * Normally, the component is an activity inside an APK.
 * This class includes everything needed to start and stop the activity
 * and lldb-server on the target device, and dismiss the "waiting for debugger"
 * dialog.
 * Here, we assume the target device is connected (or the emulator is running),
 * the APK is installed, and the ADB daemon is running.
 */
export class AndroidComponentTracker {

    private static catalog = new WeakMap<vscode.DebugSession, AndroidComponentTracker>();

    static getFromSession(session: vscode.DebugSession): AndroidComponentTracker | undefined {
        return AndroidComponentTracker.catalog.get(session);
    }

    readonly componentName: string;

    private apkDebugSession = new ApkDebugSession();

    /**
     * Component name is in the form "com.example.app/.MainActivity".
     * TODO: allow selecting the deviceId
     */
    constructor(session: vscode.DebugSession, componentName: string) {
        this.componentName = componentName;
        AndroidComponentTracker.catalog.set(session, this);
    }

    getAppId(): string {
        const parts = this.componentName.split('/');
        return parts[0];
    }

    getActivity(): string {
        const parts = this.componentName.split('/');
        if (parts.length === 1) {
            return parts[0] + ".MainActivity";
        }
        if (parts[1].startsWith('.')) {
            return parts[0] + parts[1];
        }
        return parts[1];
    }

    async startDebugSession() {
        // TODO: allow selecting the deviceId
        await this.apkDebugSession.start(undefined, this.componentName, true);
    }

    async stopDebugSession() {
        await this.apkDebugSession.stop();
    }

    async dismissWaitingForDebuggerDialog() {
        await this.apkDebugSession.dismissWaitingForDebuggerDialog();
    }
}
