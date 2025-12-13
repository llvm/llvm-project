import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { AdbClient } from "./adb-client";
import Env from "./env";

/**
 * This class controls the execution of an Android APK for debugging.
 * It can install the lldb-server in the app sandbox, start the APK and the
 * lldb-server, dismiss the "waiting for debugger" dialog and stop everything.
 * It assumes the target device is connected (or the emulator is running), the
 * APK is installed, and the ADB daemon is running.
 */
export class ApkDebugSession {

    private runningSession: SessionInfo | undefined;

    /**
     * Component name is in the form "com.example.app/.MainActivity".
     * `wfd` stays for "waiting for debugger".
     */
    async start(deviceId: string | undefined, componentName: string, wfd: boolean) {
        const addId = componentName.split('/')[0];
        const adb = new AdbClient();
        if (deviceId !== undefined) {
            adb.setDeviceId(deviceId);
        } else {
            await adb.autoDetectDeviceId();
        }
        const arch = (await adb.shellCommandToString("uname -m")).trim();
        const lldbServerPath = await Env.getLldbServerPath(arch);
        if (!lldbServerPath) {
            throw new Error("Could not find LLDB server in Android NDK");
        }
        await this.stop();
        await this.cleanUpEarlierDebugSessions(adb, addId);
        await this.installLldbServer(adb, addId, lldbServerPath);
        const pid = await this.startApk(adb, componentName, wfd);

        const abortController = new AbortController();
        const endPromise = this.startLldbServer(adb, addId, abortController.signal);
        this.runningSession = {
            adb,
            addId,
            endPromise,
            abortController,
        };

        await this.waitLldbServerReachable(adb, addId);

        // In this early version, we save session data for debugging purposes.
        // But this is temporay and will be removed in future versions.
        // TODO: remove or put this data in a log.
        await this.saveSessionData();
    }

    async stop() {
        if (this.runningSession !== undefined) {
            try {
                await this.runningSession.adb.shellCommand(`am force-stop ${this.runningSession.addId}`);
            } catch {}
            this.runningSession.abortController.abort();
            try {
                await this.runningSession.endPromise;
            } catch {}
            this.runningSession = undefined;
        }
    }

    async dismissWaitingForDebuggerDialog() {
        if (this.runningSession !== undefined) {
            const pid = await this.runningSession.adb.getPid(this.runningSession.addId);
            await this.runningSession.adb.dismissWaitingForDebuggerDialog(pid);
        }
    }

    /**
     * This function tries to clean up anything left behind by earlier debug
     * sessions.
     * The current implementation is a bit aggressive, and could impact debug
     * sessions running in parallel on other apps.
     * However, the current debugging solution is not super stable and a deep
     * clean-up phase is really needed. At the same time, cases where two debug
     * sessions run in parallel are rare.
     */
    private async cleanUpEarlierDebugSessions(adb: AdbClient, addId: string) {
        const deviceId = adb.getDeviceId();

        // stop the app
        await adb.shellCommand(`am force-stop ${addId}`);

        // kill existing gdbserver processes
        await adb.shellCommand(`run-as ${addId} killall lldb-server`);

        // clean up port forwarding
        await this.cleanUpPortForwarding(adb, addId);

        // clean up unix-domain socket files in the app data folder
        await adb.shellCommand(`run-as ${addId} find /data/data/${addId} -name 'gdbserver.*' -exec rm {} \\;`);
    }

    private async cleanUpPortForwarding(adb: AdbClient, addId: string) {
        const deviceId = adb.getDeviceId();
        const list = await adb.getPortForwardingList();
        const filteredList = list.filter(item => {
            if (item.device !== deviceId) {
                return false;
            }
            const pattern1 = `localabstract:/${addId}`;
            const regex1 = new RegExp(pattern1);
            if (regex1.test(item.remotePort)) {
                return true;
            }
            const pattern2 = `localabstract:gdbserver.`;
            const regex2 = new RegExp(pattern2);
            if (regex2.test(item.remotePort)) {
                return true;
            }
            return false;
        });
        for (const item of filteredList) {
            const localPort = parseInt(item.localPort.replace(/^tcp:/, ""), 10);
            console.log(`Removing port forwarding for local port ${localPort} (remote: ${item.remotePort})`);
            await adb.removePortForwarding(localPort);
        }
    }

    private async installLldbServer(adb: AdbClient, addId: string, lldbServerPath: string) {
        await adb.shellCommand(`mkdir -p /data/local/tmp/lldb-stuff`);
        await adb.pushFile(lldbServerPath, `/data/local/tmp/lldb-stuff/lldb-server`);
        await adb.shellCommand(`run-as ${addId} mkdir -p /data/data/${addId}/lldb-stuff`);
        await adb.shellCommand(`cat /data/local/tmp/lldb-stuff/lldb-server | run-as ${addId} sh -c 'cat > /data/data/${addId}/lldb-stuff/lldb-server'`);
        await adb.shellCommand(`run-as ${addId} chmod 700 /data/data/${addId}/lldb-stuff/lldb-server`);
    }

    private startLldbServer(adb: AdbClient, addId: string, abort: AbortSignal) {
        const command = `run-as ${addId} /data/data/${addId}/lldb-stuff/lldb-server`
            + ` platform --server --listen unix-abstract:///${addId}/lldb-platform.sock`
            + ` --log-channels "lldb process:gdb-remote packets"`;
        // TODO: open log file
        const writer = async () => {};
        return adb.shellCommandToStream(command, writer, abort)
            .then(() => {
                // TODO: close log file
            });
    }

    private async waitLldbServerReachable(adb: AdbClient, addId: string) {
        const t1 = Date.now();
        for (;;) {
            const t2 = Date.now();
            if (t2 - t1 > 10000) {
                throw new Error("Timeout waiting for lldb-server to start");
            }
            const result = await adb.shellCommandToString(`cat /proc/net/unix | grep ${addId}/lldb-platform.sock`);
            if (result.trim().length > 0) {
                return;
            }
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    private async startApk(adb: AdbClient, componentName: string, wfd: boolean): Promise<number> {
        const parts = componentName.split('/');
        const addId = parts[0];
        if (parts.length === 1) {
            componentName = parts[0] + "/.MainActivity";
        }

        await adb.shellCommand(`am start -n ${componentName} -a android.intent.action.MAIN -c android.intent.category.LAUNCHER ${wfd ? '-D' : ''}`);

        const t1 = Date.now();
        for (;;) {
            const t2 = Date.now();
            if (t2 - t1 > 10000) {
                throw new Error("Timeout waiting for app to start");
            }
            try {
                const pid = await adb.getPid(addId);
                return pid;
            } catch {}
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    private async getDataFolder(): Promise<string | undefined> {
        const home = os.homedir();
        try {
            await fs.access(home, fs.constants.R_OK | fs.constants.W_OK);
            const dataFolder = path.join(home, ".lldb", "android");
            await fs.mkdir(dataFolder, { recursive: true });
            return dataFolder;
        } catch {}
        return undefined;
    }

    private async saveSessionData() {
        if (this.runningSession === undefined)
            return;
        const dir = await this.getDataFolder();
        if (dir === undefined)
            return;
        const pid = await this.runningSession.adb.getPid(this.runningSession.addId);
        fs.writeFile(path.join(dir, `last-env`),
            `ANDROID_DEVICE_ID=${this.runningSession.adb.getDeviceId()}\n` +
            `ANDROID_APP_ID=${this.runningSession.addId}\n` +
            `ANDROID_PID=${pid}\n`);
        fs.writeFile(path.join(dir, `last-start-commands`),
            `platform select remote-android\n` +
            `command alias a0 platform connect unix-abstract-connect:///${this.runningSession.addId}/lldb-platform.sock\n` +
            `command alias a1 process attach --pid ${pid}\n` +
            `command alias a1n process attach --name ${this.runningSession.addId}\n` +
            `command alias a2 process handle SIGSEGV -n false -p true -s false\n` +
            `command alias a3 process handle SIGBUS -n false -p true -s false\n` +
            `command alias a4 process handle SIGCHLD -n false -p true -s false\n` +
            `a0\n` +
            `a1\n` +
            `a2\n` +
            `a3\n` +
            `a4\n`
        );
    }
}

type SessionInfo = {
    adb: AdbClient;
    addId: string;
    endPromise: Promise<void>;
    abortController: AbortController;
}
