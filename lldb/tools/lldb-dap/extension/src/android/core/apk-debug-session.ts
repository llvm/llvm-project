import { AdbClient } from "./adb-client";
import { Ndk } from "./ndk";

/**
 * This class controls the execution of an Android APK for debugging.
 * It can install the lldb-server in the app sandbox, start the APK and the
 * lldb-server, dismiss the "waiting for debugger" dialog and stop everything.
 * It assumes the target device is connected (or the emulator is running), the
 * APK is installed, and the ADB daemon is running.
 */
export class ApkDebugSession {
  private runningSession: SessionInfo | undefined;
  readonly env: ApkDebugEnv | undefined;
  readonly deviceSerial: string | undefined;
  readonly componentName: string;

  /**
   * Component name is in the form "com.example.app/.MainActivity".
   * If the env is undefined, a default one will be created when starting.
   * If the deviceSerial is undefined, we expect that only one device is connected.
   */
  constructor(
    env: ApkDebugEnv | undefined,
    deviceSerial: string | undefined,
    componentName: string,
  ) {
    this.env = env;
    this.deviceSerial = deviceSerial;
    this.componentName = componentName;
  }

  getLldbLaunchCommands(): string[] {
    let deviceSerial = this.deviceSerial;
    if (deviceSerial === undefined) {
      deviceSerial = "";
    }
    const appId = this.componentName.split("/")[0];
    return [
      `platform select remote-android`,
      `platform connect unix-abstract-connect://${deviceSerial}/${appId}/lldb-platform.sock`,
      `process attach --name ${appId}`,
      `process handle SIGSEGV -n false -p true -s false`,
      `process handle SIGBUS -n false -p true -s false`,
      `process handle SIGCHLD -n false -p true -s false`,
    ];
  }

  /**
   * Start the debug session.
   * `wfd` stays for "waiting for debugger".
   */
  async start(wfd: boolean) {
    const addId = this.componentName.split("/")[0];
    const adb = new AdbClient();
    if (this.deviceSerial !== undefined) {
      adb.setDeviceSerial(this.deviceSerial);
    } else {
      await adb.autoDetectDeviceSerial();
    }
    let env = this.env;
    if (!env) {
      const targetArch = (await adb.shellCommandToString("uname -m")).trim();
      env = await this.createDefaultEnv(targetArch);
    }
    await this.stop();
    await this.cleanUpEarlierDebugSessions(adb, addId);
    await this.installLldbServer(adb, addId, env.lldbServerPath);
    await this.startApk(adb, this.componentName, wfd);

    const abortController = new AbortController();
    const endPromise = this.startLldbServer(adb, addId, abortController.signal);
    this.runningSession = {
      adb,
      addId,
      endPromise,
      abortController,
    };

    await this.waitLldbServerReachable(adb, addId);
  }

  async stop() {
    if (this.runningSession !== undefined) {
      try {
        await this.runningSession.adb.shellCommand(
          `am force-stop ${this.runningSession.addId}`,
        );
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
      const pid = await this.runningSession.adb.getPid(
        this.runningSession.addId,
      );
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
    const deviceSerial = adb.getDeviceSerial();

    // stop the app
    await adb.shellCommand(`am force-stop ${addId}`);

    // kill existing gdbserver processes
    await adb.shellCommand(`run-as ${addId} killall lldb-server`);

    // clean up port forwarding
    await this.cleanUpPortForwarding(adb, addId);

    // clean up unix-domain socket files in the app data folder
    await adb.shellCommand(
      `run-as ${addId} find /data/data/${addId} -name 'gdbserver.*' -exec rm {} \\;`,
    );
  }

  private async cleanUpPortForwarding(adb: AdbClient, addId: string) {
    const deviceSerial = adb.getDeviceSerial();
    const list = await adb.getPortForwardingList();
    const filteredList = list.filter((item) => {
      if (item.device !== deviceSerial) {
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
      console.log(
        `Removing port forwarding for local port ${localPort} (remote: ${item.remotePort})`,
      );
      await adb.removePortForwarding(localPort);
    }
  }

  private async installLldbServer(
    adb: AdbClient,
    addId: string,
    lldbServerPath: string,
  ) {
    await adb.shellCommand(`mkdir -p /data/local/tmp/lldb-stuff`);
    await adb.pushFile(
      lldbServerPath,
      `/data/local/tmp/lldb-stuff/lldb-server`,
    );
    await adb.shellCommand(
      `run-as ${addId} mkdir -p /data/data/${addId}/lldb-stuff`,
    );
    await adb.shellCommand(
      `cat /data/local/tmp/lldb-stuff/lldb-server | run-as ${addId} sh -c 'cat > /data/data/${addId}/lldb-stuff/lldb-server'`,
    );
    await adb.shellCommand(
      `run-as ${addId} chmod 700 /data/data/${addId}/lldb-stuff/lldb-server`,
    );
  }

  private startLldbServer(adb: AdbClient, addId: string, abort: AbortSignal) {
    const command =
      `run-as ${addId} /data/data/${addId}/lldb-stuff/lldb-server` +
      ` platform --server --listen unix-abstract:///${addId}/lldb-platform.sock` +
      ` --log-channels "lldb process:gdb-remote packets"`;
    // TODO: open log file
    const writer = async () => {};
    return adb.shellCommandToStream(command, writer, abort).then(() => {
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
      const result = await adb.shellCommandToString(
        `cat /proc/net/unix | grep ${addId}/lldb-platform.sock`,
      );
      if (result.trim().length > 0) {
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  private async startApk(
    adb: AdbClient,
    componentName: string,
    wfd: boolean,
  ): Promise<number> {
    const parts = componentName.split("/");
    const addId = parts[0];
    if (parts.length === 1) {
      componentName = parts[0] + "/.MainActivity";
    }

    await adb.shellCommand(
      `am start -n ${componentName} -a android.intent.action.MAIN -c android.intent.category.LAUNCHER ${wfd ? "-D" : ""}`,
    );

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
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  private async createDefaultEnv(targetArch: string): Promise<ApkDebugEnv> {
    const ndkPath = await Ndk.getDefaultPath();
    if (!ndkPath) {
      throw new Error("NDK not found");
    }
    const ndkVersion = await Ndk.getVersion(ndkPath);
    if (!ndkVersion) {
      throw new Error(`Invalid NDK at path "${ndkPath}"`);
    }
    const lldbServerPath = await Ndk.getLldbServerPath(ndkPath, targetArch);
    if (!lldbServerPath) {
      throw new Error(
        `Could not find lldb-server in the NDK at path "${ndkPath}" for target architecture "${targetArch}"`,
      );
    }
    return {
      lldbServerPath,
    };
  }
}

export interface ApkDebugEnv {
  lldbServerPath: string; // path to the lldb-server executable on the host machine
}

interface SessionInfo {
  adb: AdbClient;
  addId: string;
  endPromise: Promise<void>;
  abortController: AbortController;
}
