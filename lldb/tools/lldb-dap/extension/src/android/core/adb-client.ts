import * as process from "node:process";
import { AdbConnection } from "./adb-connection";
import { Connection } from "./connection";
import Jdwp from "./jdwp";

/**
 * ADB (Android Debug Bridge) client.
 * An ADB client has access to multiple Android devices.
 * Many functionalities exposed by this class need the target device
 * to be designated before invoking that functionality.
 * The target device can be selected by invoking setDeviceSerial() or autoDetectDeviceSerial().
 * This client expects the ADB daemon to be running on the local machine.
 */
export class AdbClient {
  private deviceSerial: string | undefined = undefined;

  async getDeviceList(): Promise<string[]> {
    const connection = await this.createAdbConnection();
    try {
      const devices = await connection.getDeviceList();
      return devices;
    } finally {
      connection.close();
    }
  }

  async autoDetectDeviceSerial(): Promise<void> {
    const connection = await this.createAdbConnection();
    try {
      const devices = await connection.getDeviceList();
      if (devices.length === 1) {
        this.deviceSerial = devices[0];
        return;
      }
      if (devices.length === 0) {
        throw new Error("No connected Android devices found");
      }
      throw new Error(
        "Multiple connected Android devices found, please specify a device SN",
      );
    } finally {
      connection.close();
    }
  }

  setDeviceSerial(deviceSerial: string) {
    this.deviceSerial = deviceSerial;
  }

  getDeviceSerial(): string {
    if (this.deviceSerial === undefined) {
      throw new Error("Device SN is not set");
    }
    return this.deviceSerial;
  }

  async getDeviceName(): Promise<string> {
    let emulatorName = await this.shellCommandToString(
      "getprop ro.boot.qemu.avd_name",
    );
    emulatorName = emulatorName.trim();
    if (emulatorName) return emulatorName;
    let deviceName = await this.shellCommandToString(
      "settings get global device_name",
    );
    deviceName = deviceName.trim();
    if (deviceName) return deviceName;
    return this.getDeviceSerial();
  }

  async shellCommand(command: string): Promise<void> {
    const deviceSerial = this.getDeviceSerial();
    const connection = await this.createAdbConnection();
    try {
      await connection.setTargetDevice(deviceSerial);
      await connection.shellCommand(command);
    } finally {
      connection.close();
    }
  }

  async shellCommandToString(command: string): Promise<string> {
    const deviceSerial = this.getDeviceSerial();
    const connection = await this.createAdbConnection();
    try {
      await connection.setTargetDevice(deviceSerial);
      const output = await connection.shellCommandToString(command);
      return output;
    } finally {
      connection.close();
    }
  }

  async shellCommandToStream(
    command: string,
    writer: (data: Uint8Array) => Promise<void>,
    abort: AbortSignal,
  ): Promise<void> {
    const deviceSerial = this.getDeviceSerial();
    const connection = await this.createAdbConnection();
    abort.addEventListener("abort", () => {
      connection.close();
    });
    try {
      await connection.setTargetDevice(deviceSerial);
      await connection.shellCommandToStream(command, writer);
    } finally {
      connection.close();
    }
  }

  async getPid(packageName: string): Promise<number> {
    const deviceSerial = this.getDeviceSerial();
    const connection = await this.createAdbConnection();
    try {
      await connection.setTargetDevice(deviceSerial);
      const pid = await connection.getPid(packageName);
      return pid;
    } finally {
      connection.close();
    }
  }

  async addPortForwarding(
    remotePort: number | string,
    localPort: number = 0,
  ): Promise<number> {
    const deviceSerial = this.getDeviceSerial();
    const connection = await this.createAdbConnection();
    try {
      const port = await connection.addPortForwarding(
        deviceSerial,
        remotePort,
        localPort,
      );
      return port;
    } finally {
      connection.close();
    }
  }

  async removePortForwarding(localPort: number): Promise<void> {
    const deviceSerial = this.getDeviceSerial();
    const connection = await this.createAdbConnection();
    try {
      await connection.removePortForwarding(deviceSerial, localPort);
    } finally {
      connection.close();
    }
  }

  async getPortForwardingList(): Promise<
    { device: string; localPort: string; remotePort: string }[]
  > {
    const connection = await this.createAdbConnection();
    try {
      return await connection.getPortForwardingList();
    } finally {
      connection.close();
    }
  }

  async dismissWaitingForDebuggerDialog(pid: number): Promise<void> {
    const port = await this.addPortForwarding(`jdwp:${pid}`);
    let connection: Connection | undefined = undefined;
    const cleanup = async () => {
      try {
        connection?.close();
      } catch {}
      try {
        await this.removePortForwarding(port);
      } catch {}
    };
    try {
      connection = new Connection();
      await connection.connect("127.0.0.1", port);
      await Jdwp.handshake(connection);

      // ART is able to reply to handshake and DDM commands (command set 199)
      // without loading the JDWP agent.
      // By sending a version command, we force it to load the JDWP agent, which
      // causes the "waiting for debugger" popup to be dismissed.
      const version = await Jdwp.getVersion(connection);
      console.log("JDWP Version:", JSON.stringify(version));
    } catch (error) {
      await cleanup();
      throw error;
    }

    // When ART is in waiting-for-debugger mode, it actually runs in a loop,
    // polling for a debugger agent every 100ms. Therefore, the agent must run
    // for at least 100ms in order for the polling loop to detect it.
    // Once detected, the loop exits, the dialog is dismissed and the app
    // finally starts.
    // To keep the agent running, we have to keep the connection open. We do
    // that for 1 second, in background, i.e. without suspending this method.
    setTimeout(cleanup, 1000);
  }

  async pushData(data: Uint8Array, remoteFilePath: string): Promise<void> {
    const deviceSerial = this.getDeviceSerial();
    const connection = await this.createAdbConnection();
    try {
      await connection.setTargetDevice(deviceSerial);
      await connection.enterSyncMode();
      await connection.pushData(data, remoteFilePath);
    } finally {
      connection.close();
    }
  }

  async pushFile(localFilePath: string, remoteFilePath: string): Promise<void> {
    const deviceSerial = this.getDeviceSerial();
    const connection = await this.createAdbConnection();
    try {
      await connection.setTargetDevice(deviceSerial);
      await connection.enterSyncMode();
      await connection.pushFile(localFilePath, remoteFilePath);
    } finally {
      connection.close();
    }
  }

  private getAdbPort(): number {
    const envPort = process.env.ANDROID_ADB_SERVER_PORT;
    return envPort ? parseInt(envPort, 10) : 5037;
  }

  private async createAdbConnection(): Promise<AdbConnection> {
    const connection = new AdbConnection();
    await connection.connect("127.0.0.1", this.getAdbPort());
    return connection;
  }
}
