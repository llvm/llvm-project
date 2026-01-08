import { ErrorWithNotification } from "../ui/error-with-notification";
import { ConfigureButton, OpenSettingsButton } from "../ui/show-error-message";
import { AdbClient } from "./core/adb-client";
import { ApkDebugSession } from "./core/apk-debug-session";
import { Ndk } from "./core/ndk";

export class AndroidConfigurationBuilder {
  static async getDefaultNdkPath(): Promise<string> {
    const path = await Ndk.getDefaultPath();
    if (!path) {
      throw new ErrorWithNotification(
        `Unable to find the Android NDK. Please install it in its default location or define its path in the settings.`,
        new OpenSettingsButton("lldb-dap.androidNDKPath"),
      );
    }
    return path;
  }

  static async checkNdkAndRetrieveVersion(ndkPath: string): Promise<string> {
    const version = await Ndk.getVersion(ndkPath);
    if (!version) {
      throw new ErrorWithNotification(
        `Invalid Android NDK path "${ndkPath}". Please ensure the NDK is installed and the path is properly defined in the settings.`,
        new OpenSettingsButton("lldb-dap.androidNDKPath"),
      );
    }
    return version;
  }

  static async resolveDeviceSerial(device?: string): Promise<string> {
    const adbClient = new AdbClient();
    let deviceSerials: string[];
    try {
      deviceSerials = await adbClient.getDeviceList();
    } catch (e) {
      throw new ErrorWithNotification(
        `Could not connect to ADB server. Please make sure the ADB server is running and at least one device or emulator is connected.`,
      );
    }
    if (!device) {
      if (deviceSerials.length === 1) {
        return deviceSerials[0];
      }
      if (deviceSerials.length > 1) {
        throw new ErrorWithNotification(
          `Multiple connected Android devices found, please specify a device name or serial number in your launch configuration, property "androidDevice".`,
          new ConfigureButton(),
        );
      }
      throw new ErrorWithNotification(
        `No Android devices found. Please verify that at least one device or emulator is connected to the ADB server.`,
      );
    }
    for (const deviceSerial of deviceSerials) {
      if (deviceSerial === device) {
        return deviceSerial;
      }
    }
    for (const deviceSerial of deviceSerials) {
      const adbClient = new AdbClient();
      adbClient.setDeviceSerial(deviceSerial);
      const name = await adbClient.getDeviceName();
      if (name === device) {
        return deviceSerial;
      }
    }
    throw new ErrorWithNotification(
      `Android devices "${device}" not found. Please connect this device or emulator to the ADB server.`,
    );
  }

  /**
   * Returned arch can be: aarch64, riscv64, arm, x86_64, i386
   */
  static async getTargetArch(deviceSerial: string): Promise<string> {
    const adbClient = new AdbClient();
    adbClient.setDeviceSerial(deviceSerial);
    const arch = await adbClient.shellCommandToString("uname -m");
    return arch.trim();
  }

  static async getLldbServerPath(
    ndkPath: string,
    targetArch: string,
  ): Promise<string | undefined> {
    const path = await Ndk.getLldbServerPath(ndkPath, targetArch);
    if (!path) {
      throw new ErrorWithNotification(
        `Could not find lldb-server in the NDK at path "${ndkPath}" for target architecture "${targetArch}". Please verify that the NDK path is correct and that the NDK includes support for this architecture.`,
      );
    }
    return path;
  }

  static getLldbLaunchCommands(
    deviceSerial: string | undefined,
    componentName: string,
  ): string[] {
    // We create a temporary ApkDebugSession instance just to get the launch commands.
    const apkDebugSession = new ApkDebugSession(
      undefined,
      deviceSerial,
      componentName,
    );
    return apkDebugSession.getLldbLaunchCommands();
  }
}
