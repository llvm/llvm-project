import { ErrorWithNotification } from "../ui/error-with-notification";
import { ConfigureButton } from "../ui/show-error-message";
import { AdbClient } from "./adb-client";
import { ApkDebugSession } from "./apk-debug-session";

export class AndroidConfigurationBuilder {

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

    static getLldbLaunchCommands(deviceSerial: string | undefined, componentName: string): string[] {
        const apkDebugSession = new ApkDebugSession(deviceSerial, componentName);
        return apkDebugSession.getLldbLaunchCommands();
    }
}
