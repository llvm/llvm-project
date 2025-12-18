import { ErrorWithNotification } from "../ui/error-with-notification";
import { ConfigureButton } from "../ui/show-error-message";
import { AdbClient } from "./adb-client";

export class AndroidConfigurationBuilder {

    static async resolveDeviceSerial(device?: string): Promise<string> {
        const adbClient = new AdbClient();
        const deviceSerials = await adbClient.getDeviceList();
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
                `No connected Android devices found.`,
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
            `Android devices "${device}" not found, please connect this device.`,
            new ConfigureButton(),
        );
    }

    static getLldbLaunchCommands(deviceSerial: string | undefined, componentName: string): string[] {
        if (deviceSerial === undefined) {
            deviceSerial = "";
        }
        const appId = componentName.split("/")[0];
        return [
            `platform select remote-android`,
            `platform connect unix-abstract-connect://${deviceSerial}/${appId}/lldb-platform.sock`,
            `process attach --name ${appId}`,
            `process handle SIGSEGV -n false -p true -s false`,
            `process handle SIGBUS -n false -p true -s false`,
            `process handle SIGCHLD -n false -p true -s false`,
        ];
    }
}
