
export class ApkDebugConfiguration {

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
