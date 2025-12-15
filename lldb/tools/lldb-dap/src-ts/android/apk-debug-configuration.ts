import * as os from "node:os";

export class ApkDebugConfiguration {

    static getLldbLaunchCommands(deviceId: string | undefined, componentName: string): string[] {
        // TODO: return the real commands needed to connect to lldb-server; last-start-commands should not be used here
        // TODO: manage deviceId
        const home = os.homedir();
        return [
            `command source -s 0 -e 0 '${home}/.lldb/android/last-start-commands'`
        ];
    }
}
