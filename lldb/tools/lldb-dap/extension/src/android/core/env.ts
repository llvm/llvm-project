import * as path from "node:path";
import * as fs from "node:fs/promises";
import * as os from "node:os";

/**
 * TODO: probably not useful anymore
 * @deprecated
 */
namespace Env {
    async function getDataFolder(): Promise<string | undefined> {
        const home = os.homedir();
        try {
            await fs.access(home, fs.constants.R_OK | fs.constants.W_OK);
            const dataFolder = path.join(home, ".lldb", "android");
            await fs.mkdir(dataFolder, { recursive: true });
            return dataFolder;
        } catch {}
        return undefined;
    }
}

export default Env;
