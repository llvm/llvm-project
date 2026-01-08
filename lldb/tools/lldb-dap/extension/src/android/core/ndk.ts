import * as path from "node:path";
import * as fs from "node:fs/promises";
import * as os from "node:os";

/**
 * Gives access to elements of the Android NDK.
 */
export class Ndk {

    static async getDefaultPath(): Promise<string | undefined> {
        const home = os.homedir();
        const ndk = path.join(home, "Library", "Android", "sdk", "ndk");
        let entries: string[] = [];
        try {
            entries = await fs.readdir(ndk);
        } catch {}
        if (entries.length === 0) {
            return undefined;
        }
        entries.sort((a, b) => b.localeCompare(a, 'en-US', { numeric: true }));
        return path.join(ndk, entries[0]);
    }

    static async getVersion(ndkPath: string): Promise<string | undefined> {
        const sourcePropsPath = path.join(ndkPath, "source.properties");
        try {
            const content = await fs.readFile(sourcePropsPath, { encoding: "utf-8" });
            const lines = content.split("\n");
            for (const line of lines) {
                const match = line.match(/^Pkg.Revision\s*=\s*(.+)$/);
                if (match) {
                    return match[1].trim();
                }
            }
        } catch {}
    }

    static async getLldbServerPath(ndkPath: string, targetArch: string): Promise<string | undefined> {
        const root1 = path.join(ndkPath, "toolchains", "llvm", "prebuilt");
        try {
            const entries1 = await fs.readdir(root1);
            for (const entry1 of entries1) {
                if (entry1.startsWith("darwin-")) {
                    const root2 = path.join(root1, entry1, "lib", "clang");
                    try {
                        const entries2 = await fs.readdir(root2);
                        for (const entry2 of entries2) {
                            const root3 = path.join(root2, entry2, "lib", "linux");
                            try {
                                const entries3 = await fs.readdir(root3);
                                for (const entry3 of entries3) {
                                    if (entry3 === targetArch) {
                                        const candidate = path.join(root3, entry3, "lldb-server");
                                        try {
                                            await fs.access(candidate, fs.constants.R_OK);
                                            return candidate;
                                        } catch {}
                                    }
                                }
                            } catch {}
                        }
                    } catch {}
                }
            }
        } catch {}
    }
}
