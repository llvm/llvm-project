import * as path from "node:path";
import * as fs from "node:fs/promises";
import * as os from "node:os";

/**
 * TODO: revisit everything!
 */
namespace Env {

    export async function getAndroidNdkPath(): Promise<string | undefined> {
        const home = os.homedir();
        const ndk = path.join(home, "Library", "Android", "sdk", "ndk");
        const entries = await fs.readdir(ndk);
        if (entries.length === 0) {
            return undefined;
        }
        entries.sort((a, b) => b.localeCompare(a, 'en-US', { numeric: true }));
        return path.join(ndk, entries[0]);
    }

    export async function getLldbServerPath(arch: string): Promise<string | undefined> {
        // supported arch: aarch64, riscv64, arm, x86_64, i386
        const ndk = await getAndroidNdkPath();
        if (ndk) {
            const root1 = path.join(ndk, "toolchains", "llvm", "prebuilt");
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
                                        if (entry3 === arch) {
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
        return undefined;
    }
}

export default Env;
