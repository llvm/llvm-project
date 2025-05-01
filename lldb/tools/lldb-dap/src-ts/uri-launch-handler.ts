import * as vscode from "vscode";

export class LaunchUriHandler implements vscode.UriHandler {
    async handleUri(uri: vscode.Uri) {
        try {
            const params = new URLSearchParams(uri.query);
            if (uri.path == '/start') {
                // Some properties have default values
                let debugConfig: vscode.DebugConfiguration = {
                    type: 'lldb-dap',
                    request: 'launch',
                    name: '',
                };
                // The `config` parameter allows providing a complete JSON-encoded configuration
                const configJson = params.get("config");
                if (configJson !== null) {
                    Object.assign(debugConfig, JSON.parse(configJson));
                }
                // Furthermore, some frequently used parameters can also be provided as separate parameters
                const stringKeys = ["name", "request", "program", "cwd", "debuggerRoot"];
                const numberKeys = ["pid"];
                const arrayKeys = [
                    "args", "initCommands", "preRunCommands", "stopCommands", "exitCommands",
                    "terminateCommands", "launchCommands", "attachCommands"
                ];
                for (const key of stringKeys) {
                    const value = params.get(key);
                    if (value) {
                        debugConfig[key] = value;
                    }
                }
                for (const key of numberKeys) {
                    const value = params.get(key);
                    if (value) {
                        debugConfig[key] = Number(value);
                    }
                }
                for (const key of arrayKeys) {
                    // `getAll()` returns an array of strings.
                    const value = params.getAll(key);
                    if (value) {
                        debugConfig[key] = value;
                    }
                }
                // Report an error if we received any unknown parameters
                const supportedKeys = new Set<string>(["config"].concat(stringKeys).concat(numberKeys).concat(arrayKeys));
                const presentKeys = new Set<string>(params.keys());
                // FIXME: Use `Set.difference` as soon as ES2024 is widely available
                const unknownKeys = new Set<string>();
                for (const k of presentKeys.keys()) {
                    if (!supportedKeys.has(k)) {
                        unknownKeys.add(k);
                    }
                }
                if (unknownKeys.size > 0) {
                    throw new Error(`Unsupported URL parameters: ${Array.from(unknownKeys.keys()).join(", ")}`);
                }
                // Prodide a default for the config name
                const defaultName = debugConfig.request == 'launch' ? "URL-based Launch" : "URL-based Attach";
                debugConfig.name = debugConfig.name || debugConfig.program || defaultName;
                // Force the type to `lldb-dap`. We don't want to allow launching any other
                // Debug Adapters using this URI scheme.
                if (debugConfig.type != "lldb-dap") {
                    throw new Error(`Unsupported debugger type: ${debugConfig.type}`);
                }
                await vscode.debug.startDebugging(undefined, debugConfig);
            } else {
                throw new Error(`Unsupported Uri path: ${uri.path}`);
            }
        } catch (err) {
            if (err instanceof Error) {
                await vscode.window.showErrorMessage(`Failed to handle lldb-dap URI request: ${err.message}`);
            } else {
                await vscode.window.showErrorMessage(`Failed to handle lldb-dap URI request: ${JSON.stringify(err)}`);
            }
        }
    }
}
