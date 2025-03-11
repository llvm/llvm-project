import * as vscode from "vscode";

export class LaunchUriHandler implements vscode.UriHandler {
    async handleUri(uri: vscode.Uri) {
        try {
            const params = new URLSearchParams(uri.query);
            if (uri.path == '/start') {
                const configJson = params.get("config");
                if (configJson === null) {
                    throw new Error("Missing `config` URI parameter");
                }
                // Build the debug config.
                let debugConfig: vscode.DebugConfiguration = {
                    type: 'lldb-dap',
                    request: 'launch',
                    name: '',
                };
                Object.assign(debugConfig, JSON.parse(configJson));
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
