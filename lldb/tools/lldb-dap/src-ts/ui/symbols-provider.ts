import * as vscode from "vscode";
import { DebugProtocol } from "@vscode/debugprotocol";

import { DebugSessionTracker } from "../debug-session-tracker";
import { DisposableContext } from "../disposable-context";

export class SymbolsProvider extends DisposableContext {
  constructor(
    private readonly tracker: DebugSessionTracker,
    private readonly extensionContext: vscode.ExtensionContext,
  ) {
    super();

    this.pushSubscription(vscode.commands.registerCommand(
      "lldb-dap.modules.showSymbols",
      () => {
        this.SelectModuleAndShowSymbols();
      },
    ));
  }

  static async doesServerSupportSymbolsRequest(session: vscode.DebugSession): Promise<boolean> {
    try {
      const dummyArguments = { _dummy: true };
      const _result = await session.customRequest("dapGetModuleSymbols", dummyArguments);
      return true;
    } catch (_error) {
      return false;
    }
  }

  private async SelectModuleAndShowSymbols() {
    const session = vscode.debug.activeDebugSession;
    if (!session) {
        return;
    }

    if (!await SymbolsProvider.doesServerSupportSymbolsRequest(session)) {
        vscode.window.showErrorMessage("The debug adapter does not support symbol requests.");
        return;
    }

    const modules = this.tracker.debugSessionModules(session);
    if (!modules || modules.length === 0) {
        return;
    }

    // Let the user select a module to show symbols for
    const selectedModule = await vscode.window.showQuickPick(modules.map(m => new ModuleQuickPickItem(m)), {
        placeHolder: "Select a module to show symbols for"
    });
    if (!selectedModule) {
        return;
    }

    try {
      const symbols = await this.getSymbolsForModule(session, selectedModule.module.id.toString());
      this.showSymbolsForModule(selectedModule.module.name.toString(), symbols);
    } catch (error) {
      if (error instanceof Error) {
        vscode.window.showErrorMessage("Failed to retrieve symbols: " + error.message);
      } else {
        vscode.window.showErrorMessage("Failed to retrieve symbols due to an unknown error.");
      }
      
      return;
    }
  }

  private async getSymbolsForModule(session: vscode.DebugSession, moduleId: string): Promise<DAPSymbol[]> {
    console.log(`Getting symbols for module: ${moduleId}`);
    const symbols_response: { symbols: Array<DAPSymbolType> } = await session.customRequest("dapGetModuleSymbols", { moduleId });


    return symbols_response?.symbols.map(symbol => new DAPSymbol(
      symbol.userId,
      symbol.isDebug,
      symbol.isSynthetic,
      symbol.isExternal,
      symbol.type,
      symbol.fileAddress,
      symbol.loadAddress,
      symbol.size,
      symbol.name,
    )) || [];
  }

  private async showSymbolsForModule(moduleName: string, symbols: DAPSymbol[]) {
    const panel = vscode.window.createWebviewPanel(
      "lldb-dap.symbols",
      `Symbols for ${moduleName}`,
      vscode.ViewColumn.Active,
      {
        enableScripts: true,
        localResourceRoots: [
          this.getExtensionResourcePath()
        ]
      }
    );

    const tabulatorJsPath = panel.webview.asWebviewUri(vscode.Uri.joinPath(this.getExtensionResourcePath(), "tabulator.min.js"));
    const tabulatorCssPath = panel.webview.asWebviewUri(vscode.Uri.joinPath(this.getExtensionResourcePath(), "tabulator.min.css"));

    panel.webview.html = this.getHTMLContentForSymbols(tabulatorJsPath, tabulatorCssPath, symbols);
  }

  private getHTMLContentForSymbols(tabulatorJsPath: vscode.Uri, tabulatorCssPath: vscode.Uri, symbols: DAPSymbol[]): string {
    const dataJson = JSON.stringify(symbols);
    return `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <link href="${tabulatorCssPath}" rel="stylesheet">
    <style>
      body { font-family: sans-serif; margin: 0; padding: 0; }
      #table { height: 100vh; }
    </style>
</head>
<body>
    <div id="table"></div>
    <script src="${tabulatorJsPath}"></script>
    <script>
        const tableData = ${dataJson};
        new Tabulator("#table", {
            data: tableData,
            layout: "fitColumns",
            columns: [
                {title: "User ID", field: "userId", sorter: "number"},
                {title: "Debug", field: "isDebug", sorter: "boolean"},
                {title: "Synthetic", field: "isSynthetic", sorter: "boolean"},
                {title: "External", field: "isExternal", sorter: "boolean"},
                {title: "Type", field: "type", sorter: "string"},
                {title: "File Address", field: "fileAddress", sorter: "number"},
                {title: "Load Address", field: "loadAddress", sorter: "number"},
                {title: "Size", field: "size", sorter: "number"},
                {title: "Name", field: "name", sorter: "string"},
            ]
        });
    </script>
</body>
</html>`;
  }

  private getExtensionResourcePath(): vscode.Uri {
    return vscode.Uri.joinPath(this.extensionContext.extensionUri, "out");
  }
}

class ModuleQuickPickItem implements vscode.QuickPickItem {
  constructor(public readonly module: DebugProtocol.Module) {}

  get label(): string {
    return this.module.name;
  }

  get description(): string {
    return this.module.id.toString();
  }
}

/// The symbol type we get from the lldb-dap server
type DAPSymbolType = {
  userId: number;
  isDebug: boolean;
  isSynthetic: boolean;
  isExternal: boolean;
  type: string;
  fileAddress: number;
  loadAddress?: number;
  size: number;
  name: string;
};

class DAPSymbol {
  constructor(
    public readonly userId: number,
    public readonly isDebug: boolean,
    public readonly isSynthetic: boolean,
    public readonly isExternal: boolean,
    public readonly type: string,
    public readonly fileAddress: number,
    public readonly loadAddress: number | undefined,
    public readonly size: number,
    public readonly name: string,
  ) {}
}
