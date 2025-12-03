import * as vscode from 'vscode';
import * as path from 'path';
import { Command } from './command';
import { LLVMContext } from './llvmContext';
import {
  LlvmGetCfg,
  LlvmBbLocation,
} from './lspCustomMessages';

export class LLVMGetCfgCommand extends Command {

  constructor(context: LLVMContext) {
    super('llvm.cfg', context);
  }

  async execute(...args: any[]) {
    // Only works when there is an active open editor with a .ll file
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
      vscode.window.showInformationMessage('No active text editor.');
      return;
    }
    const currentFileLanguageId = activeEditor.document.languageId;
    if (currentFileLanguageId !== 'llvm') {
      vscode.window.showInformationMessage('Only supported for language `llvm\'.');
      return;
    }
    const currentFileUri = activeEditor.document.uri;
    const client = await this.context.getOrActivateLanguageClient(currentFileUri, currentFileLanguageId);
    if (!client || !client.initializeResult) {
      vscode.window.showErrorMessage('Language server is not yet ready.');
      return;
    }

    // Ask lsp server
    // TODO: should we send uri.fspath instead? what's the system-agnostic way to pass uri?
    //   In general, I think we should use vscode.Uri everywhere instead of strings (the context maps etc.)
    let result: LlvmGetCfg.Response = undefined;
    try {
      const params: LlvmGetCfg.Params = {
        uri: currentFileUri.toString(),
        position: activeEditor.selection.active,
      };
      const response = await client.sendRequest(LlvmGetCfg.Type, params);
      // TODO: should check if the IDs match??
      if (response['error'] !== undefined) {
        this.context.outputChannel.appendLine(`Error during custom request LlvmGetCfg: server returned error`);
        return;
      }
      result = response;
    } catch (error) {
      this.context.outputChannel.appendLine(`Error during custom request LlvmGetCfg: ${error}`);
      return;
    }

    // Read the cfg from the server's response
    const cfgFileUri = vscode.Uri.parse(result['uri']);
    const cfgFilePath = cfgFileUri.fsPath;
    const cfgDir = path.dirname(cfgFilePath);
    let targetFileContent: string;
    try {
      const targetUri = vscode.Uri.file(cfgFilePath);
      const fileBytes = await vscode.workspace.fs.readFile(targetUri);
      targetFileContent = Buffer.from(fileBytes).toString('utf8');
    } catch (error) {
      this.context.outputChannel.appendLine(`Could not read file: ${cfgFilePath}. Error: ${error}`);
      return;
    }

    const workspaceFolder = vscode.workspace.getWorkspaceFolder(currentFileUri);
    let workspaceFolderStr =
      workspaceFolder ? workspaceFolder.uri.toString() : "";
    const folderContext = this.context.workspaceFolders.get(workspaceFolderStr);

    let newPanelCreated = false;

    // Get saved webview panel that has the desired CFG or create a new one
    const panel = (folderContext.cfgWebViews.has(currentFileUri.fsPath) && folderContext.cfgWebViews.get(currentFileUri.fsPath).has(result['function'])) ?
      folderContext.cfgWebViews.get(currentFileUri.fsPath).get(result['function']) :

      // Create the webview panel and show the svg in it
      await (async () => {
        const panel = vscode.window.createWebviewPanel(
          'embeddedView',
          `CFG for ${result['function']} from ${path.basename(currentFileUri.fsPath)}`,
          vscode.ViewColumn.Beside,
          {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(cfgDir)]
          }
        );
        panel.webview.html = await getWebviewContentWithInteraction(
          this.context,
          {
            svgContent: targetFileContent,
            fileName: cfgFilePath
          });
        newPanelCreated = true;
        return panel;
      })();
    if (!folderContext.cfgWebViews.has(currentFileUri.fsPath)) {
      folderContext.cfgWebViews.set(currentFileUri.fsPath, new Map())
    }
    folderContext.cfgWebViews.get(currentFileUri.fsPath).set(result['function'], panel);

    // When panel is closed delete it from the map
    panel.onDidDispose(() => folderContext.cfgWebViews.delete(currentFileUri.fsPath))

    // Focus on the panel
    panel.reveal(panel.viewColumn);

    // Send message to center on node to webview
    const nodeToCenter = result['node_id'];
    this.context.outputChannel.appendLine(`Node To Center: ID = ${nodeToCenter}`);
    panel.webview.postMessage({ command: "centerOn", node: nodeToCenter });

    // Handle messages from the webview
    if (newPanelCreated) {
      this.context.subscriptions.push(
        panel.webview.onDidReceiveMessage(
          async message => {
            switch (message.command) {
              case 'cfgViewerDebug': {
                this.context.outputChannel.appendLine(message.msg);
                return;
              }
              case 'svgElementClicked':
                const elementId = message.elementId;
                this.context.outputChannel.appendLine(`SVG Element Clicked: ID = ${elementId}`);

                let result: LlvmBbLocation.Response = undefined;
                try {
                  const params: LlvmBbLocation.Params = {
                    uri: cfgFileUri.toString(),
                    node_id: elementId,
                  };
                  const response = await client.sendRequest(LlvmBbLocation.Type, params);
                  // TODO: should check if the IDs match??
                  if (response['error'] !== undefined) {
                    this.context.outputChannel.appendLine(`Error during custom request LlvmBbLocation: server returned error`);
                    return;
                  }
                  result = response;
                } catch (error) {
                  this.context.outputChannel.appendLine(`Error during custom request LlvmGetCfg: ${error}`);
                  return;
                }

                const targetUri = vscode.Uri.parse(result['uri']);
                // can I have just this since we send the right shape?
                // const selection = result['range'];
                const startCol = 0;
                const endCol = Math.max(0, result['range']['end']['character']);
                const startLine = Math.max(0, result['range']['start']['line']);
                // hack since the bb end is marked as the line with the following one
                const endLine = Math.max(startLine, Math.max(0, result['range']['end']['line']) - (endCol == 0 ? 1 : 0));
                const targetEditor = vscode.window.visibleTextEditors.find(editor => {
                  return editor.document.uri.toString() === targetUri.toString();
                });
                if (targetEditor) {
                  const selection = new vscode.Range(
                    new vscode.Position(startLine, startCol),
                    targetEditor.document.lineAt(endLine).range.end);
                  await vscode.window.showTextDocument(targetEditor.document, {
                    viewColumn: targetEditor.viewColumn,
                    selection: selection,
                    preserveFocus: false,
                  });
                  targetEditor.revealRange(selection, vscode.TextEditorRevealType.InCenter);
                } else {
                  const document = await vscode.workspace.openTextDocument(targetUri);
                  const selection = new vscode.Range(
                    new vscode.Position(startLine, startCol),
                    document.lineAt(endLine).range.end);
                  await vscode.window.showTextDocument(document, {
                    viewColumn: findTabGroupColumn(targetUri, vscode.ViewColumn.Beside),
                    selection: selection,
                    preserveFocus: false,
                    preview: false,
                  });
                }

                this.context.outputChannel.appendLine(`Navigated to: ${targetUri.fsPath}`);
                return;
            }
          },
          undefined,
          this.context.subscriptions
        )
      );
    }
  }
}

/**
 * Find column of open document or fallback
 *
 * @param uri URI of file to open
 * @param column Fallback column
 * @returns Column with open editor of `uri`
 */
function findTabGroupColumn(uri: vscode.Uri, column: vscode.ViewColumn): vscode.ViewColumn {
  if (vscode.window.tabGroups.all.length === 1) {
    return column;
  }

  for (const tab of vscode.window.tabGroups.activeTabGroup.tabs) {
    if (isTabOfUri(tab, uri)) {
      return tab.group.viewColumn;
    }
  }

  for (const tabGroup of vscode.window.tabGroups.all) {
    if (tabGroup.viewColumn === column)
      continue;

    for (const tab of tabGroup.tabs) {
      if (isTabOfUri(tab, uri)) {
        return tab.group.viewColumn;
      }
    }
  }

  return column;
}

function isTabOfUri(tab: vscode.Tab, uri: vscode.Uri): boolean {
  return tab.input instanceof vscode.TabInputText &&
    tab.input.uri.fsPath.toLocaleLowerCase() === uri.fsPath.toLocaleLowerCase()
}

async function getWebviewContentWithInteraction(context: LLVMContext, data: Record<string, string>) {
  const filePath = path.join(context.context.extensionPath, 'templates', 'cfgViewer.html');
  let templateBytes = await vscode.workspace.fs.readFile(vscode.Uri.file(filePath));
  let targetFileContent = Buffer.from(templateBytes).toString('utf8');

  // TODO: safety!
  for (const [key, value] of Object.entries(data)) {
    const placeholder = new RegExp(`\\$\\{${key}\\}`, 'g');
    targetFileContent = targetFileContent.replace(placeholder, value);
  }

  context.outputChannel.appendLine(`--- WEBVIEW SOURCE ---\n${targetFileContent}`);

  return targetFileContent;
}
