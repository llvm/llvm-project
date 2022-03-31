import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient';

import * as config from './config';
import * as configWatcher from './configWatcher';

/**
 *  This class manages all of the MLIR extension state,
 *  including the language client.
 */
export class MLIRContext implements vscode.Disposable {
  subscriptions: vscode.Disposable[] = [];
  client!: vscodelc.LanguageClient;
  pdllClient!: vscodelc.LanguageClient;

  /**
   *  Activate the MLIR context, and start the language clients.
   */
  async activate(outputChannel: vscode.OutputChannel,
                 warnOnEmptyServerPath: boolean) {
    // Create the language clients for mlir and pdll.
    let mlirServerPath: string, pdllServerPath: string;
    [this.client, mlirServerPath] = await this.startLanguageClient(
        outputChannel, warnOnEmptyServerPath, 'server_path', 'mlir');
    [this.pdllClient, pdllServerPath] = await this.startLanguageClient(
        outputChannel, warnOnEmptyServerPath, 'pdll_server_path', 'pdll');

    // Watch for configuration changes.
    const serverPathsToWatch = [ mlirServerPath, pdllServerPath ];
    await configWatcher.activate(this, serverPathsToWatch);
  }

  /**
   *  Start a new language client for the given language. Returns an array
   *  containing the opened server, or null if the server could not be started,
   *  and the resolved server path.
   */
  async startLanguageClient(outputChannel: vscode.OutputChannel,
                            warnOnEmptyServerPath: boolean,
                            serverSettingName: string, languageName: string):
      Promise<[ vscodelc.LanguageClient, string ]> {
    const clientTitle = languageName.toUpperCase() + ' Language Client';

    // Get the path of the lsp-server that is used to provide language
    // functionality.
    var serverPath = await this.resolveServerPath(serverSettingName);

    // If we aren't emitting warnings on an empty server path, and the server
    // path is empty, bail.
    if (!warnOnEmptyServerPath && serverPath === '') {
      return [ null, serverPath ];
    }

    // Check that the file actually exists.
    if (serverPath === '' || !fs.existsSync(serverPath)) {
      vscode.window
          .showErrorMessage(
              `${clientTitle}: Unable to resolve path for '${
                  serverSettingName}', please ensure the path is correct`,
              "Open Setting")
          .then((value) => {
            if (value === "Open Setting") {
              vscode.commands.executeCommand(
                  'workbench.action.openWorkspaceSettings',
                  {openToSide : false, query : `mlir.${serverSettingName}`});
            }
          });
      return [ null, serverPath ];
    }

    // Configure the server options.
    const serverOptions: vscodelc.ServerOptions = {
      run : {
        command : serverPath,
        transport : vscodelc.TransportKind.stdio,
        args : []
      },
      debug : {
        command : serverPath,
        transport : vscodelc.TransportKind.stdio,
        args : []
      }
    };

    // Configure the client options.
    const clientOptions: vscodelc.LanguageClientOptions = {
      documentSelector : [ {scheme : 'file', language : languageName} ],
      synchronize : {
        // Notify the server about file changes to language files contained in
        // the workspace.
        fileEvents :
            vscode.workspace.createFileSystemWatcher('**/*.' + languageName)
      },
      outputChannel : outputChannel,
    };

    // Create the language client and start the client.
    let languageClient = new vscodelc.LanguageClient(
        languageName + '-lsp', clientTitle, serverOptions, clientOptions);
    this.subscriptions.push(languageClient.start());
    return [ languageClient, serverPath ];
  }

  /**
   * Given a server setting, return the default server path.
   */
  static getDefaultServerFilename(serverSettingName: string): string {
    if (serverSettingName === 'pdll_server_path') {
      return 'mlir-pdll-lsp-server';
    }
    if (serverSettingName === 'server_path') {
      return 'mlir-lsp-server';
    }
    return '';
  }

  /**
   * Try to resolve the path for the given server setting.
   */
  async resolveServerPath(serverSettingName: string): Promise<string> {
    let configServerPath = config.get<string>(serverSettingName);
    let serverPath = configServerPath;

    // If the path is already fully resolved, there is nothing to do.
    if (path.isAbsolute(serverPath)) {
      return serverPath;
    }

    // If a path hasn't been set, try to use the default path.
    if (serverPath === '') {
      serverPath = MLIRContext.getDefaultServerFilename(serverSettingName);
      if (serverPath === '') {
        return serverPath;
      }
      // Fallthrough to try resolving the default path.
    }

    // Try to resolve the path relative to the workspace.
    const foundUris: vscode.Uri[] =
        await vscode.workspace.findFiles('**/' + serverPath, null, 1);
    if (foundUris.length === 0) {
      // If we couldn't resolve it, just return the current configuration path
      // anyways. The file might not exist yet.
      return configServerPath;
    }
    // Otherwise, return the resolved path.
    return foundUris[0].fsPath;
  }

  dispose() {
    this.subscriptions.forEach((d) => { d.dispose(); });
    this.subscriptions = [];
  }
}
