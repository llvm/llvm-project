import * as fs from 'fs';
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
    this.pdllClient = this.startLanguageClient(
        outputChannel, warnOnEmptyServerPath, 'pdll_server_path', 'pdll');
    this.client = this.startLanguageClient(outputChannel, warnOnEmptyServerPath,
                                           'server_path', 'mlir');

    // Watch for configuration changes.
    configWatcher.activate(this);
  }

  /**
   *  Start a new language client for the given language.
   */
  startLanguageClient(outputChannel: vscode.OutputChannel,
                      warnOnEmptyServerPath: boolean, serverSettingName: string,
                      languageName: string): vscodelc.LanguageClient {
    const clientTitle = languageName.toUpperCase() + ' Language Client';

    // Get the path of the lsp-server that is used to provide language
    // functionality.
    const serverPath = config.get<string>(serverSettingName);

    // If we aren't emitting warnings on an empty server path, and the server
    // path is empty, bail.
    if (!warnOnEmptyServerPath && serverPath === '') {
      return null;
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
      return null;
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
    return languageClient;
  }

  dispose() {
    this.subscriptions.forEach((d) => { d.dispose(); });
    this.subscriptions = [];
  }
}
