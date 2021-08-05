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

  /**
   *  Activate the MLIR context, and start the language client.
   */
  async activate(outputChannel: vscode.OutputChannel) {
    // Get the path of the mlir-lsp-server that is used to provide language
    // functionality.
    const userDefinedServerPath = config.get<string>('server_path');
    const serverPath = (userDefinedServerPath === '') ? 'mlir-lsp-server'
                                                      : userDefinedServerPath;

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
      documentSelector : [ {scheme : 'file', language : 'mlir'} ],
      synchronize : {
        // Notify the server about file changes to *.mlir files contained in the
        // workspace.
        fileEvents : vscode.workspace.createFileSystemWatcher('**/*.mlir')
      },
      outputChannel : outputChannel,
    };

    // Create the language client and start the client.
    this.client = new vscodelc.LanguageClient(
        'mlir-lsp', 'MLIR Language Client', serverOptions, clientOptions);
    this.subscriptions.push(this.client.start());

    // Watch for configuration changes.
    configWatcher.activate(this);
  }

  dispose() {
    this.subscriptions.forEach((d) => { d.dispose(); });
    this.subscriptions = [];
  }
}
