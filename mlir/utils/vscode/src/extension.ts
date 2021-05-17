import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient';

let client: vscodelc.LanguageClient;

/**
 *  This method is called when the extension is activated. The extension is
 *  activated the very first time a command is executed.
 */
export function activate(context: vscode.ExtensionContext) {
  // Get the path of the mlir-lsp-server that is used to provide language
  // functionality.
  const config = vscode.workspace.getConfiguration('mlir');
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
    }
  };

  // Create the language client and start the client.
  client = new vscodelc.LanguageClient('mlir-lsp', 'MLIR Language Client',
                                       serverOptions, clientOptions);
  client.start();
}

/**
 *  This method is called when the extension is deactivated.
 */
export function deactivate(): Thenable<void>|undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}
