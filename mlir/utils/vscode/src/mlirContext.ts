import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient';

import * as config from './config';
import * as configWatcher from './configWatcher';

/**
 *  This class represents the context of a specific workspace folder.
 */
class WorkspaceFolderContext implements vscode.Disposable {
  dispose() {
    this.clients.forEach(client => client.stop());
    this.clients.clear();
  }

  clients: Map<string, vscodelc.LanguageClient> = new Map();
}

/**
 *  This class manages all of the MLIR extension state,
 *  including the language client.
 */
export class MLIRContext implements vscode.Disposable {
  subscriptions: vscode.Disposable[] = [];
  workspaceFolders: Map<string, WorkspaceFolderContext> = new Map();

  /**
   *  Activate the MLIR context, and start the language clients.
   */
  async activate(outputChannel: vscode.OutputChannel) {
    // This lambda is used to lazily start language clients for the given
    // document. It removes the need to pro-actively start language clients for
    // every folder within the workspace and every language type we provide.
    const startClientOnOpenDocument = async (document: vscode.TextDocument) => {
      if (document.uri.scheme !== 'file') {
        return;
      }
      let serverSettingName: string;
      if (document.languageId === 'mlir') {
        serverSettingName = 'server_path';
      } else if (document.languageId === 'pdll') {
        serverSettingName = 'pdll_server_path';
      } else {
        return;
      }

      // Resolve the workspace folder if this document is in one. We use the
      // workspace folder when determining if a server needs to be started.
      const uri = document.uri;
      let workspaceFolder = vscode.workspace.getWorkspaceFolder(uri);
      let workspaceFolderStr =
          workspaceFolder ? workspaceFolder.uri.toString() : "";

      // Get or create a client context for this folder.
      let folderContext = this.workspaceFolders.get(workspaceFolderStr);
      if (!folderContext) {
        folderContext = new WorkspaceFolderContext();
        this.workspaceFolders.set(workspaceFolderStr, folderContext);
      }
      // Start the client for this language if necessary.
      if (!folderContext.clients.has(document.languageId)) {
        let client = await this.activateWorkspaceFolder(
            workspaceFolder, serverSettingName, document.languageId,
            outputChannel);
        folderContext.clients.set(document.languageId, client);
      }
    };
    // Process any existing documents.
    vscode.workspace.textDocuments.forEach(startClientOnOpenDocument);

    // Watch any new documents to spawn servers when necessary.
    this.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument(startClientOnOpenDocument));
    this.subscriptions.push(
        vscode.workspace.onDidChangeWorkspaceFolders((event) => {
          for (const folder of event.removed) {
            const client = this.workspaceFolders.get(folder.uri.toString());
            if (client) {
              client.dispose();
              this.workspaceFolders.delete(folder.uri.toString());
            }
          }
        }));
  }

  /**
   *  Activate the language client for the given language in the given workspace
   *  folder.
   */
  async activateWorkspaceFolder(workspaceFolder: vscode.WorkspaceFolder,
                                serverSettingName: string, languageName: string,
                                outputChannel: vscode.OutputChannel):
      Promise<vscodelc.LanguageClient> {
    const [server, serverPath] = await this.startLanguageClient(
        workspaceFolder, outputChannel, serverSettingName, languageName);

    // Watch for configuration changes on this folder.
    await configWatcher.activate(this, workspaceFolder, serverSettingName,
                                 serverPath);
    return server;
  }

  /**
   *  Start a new language client for the given language. Returns an array
   *  containing the opened server, or null if the server could not be started,
   *  and the resolved server path.
   */
  async startLanguageClient(workspaceFolder: vscode.WorkspaceFolder,
                            outputChannel: vscode.OutputChannel,
                            serverSettingName: string, languageName: string):
      Promise<[ vscodelc.LanguageClient, string ]> {
    const clientTitle = languageName.toUpperCase() + ' Language Client';

    // Get the path of the lsp-server that is used to provide language
    // functionality.
    var serverPath =
        await this.resolveServerPath(serverSettingName, workspaceFolder);

    // If the server path is empty, bail. We don't emit errors if the user
    // hasn't explicitly configured the server.
    if (serverPath === '') {
      return [ null, serverPath ];
    }

    // Check that the file actually exists.
    if (!fs.existsSync(serverPath)) {
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

    // Configure file patterns relative to the workspace folder.
    let filePattern: vscode.GlobPattern = '**/*.' + languageName;
    let selectorPattern: string = null;
    if (workspaceFolder) {
      filePattern = new vscode.RelativePattern(workspaceFolder, filePattern);
      selectorPattern = `${workspaceFolder.uri.fsPath}/**/*`;
    }

    // Configure the middleware of the client. This is sort of abused to allow
    // for defining a "fallback" language server that operates on non-workspace
    // folders. Workspace folder language servers can properly filter out
    // documents not within the folder, but we can't effectively filter for
    // documents outside of the workspace. To support this, and avoid having two
    // servers targeting the same set of files, we use middleware to inject the
    // dynamic logic for checking if a document is in the workspace.
    let middleware = {};
    if (!workspaceFolder) {
      middleware = {
        didOpen : (document, next) => {
          if (!vscode.workspace.getWorkspaceFolder(document.uri)) {
            next(document);
          }
        }
      };
    }

    // Configure the client options.
    const clientOptions: vscodelc.LanguageClientOptions = {
      documentSelector : [
        {scheme : 'file', language : languageName, pattern : selectorPattern}
      ],
      synchronize : {
        // Notify the server about file changes to language files contained in
        // the workspace.
        fileEvents : vscode.workspace.createFileSystemWatcher(filePattern)
      },
      outputChannel : outputChannel,
      workspaceFolder : workspaceFolder,
      middleware : middleware
    };

    // Create the language client and start the client.
    let languageClient = new vscodelc.LanguageClient(
        languageName + '-lsp', clientTitle, serverOptions, clientOptions);
    languageClient.start();
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
   * Try to resolve the path for the given server setting, with an optional
   * workspace folder.
   */
  async resolveServerPath(serverSettingName: string,
                          workspaceFolder: vscode.WorkspaceFolder):
      Promise<string> {
    const configServerPath =
        config.get<string>(serverSettingName, workspaceFolder);
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
    let filePattern: vscode.GlobPattern = '**/' + serverPath;
    if (workspaceFolder) {
      filePattern = new vscode.RelativePattern(workspaceFolder, filePattern);
    }
    let foundUris = await vscode.workspace.findFiles(filePattern, null, 1);
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
    this.workspaceFolders.forEach((d) => { d.dispose(); });
    this.workspaceFolders.clear();
  }
}
