import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient';

import * as config from './config';
import * as configWatcher from './configWatcher';

/**
 *  This class represents the context of a specific workspace folder.
 */
class WorkspaceFolderContext {
  constructor(mlirServer: vscodelc.LanguageClient,
              pdllServer: vscodelc.LanguageClient) {
    this.mlirServer = mlirServer;
    this.pdllServer = pdllServer;
  }
  mlirServer!: vscodelc.LanguageClient;
  pdllServer!: vscodelc.LanguageClient;
}

/**
 *  This class manages all of the MLIR extension state,
 *  including the language client.
 */
export class MLIRContext implements vscode.Disposable {
  subscriptions: vscode.Disposable[] = [];
  workspaceFolders: WorkspaceFolderContext[] = [];

  /**
   *  Activate the MLIR context, and start the language clients.
   */
  async activate(outputChannel: vscode.OutputChannel,
                 warnOnEmptyServerPath: boolean) {
    // Start clients for each workspace folder.
    if (vscode.workspace.workspaceFolders &&
        vscode.workspace.workspaceFolders.length > 0) {
      for (const workspaceFolder of vscode.workspace.workspaceFolders) {
        this.workspaceFolders.push(await this.activateWorkspaceFolder(
            workspaceFolder, outputChannel, warnOnEmptyServerPath));
      }
    }
    this.workspaceFolders.push(await this.activateWorkspaceFolder(
        null, outputChannel, warnOnEmptyServerPath));
  }

  /**
   *  Activate the context for the given workspace folder, and start the
   *  language clients.
   */
  async activateWorkspaceFolder(workspaceFolder: vscode.WorkspaceFolder,
                                outputChannel: vscode.OutputChannel,
                                warnOnEmptyServerPath: boolean):
      Promise<WorkspaceFolderContext> {
    // Create the language clients for mlir and pdll.
    const [mlirServer, mlirServerPath] = await this.startLanguageClient(
        workspaceFolder, outputChannel, warnOnEmptyServerPath, 'server_path',
        'mlir');
    const [pdllServer, pdllServerPath] = await this.startLanguageClient(
        workspaceFolder, outputChannel, warnOnEmptyServerPath,
        'pdll_server_path', 'pdll');

    // Watch for configuration changes on this folder.
    const serverPathsToWatch = [ mlirServerPath, pdllServerPath ];
    await configWatcher.activate(this, workspaceFolder, serverPathsToWatch);

    return new WorkspaceFolderContext(mlirServer, pdllServer);
  }

  /**
   *  Start a new language client for the given language. Returns an array
   *  containing the opened server, or null if the server could not be started,
   *  and the resolved server path.
   */
  async startLanguageClient(workspaceFolder: vscode.WorkspaceFolder,
                            outputChannel: vscode.OutputChannel,
                            warnOnEmptyServerPath: boolean,
                            serverSettingName: string, languageName: string):
      Promise<[ vscodelc.LanguageClient, string ]> {
    const clientTitle = languageName.toUpperCase() + ' Language Client';

    // Get the path of the lsp-server that is used to provide language
    // functionality.
    var serverPath =
        await this.resolveServerPath(serverSettingName, workspaceFolder);

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
    this.workspaceFolders = [];
  }
}
