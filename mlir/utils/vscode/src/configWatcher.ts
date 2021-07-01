import * as chokidar from 'chokidar';
import * as path from 'path';
import * as vscode from 'vscode';

import * as config from './config';
import {MLIRContext} from './mlirContext';

/**
 *  Prompt the user to see if we should restart the server.
 */
async function promptRestart(settingName: string, promptMessage: string) {
  switch (config.get<string>(settingName)) {
  case 'restart':
    vscode.commands.executeCommand('mlir.restart');
    break;
  case 'ignore':
    break;
  case 'prompt':
  default:
    switch (await vscode.window.showInformationMessage(
        promptMessage, 'Yes', 'Yes, always', 'No, never')) {
    case 'Yes':
      vscode.commands.executeCommand('mlir.restart');
      break;
    case 'Yes, always':
      vscode.commands.executeCommand('mlir.restart');
      config.update<string>(settingName, 'restart',
                            vscode.ConfigurationTarget.Global);
      break;
    case 'No, never':
      config.update<string>(settingName, 'ignore',
                            vscode.ConfigurationTarget.Global);
      break;
    default:
      break;
    }
    break;
  }
}

/**
 *  Activate the watchers that track configuration changes which decide when to
 *  restart the server.
 */
export function activate(mlirContext: MLIRContext) {
  // When a configuration change happens, check to see if we should restart the
  // server.
  mlirContext.subscriptions.push(vscode.workspace.onDidChangeConfiguration(event => {
    const settings: string[] = [ 'server_path' ];
    for (const setting of settings) {
      const expandedSetting = `mlir.${setting}`;
      if (event.affectsConfiguration(expandedSetting)) {
        promptRestart(
            'onSettingsChanged',
            `setting '${
                expandedSetting}' has changed. Do you want to reload the server?`);
        break;
      }
    }
  }));

  // Track the server file in case it changes. We use `fs` here because the
  // server may not be in a workspace directory.
  const userDefinedServerPath = config.get<string>('server_path');
  const serverPath =
      path.resolve((userDefinedServerPath === '') ? 'mlir-lsp-server'
                                                  : userDefinedServerPath);
  const fileWatcherConfig = {
    disableGlobbing : true,
    followSymlinks : true,
    ignoreInitial : true,
  };
  const fileWatcher = chokidar.watch(serverPath, fileWatcherConfig);
  fileWatcher.on('all', (_event, _filename, _details) => {
    promptRestart(
        'onSettingsChanged',
        'MLIR language server binary has changed. Do you want to reload the server?');
  });
  mlirContext.subscriptions.push(
      new vscode.Disposable(() => { fileWatcher.close(); }));
}
