import * as vscode from 'vscode';

import {MLIRContext} from './mlirContext';

/**
 *  This method is called when the extension is activated. The extension is
 *  activated the very first time a command is executed.
 */
export function activate(context: vscode.ExtensionContext) {
  const outputChannel = vscode.window.createOutputChannel('MLIR');
  context.subscriptions.push(outputChannel);

  const mlirContext = new MLIRContext();
  context.subscriptions.push(mlirContext);

  // Initialize the commands of the extension.
  context.subscriptions.push(
      vscode.commands.registerCommand('mlir.restart', async () => {
        mlirContext.dispose();
        await mlirContext.activate(outputChannel);
      }));

  mlirContext.activate(outputChannel);
}
