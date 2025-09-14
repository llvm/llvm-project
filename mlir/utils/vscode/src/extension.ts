import * as vscode from 'vscode';

import {registerMLIRExtensions} from './MLIR/mlir';
import {MLIRContext} from './mlirContext';
import {registerPDLLExtensions} from './PDLL/pdll';
import { LitTestProvider } from './LIT/lit';
/**
 *  This method is called when the extension is activated. The extension is
 *  activated the very first time a command is executed.
 */
export function activate(context: vscode.ExtensionContext) {
  const outputChannel = vscode.window.createOutputChannel('MLIR');
  context.subscriptions.push(outputChannel);

  const mlirContext = new MLIRContext();
  context.subscriptions.push(mlirContext);

  const litTests = new LitTestProvider(context);  // Instantiate the LitTestProvider
  context.subscriptions.push(litTests);  // Push it to context.subscriptions to handle cleanup

  // Initialize the commands of the extension.
  context.subscriptions.push(
      vscode.commands.registerCommand('mlir.restart', async () => {
        // Dispose and reactivate the context.
        mlirContext.dispose();
        await mlirContext.activate(outputChannel);
      }));
  context.subscriptions.push(vscode.commands.registerCommand('lit.reconfigure', () => {
    litTests.reconfigureLitSettings();
  }));
  registerMLIRExtensions(context, mlirContext);
  registerPDLLExtensions(context, mlirContext);

  mlirContext.activate(outputChannel);
}
