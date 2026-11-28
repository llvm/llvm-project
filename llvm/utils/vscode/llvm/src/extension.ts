/**
 * This file was copied from /mlir/utils/vscode/src/extension.ts and adapted for use in LLVM
 */

import * as vscode from 'vscode';

import { LITTaskProvider } from './litTaskProvider';
import { LLVMContext } from './llvmContext';

/**
 *  This method is called when the extension is activated. The extension is
 *  activated the very first time a command is executed.
 */
export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(vscode.tasks.registerTaskProvider(LITTaskProvider.LITType, new LITTaskProvider()));

  const outputChannel = vscode.window.createOutputChannel('llvm-lsp-server', 'Log');
  context.subscriptions.push(outputChannel);

  const llvmContext = new LLVMContext(context, outputChannel);
  context.subscriptions.push(llvmContext);

  // Initialize the commands of the extension.
  context.subscriptions.push(
    vscode.commands.registerCommand('llvm.restart', async () => {
      // Dispose and reactivate the context.
      llvmContext.dispose();
      await llvmContext.activate();
    }));

  llvmContext.activate();
  outputChannel.appendLine("LLVM: extension activated!");
}
