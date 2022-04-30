import * as vscode from 'vscode';

import {MLIRContext} from '../mlirContext';
import {ViewPDLLCommand} from './commands/viewOutput';

/**
 *  Register the necessary context and commands for PDLL.
 */
export function registerPDLLCommands(context: vscode.ExtensionContext,
                                     mlirContext: MLIRContext) {
  context.subscriptions.push(new ViewPDLLCommand(mlirContext));
}
