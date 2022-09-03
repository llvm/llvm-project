import * as vscode from 'vscode';

import {MLIRContext} from '../mlirContext';
import {ViewPDLLCommand} from './commands/viewOutput';

/**
 *  Register the necessary extensions for supporting PDLL.
 */
export function registerPDLLExtensions(context: vscode.ExtensionContext,
                                       mlirContext: MLIRContext) {
  context.subscriptions.push(new ViewPDLLCommand(mlirContext));
}
