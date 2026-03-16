import * as vscode from 'vscode';

import {MLIRContext} from '../mlirContext';
import {registerMLIRBytecodeExtensions} from './bytecodeProvider';
import {RunTestCommand} from './commands/runTest';

/**
 *  Register the necessary extensions for supporting MLIR.
 */
export function registerMLIRExtensions(context: vscode.ExtensionContext,
                                       mlirContext: MLIRContext) {
  registerMLIRBytecodeExtensions(context, mlirContext);
  context.subscriptions.push(new RunTestCommand(mlirContext));
}
