import * as vscode from 'vscode';

import {MLIRContext} from '../mlirContext';
import {registerMLIRBytecodeExtensions} from './bytecodeProvider';

/**
 *  Register the necessary extensions for supporting MLIR.
 */
export function registerMLIRExtensions(context: vscode.ExtensionContext,
                                       mlirContext: MLIRContext) {
  registerMLIRBytecodeExtensions(context, mlirContext);
}
