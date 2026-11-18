/**
 * This file was copied from /mlir/utils/vscode/src/command.ts and adapted for use in LLVM
  */

import * as vscode from 'vscode';
import { LLVMContext } from './llvmContext';

/**
 * This class represents a base vscode command. It handles all of the necessary
 * command registration and disposal boilerplate.
 */
export abstract class Command extends vscode.Disposable {
  private disposable: vscode.Disposable;
  protected context: LLVMContext;

  constructor(command: string, context: LLVMContext) {
    super(() => this.dispose());
    this.disposable =
      vscode.commands.registerCommand(command, this.execute, this);
    this.context = context;
  }

  dispose() { this.disposable && this.disposable.dispose(); }

  /**
   * The function executed when this command is invoked.
   */
  abstract execute(...args: any[]): any;
}
