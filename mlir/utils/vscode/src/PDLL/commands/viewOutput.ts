import * as vscode from 'vscode'

import {Command} from '../../command';
import {MLIRContext} from '../../mlirContext';

/**
 * The parameters to the pdll/viewOutput command. These parameters are:
 * - `uri`: The URI of the file to view.
 * - `kind`: The kind of the output to generate.
 */
type ViewOutputParams = Partial<{uri : string, kind : string}>;

/**
 * The output of the commands:
 * - `output`: The output string of the command, e.g. a .mlir PDL string.
 */
type ViewOutputResult = Partial<{output : string}>;

/**
 * A command that displays the output of the current PDLL document.
 */
export class ViewPDLLCommand extends Command {
  constructor(context: MLIRContext) { super('mlir.viewPDLLOutput', context); }

  async execute() {
    const editor = vscode.window.activeTextEditor;
    if (editor.document.languageId != 'pdll')
      return;

    // Check to see if a language client is active for this document.
    const pdllClient =
        this.context.getLanguageClient(editor.document.uri, "pdll");
    if (!pdllClient) {
      return;
    }

    // Ask the user for the desired output type.
    const outputType =
        await vscode.window.showQuickPick([ 'ast', 'mlir', 'cpp' ]);
    if (!outputType) {
      return;
    }

    // If we have the language client, ask it to try compiling the document.
    let outputParams: ViewOutputParams = {
      uri : editor.document.uri.toString(),
      kind : outputType,
    };
    const result: ViewOutputResult|undefined =
        await pdllClient.sendRequest('pdll/viewOutput', outputParams);
    if (!result || result.output.length === 0) {
      return;
    }

    // Display the output in a new editor.
    let outputFileType = 'plaintext';
    if (outputType == 'mlir') {
      outputFileType = 'mlir';
    } else if (outputType == 'cpp') {
      outputFileType = 'cpp';
    }
    await vscode.workspace.openTextDocument(
        {language : outputFileType, content : result.output});
  }
}
