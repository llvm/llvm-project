import * as vscode from "vscode";

export interface CommandArgs {
  command: string;
}

export class CommandTool implements vscode.LanguageModelTool<CommandArgs> {
  async invoke(
    options: vscode.LanguageModelToolInvocationOptions<CommandArgs>,
    token: vscode.CancellationToken,
  ): Promise<vscode.LanguageModelToolResult> {
    const session = vscode.debug.activeDebugSession;
    if (session === undefined) {
      return new vscode.LanguageModelToolResult([
        new vscode.LanguageModelTextPart("Error: no debug session is active"),
      ]);
    }
    const response = await session.customRequest("evaluate", {
      expression: options.input.command,
      context: "repl",
    });
    return new vscode.LanguageModelToolResult([
      new vscode.LanguageModelTextPart(response.result),
    ]);
  }
}
