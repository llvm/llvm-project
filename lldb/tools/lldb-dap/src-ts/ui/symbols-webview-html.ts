import * as vscode from "vscode";

export function getSymbolsTableHTMLContent(tabulatorJsPath: vscode.Uri, tabulatorCssPath: vscode.Uri, symbolsTableScriptPath: vscode.Uri): string {
    return `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <link href="${tabulatorCssPath}" rel="stylesheet">
    <style>
      .tabulator {
        background-color: var(--vscode-editor-background);
        color: var(--vscode-editor-foreground);
      }

      .tabulator .tabulator-header .tabulator-col {
        background-color: var(--vscode-editor-background);
        color: var(--vscode-editor-foreground);
      }

      .tabulator-row {
        background-color: var(--vscode-editor-background);
        color: var(--vscode-editor-foreground);
      }

      .tabulator-row.tabulator-row-even {
        background-color: var(--vscode-editor-background);
        color: var(--vscode-editor-foreground);
      }

      .tabulator-row.tabulator-selected {
        background-color: var(--vscode-editor-background);
        color: var(--vscode-editor-foreground);
      }

      .tabulator-cell {
        text-overflow: clip !important;
      }

      #symbols-table {
        width: 100%;
        height: 100vh;
      }
    </style>
</head>
<body>
    <div id="symbols-table"></div>
    <script src="${tabulatorJsPath}"></script>
    <script src="${symbolsTableScriptPath}"></script>
</body>
</html>`;
}