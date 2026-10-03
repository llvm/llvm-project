# LLVM LSP server

## Usage
To use this language server in your favourite editor, please refer to the documentation of that editor.

First party support for this LSP server in VS Code is the LLVM VS Code extension, to use LLVM IR LSP server in VS Code
install that extension.


## Build
Setup cmake using the tutorial from https://llvm.org/docs/CMake.html#quick-start
and build the `llvm-lsp-server` target.

## Features

This LSP server is built to the [Language Server Protocol Specification 3.17](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/). It provides several standard features to enhance the development experience.

---

### Standard Capabilities

The server supports the following standard LSP capabilities:

* `textDocumentSync.openClose`: Synchronizes document content with the server.
* `referencesProvider`: Finds all references to a symbol.
* `documentsSymbolProvider`: Provides a tree of document symbols, enables breadcrums navigation in the editor.
