# MLIR : Language Server Protocol

[TOC]

This document describes the tools and utilities related to supporting
[LSP](https://microsoft.github.io/language-server-protocol/) IDE language
extensions for the MLIR textual assembly format. An LSP language extension is
generally comprised of two components; a language client and a language server.
A language client is a piece of code that interacts with the IDE that you are
using, such as VSCode. A language server acts as the backend for queries that
the client may want to perform, such as "Find Definition", "Find References",
etc.

## MLIR LSP Language Server : `mlir-lsp-server`

MLIR provides an implementation of an LSP language server in the form of the
`mlir-lsp-server` tool. This tool interacts with the MLIR C++ API to support
rich language queries, such as "Find Definition".

### Supporting custom dialects and passes

`mlir-lsp-server`, like many other MLIR based tools, relies on having the
appropriate dialects registered to be able to parse in the custom assembly
formats used in the textual .mlir files. The `mlir-lsp-server` found within the
main MLIR repository provides support for all of the upstream MLIR dialects and
passes. Downstream and out-of-tree users will need to provide a custom
`mlir-lsp-server` executable that registers the entities that they are
interested in. The implementation of `mlir-lsp-server` is provided as a library,
making it easy for downstream users to register their dialect/passes and simply
call into the main implementation. A simple example is shown below:

```c++
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerMyDialects(registry);
  registerMyPasses();
  return failed(mlir::MlirLspServerMain(argc, argv, registry));
}
```

### Design

The design of `mlir-lsp-server` is largely comprised of three different
components:

*   Communication and Transport (via JSON-RPC)
*   Language Server Protocol
*   MLIR Language Server

![Index Map Example](/includes/img/mlir-lsp-server-server_diagram.svg)

#### Communication and Transport

`mlir-lsp-server` communicates with the language client via JSON-RPC over
stdin/stdout. In the code, this is the `JSONTransport` class. This class knows
nothing about the Language Server Protocol, it only knows that JSON-RPC messages
are coming in and JSON-RPC messages are going out. The handling of incoming and
outgoing LSP messages is left to the `MessageHandler` class. This class routes
incoming messages to handlers in the `Language Server Protocol` layer for
interpretation, and packages outgoing messages for transport. This class also
has limited knowledge of the LSP, and only has information about the three main
classes of messages: notifications, calls, and replies.

#### Language Server Protocol

`LSPServer` handles the interpretation of the finer LSP details. This class
registers handlers for LSP messages and then forwards to the `MLIR Language
Server` for processing. The intent of this component is to hold all of the
necessary glue when communicating from the MLIR world to the LSP world. In most
cases, the LSP message handlers simply forward to the `MLIR Language Server`. In
some cases however, the impedance mismatch between the two requires more
complicated glue code.

#### MLIR Language Server

`MLIRServer` provides the internal MLIR-based implementation of all of LSP
queries. This is the class that directly interacts with the MLIR C++ API,
including parsing .mlir text files, running passes, etc.

## Editor Plugins

LSP Language plugins are available for many popular editors, and in principle
`mlir-lsp-server` should work with any of them, though feature set and interface
may vary. Below are a set of plugins that are known to work:

### Visual Studio Code

Provides MLIR language IDE features for VS code.

#### Setup

This extension requires the `mlir-lsp-server` language server. If not found in
your path, you must specify the path of the server in the settings of this
extension.

#### Developing in the LLVM monorepo

This extension is actively developed within the LLVM monorepo, at
`mlir/utils/vscode`. When developing or deploying this extension within the LLVM
monorepo, a few extra steps for setup are required:

*   Copy `mlir/utils/textmate/mlir.json` to the extension directory and rename
    to `grammar.json`.

#### Features

*   Syntax highlighting for .mlir files and `mlir` markdown blocks
*   go-to-definition and cross references
    *   Definitions include the source file locations of operations in the .mlir
*   Hover over IR entities to see more information about them
    *   e.g. for a Block, you can see its block number as well as any
        predecessors or successors.
