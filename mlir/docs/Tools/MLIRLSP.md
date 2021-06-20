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
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
```

### Features

This section details a few of the features that the MLIR language server
provides. The screenshots are shown in [VSCode](https://code.visualstudio.com/),
but the exact feature set available will depend on your editor client.

#### Diagnostics

The language server runs actively runs verification on the IR as you type,
showing any generate diagnostics in-place.

![IMG](/mlir-lsp-server/diagnostics.png)

#### Cross-references

Cross references allow for navigating the use/def chains of SSA values (i.e.
operation results and block arguments), [Symbols](../SymbolsAndSymbolTables.md),
and Blocks.

##### Find definition

Jump to the definition of the IR entity under the cursor. A few examples are
shown below:

*   SSA Values

![SSA](/mlir-lsp-server/goto_def_ssa.gif)

*   Symbol References

![Symbols](/mlir-lsp-server/goto_def_symbol.gif)

The definition of an operation will also take into account the source location
attached, allowing for navigating into the source file that generated the
operation.

![External Locations](/mlir-lsp-server/goto_def_external.gif)

##### Find references

Show all references of the IR entity under the cursor.

![IMG](/mlir-lsp-server/find_references.gif)

#### Hover

Hover over an IR entity to see more information about it. The exact information
displayed is dependent on the type of IR entity under the cursor. For example,
hovering over an `Operation` may show its generic format.

![IMG](/mlir-lsp-server/hover.png)

#### Navigation

The language server will also inform the editor about the structure of symbol
tables within the IR. This allows for jumping directly to the definition of a
symbol, such as a `func`, within the file.

![IMG](/mlir-lsp-server/navigation.gif)

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

Provides [MLIR](https://mlir.llvm.org/) language IDE features for VS code:

*   Syntax highlighting for .mlir files and `mlir` markdown blocks
*   go-to-definition and cross references
*   Detailed information when hovering over IR entities
*   Outline and navigation of symbols and symbol tables
*   Live parser and verifier diagnostics

#### Setup

This extension requires the
[`mlir-lsp-server` language server](https://mlir.llvm.org/docs/Tools/MLIRLSP/).
If not found in your path, you must specify the path of the server in the
settings of this extension.

#### Contributing

This extension is actively developed within the
[LLVM monorepo](https://github.com/llvm/llvm-project/tree/main/mlir/utils/vscode),
at `mlir/utils/vscode`. As such, contributions should follow the
[normal LLVM guidelines](https://llvm.org/docs/Contributing.html), with code
reviews sent to
[phabricator](https://llvm.org/docs/Contributing.html#how-to-submit-a-patch).

When developing or deploying this extension within the LLVM monorepo, a few
extra setup steps are required:

*   Copy `mlir/utils/textmate/mlir.json` to the extension directory and rename
    to `grammar.json`.

Please follow the existing code style when contributing to the extension, we
recommend to run `npm run format` before sending a patch.
