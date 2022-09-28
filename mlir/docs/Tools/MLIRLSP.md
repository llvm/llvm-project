# MLIR : Language Server Protocol

[TOC]

This document describes the tools and utilities related to supporting
[LSP](https://microsoft.github.io/language-server-protocol/) IDE language
extensions for various MLIR-related languages. An LSP language extension is
generally comprised of two components; a language client and a language server.
A language client is a piece of code that interacts with the IDE that you are
using, such as VSCode. A language server acts as the backend for queries that
the client may want to perform, such as "Find Definition", "Find References",
etc.

## MLIR LSP Language Server : `mlir-lsp-server`

MLIR provides an implementation of an LSP language server for `.mlir` text files
in the form of the `mlir-lsp-server` tool. This tool interacts with the MLIR C++
API to support rich language queries, such as "Find Definition".

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

See the [Editor Plugins](#editor-plugins) section below for details on how to
setup support in a few known LSP clients, such as vscode.

### Features

This section details a few of the features that the MLIR language server
provides. The screenshots are shown in [VSCode](https://code.visualstudio.com/),
but the exact feature set available will depend on your editor client.

[mlir features]: #

#### Diagnostics

The language server actively runs verification on the IR as you type, showing
any generated diagnostics in-place.

![IMG](/mlir-lsp-server/diagnostics.png)

##### Automatically insert `expected-` diagnostic checks

MLIR provides
[infrastructure](https://mlir.llvm.org/docs/Diagnostics/#sourcemgr-diagnostic-verifier-handler)
for checking expected diagnostics, which is heavily utilized when defining IR
parsing and verification. The language server provides code actions for
automatically inserting the checks for diagnostics it knows about.

![IMG](/mlir-lsp-server/diagnostics_action.gif)

#### Code completion

The language server provides suggestions as you type, offering completions for
dialect constructs (such as attributes, operations, and types), block names, SSA
value names, keywords, and more.

![IMG](/mlir-lsp-server/code_complete.gif)

#### Cross-references

Cross references allow for navigating the use/def chains of SSA values (i.e.
operation results and block arguments), [Symbols](../SymbolsAndSymbolTables.md),
and Blocks.

##### Find definition

Jump to the definition of the IR entity under the cursor. A few examples are
shown below:

- SSA Values

![SSA](/mlir-lsp-server/goto_def_ssa.gif)

- Symbol References

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
symbol, such as a `func.func`, within the file.

![IMG](/mlir-lsp-server/navigation.gif)

## PDLL LSP Language Server : `mlir-pdll-lsp-server`

MLIR provides an implementation of an LSP language server for `.pdll` text files
in the form of the `mlir-pdll-lsp-server` tool. This tool interacts with the
PDLL C++ API to support rich language queries, such as code completion and "Find
Definition".

### Compilation Database

Similarly to
[`clangd`](https://clang.llvm.org/docs/JSONCompilationDatabase.html), and
language servers for various other programming languages, the PDLL language
server relies on a compilation database to provide build-system information for
`.pdll` files. This information includes, for example, the include directories
available for that file. This database allows for the server to interact with
`.pdll` files using the same configuration as when building.

#### Format

A PDLL compilation database is a YAML file, conventionally named
`pdll_compile_commands.yml`, that contains a set of `FileInfo` documents
providing information for individiual `.pdll` files.

Example:

```yaml
--- !FileInfo:
  filepath: "/home/user/llvm/mlir/lib/Dialect/Arithmetic/IR/ArithmeticCanonicalization.pdll"
  includes: "/home/user/llvm/mlir/lib/Dialect/Arithmetic/IR;/home/user/llvm/mlir/include"
```

- filepath: <string> - Absolute file path of the file.
- includes: <string> - Semi-colon delimited list of absolute include directories.

#### Build System Integration

Per convention, PDLL compilation databases should be named
`pdll_compile_commands.yml` and placed at the top of the build directory. When
using CMake and `mlir_pdll`, a compilation database is generally automatically
built and placed in the appropriate location.

### Features

This section details a few of the features that the PDLL language server
provides. The screenshots are shown in [VSCode](https://code.visualstudio.com/),
but the exact feature set available will depend on your editor client.

[pdll features]: #

#### Diagnostics

The language server actively runs verification as you type, showing any
generated diagnostics in-place.

![IMG](/mlir-pdll-lsp-server/diagnostics.png)

#### Code completion and signature help

The language server provides suggestions as you type based on what constraints,
rewrites, dialects, operations, etc are available in this context. The server
also provides information about the structure of constraint and rewrite calls,
operations, and more as you fill them in.

![IMG](/mlir-pdll-lsp-server/code_complete.gif)

#### Cross-references

Cross references allow for navigating the code base.

##### Find definition

Jump to the definition of a symbol under the cursor:

![IMG](/mlir-pdll-lsp-server/goto_def.gif)

If ODS information is available, we can also jump to the definition of operation
names and more:

![IMG](/mlir-pdll-lsp-server/goto_def_ods.gif)

##### Find references

Show all references of the symbol under the cursor.

![IMG](/mlir-pdll-lsp-server/find_references.gif)

#### Hover

Hover over a symbol to see more information about it, such as its type,
documentation, and more.

![IMG](/mlir-pdll-lsp-server/hover.png)

If ODS information is available, we can also show information directly from the
operation definitions:

![IMG](/mlir-pdll-lsp-server/hover_ods.png)

#### Navigation

The language server will also inform the editor about the structure of symbols
within the IR.

![IMG](/mlir-pdll-lsp-server/navigation.gif)

#### View intermediate output

The language server provides support for introspecting various intermediate
stages of compilation, such as the AST, the `.mlir` containing the generated
PDL, and the generated C++ glue. This is a custom LSP extension, and is not
necessarily provided by all IDE clients.

![IMG](/mlir-pdll-lsp-server/view_output.gif)

#### Inlay hints

The language server provides additional information inline with the source code.
Editors usually render this using read-only virtual text snippets interspersed
with code. Hints may be shown for:

* types of local variables
* names of operand and result groups
* constraint and rewrite arguments

![IMG](/mlir-pdll-lsp-server/inlay_hints.png)

## TableGen LSP Language Server : `tblgen-lsp-server`

MLIR provides an implementation of an LSP language server for `.td` text files
in the form of the `tblgen-lsp-server` tool. This tool interacts with the
TableGen C++ API to support rich language queries, such as "Find Definition".

### Compilation Database

Similarly to
[`clangd`](https://clang.llvm.org/docs/JSONCompilationDatabase.html), and
language servers for various other programming languages, the TableGen language
server relies on a compilation database to provide build-system information for
`.td` files. This information includes, for example, the include directories
available for that file. This database allows for the server to interact with
`.td` files using the same configuration as when building.

#### Format

A TableGen compilation database is a YAML file, conventionally named
`tablegen_compile_commands.yml`, that contains a set of `FileInfo` documents
providing information for individiual `.td` files.

Example:

```yaml
--- !FileInfo:
  filepath: "/home/user/llvm/mlir/lib/Dialect/Arithmetic/IR/ArithmeticCanonicalization.td"
  includes: "/home/user/llvm/mlir/lib/Dialect/Arithmetic/IR;/home/user/llvm/mlir/include"
```

- filepath: <string> - Absolute file path of the file.
- includes: <string> - Semi-colon delimited list of absolute include directories.

#### Build System Integration

Per convention, TableGen compilation databases should be named
`tablegen_compile_commands.yml` and placed at the top of the build directory.
When using CMake and `mlir_tablegen`, a compilation database is generally
automatically built and placed in the appropriate location.

### Features

This section details a few of the features that the TableGen language server
provides. The screenshots are shown in [VSCode](https://code.visualstudio.com/),
but the exact feature set available will depend on your editor client.

[tablegen features]: #

#### Diagnostics

The language server actively runs verification as you type, showing any
generated diagnostics in-place.

![IMG](/tblgen-lsp-server/diagnostics.png)

#### Cross-references

Cross references allow for navigating the code base.

##### Find definition

Jump to the definition of a symbol under the cursor:

![IMG](/tblgen-lsp-server/goto_def.gif)

##### Find references

Show all references of the symbol under the cursor.

![IMG](/tblgen-lsp-server/find_references.gif)

#### Hover

Hover over a symbol to see more information about it, such as its type,
documentation, and more.

![IMG](/tblgen-lsp-server/hover_def.png)

Hovering over an overridden field will also show you information such as
documentation from the base value:

![IMG](/tblgen-lsp-server/hover_field.png)

## Language Server Design

The design of the various language servers provided by MLIR are effectively the
same, and are largely comprised of three different components:

- Communication and Transport (via JSON-RPC)
- Language Server Protocol
- Language-Specific Server

![Index Map Example](/includes/img/mlir-lsp-server-server_diagram.svg)

### Communication and Transport

The language server, such as `mlir-lsp-server`, communicates with the language
client via JSON-RPC over stdin/stdout. In the code, this is the `JSONTransport`
class. This class knows nothing about the Language Server Protocol, it only
knows that JSON-RPC messages are coming in and JSON-RPC messages are going out.
The handling of incoming and outgoing LSP messages is left to the
`MessageHandler` class. This class routes incoming messages to handlers in the
`Language Server Protocol` layer for interpretation, and packages outgoing
messages for transport. This class also has limited knowledge of the LSP, and
only has information about the three main classes of messages: notifications,
calls, and replies.

### Language Server Protocol

`LSPServer` handles the interpretation of the finer LSP details. This class
registers handlers for LSP messages and then forwards to the
[`Language-Specific Server`](#language-specific-server) for processing. The
intent of this component is to hold all of the necessary glue when communicating
from the LSP world to the language-specific world (e.g. MLIR, PDLL, etc.). In
most cases, the LSP message handlers simply forward directly to the
`Language-Specific Server`. In some cases, however, the impedance mismatch
between the two requires more complicated glue code.

### Language-Specific Server

The language specific server, such as `MLIRServer` or `PDLLServer`, provides the
internal implementation of all of LSP queries for a specific language. These are
the classes that directly interacts with the C++ API for the language, including
parsing text files, interpreting definition/reference information, etc.

## Editor Plugins

LSP Language plugins are available for many popular editors, and in principle
the language servers provided by MLIR should work with any of them, though
feature sets and interfaces may vary. Below are a set of plugins that are known
to work:

### Visual Studio Code

Provides language IDE features for [MLIR](https://mlir.llvm.org/) related
languages: [MLIR](#mlir---mlir-textual-assembly-format),
[PDLL](#pdll---mlir-pdll-pattern-files), and [TableGen](#td---tablegen-files)

#### `.mlir` - MLIR textual assembly format:

The MLIR extension adds language support for the
[MLIR textual assembly format](https://mlir.llvm.org/docs/LangRef/):

##### Features

- Syntax highlighting for `.mlir` files and `mlir` markdown blocks
- go-to-definition and cross references
- Detailed information when hovering over IR entities
- Outline and navigation of symbols and symbol tables
- Code completion
- Live parser and verifier diagnostics

[mlir-vscode features]: #

##### Setup

###### `mlir-lsp-server`

The various `.mlir` language features require the
[`mlir-lsp-server` language server](https://mlir.llvm.org/docs/Tools/MLIRLSP/#mlir-lsp-language-server--mlir-lsp-server).
If `mlir-lsp-server` is not found within your workspace path, you must specify
the path of the server via the `mlir.server_path` setting. The path of the
server may be absolute or relative within your workspace.

#### `.pdll` - MLIR PDLL pattern files:

The MLIR extension adds language support for the
[PDLL pattern language](https://mlir.llvm.org/docs/PDLL/).

##### Features

- Syntax highlighting for `.pdll` files and `pdll` markdown blocks
- go-to-definition and cross references
- Types and documentation on hover
- Code completion and signature help
- View intermediate AST, MLIR, or C++ output

[pdll-vscode features]: #

##### Setup

###### `mlir-pdll-lsp-server`

The various `.pdll` language features require the
[`mlir-pdll-lsp-server` language server](https://mlir.llvm.org/docs/Tools/MLIRLSP/#pdll-lsp-language-server--mlir-pdll-lsp-server).
If `mlir-pdll-lsp-server` is not found within your workspace path, you must
specify the path of the server via the `mlir.pdll_server_path` setting. The path
of the server may be absolute or relative within your workspace.

###### Project setup

To properly understand and interact with `.pdll` files, the language server must
understand how the project is built (compile flags).
[`pdll_compile_commands.yml` files](https://mlir.llvm.org/docs/Tools/MLIRLSP/#compilation-database)
related to your project should be provided to ensure files are properly
processed. These files can usually be generated by the build system, and the
server will attempt to find them within your `build/` directory. If not
available in or a unique location, additional `pdll_compile_commands.yml` files
may be specified via the `mlir.pdll_compilation_databases` setting. The paths of
these databases may be absolute or relative within your workspace.

#### `.td` - TableGen files:

The MLIR extension adds language support for the
[TableGen language](https://llvm.org/docs/TableGen/ProgRef.html).

##### Features

- Syntax highlighting for `.td` files and `tablegen` markdown blocks
- go-to-definition and cross references

[tablegen-vscode features]: #

##### Setup

###### `tblgen-lsp-server`

The various `.td` language features require the
[`tblgen-lsp-server` language server](https://mlir.llvm.org/docs/Tools/MLIRLSP/#tablegen-lsp-language-server--tblgen-lsp-server).
If `tblgen-lsp-server` is not found within your workspace path, you must specify
the path of the server via the `mlir.tablegen_server_path` setting. The path of
the server may be absolute or relative within your workspace.

###### Project setup

To properly understand and interact with `.td` files, the language server must
understand how the project is built (compile flags).
[`tablegen_compile_commands.yml` files](https://mlir.llvm.org/docs/Tools/MLIRLSP/#compilation-database-1)
related to your project should be provided to ensure files are properly
processed. These files can usually be generated by the build system, and the
server will attempt to find them within your `build/` directory. If not
available in or a unique location, additional `tablegen_compile_commands.yml`
files may be specified via the `mlir.tablegen_compilation_databases` setting.
The paths of these databases may be absolute or relative within your workspace.

#### Contributing

This extension is actively developed within the
[LLVM monorepo](https://github.com/llvm/llvm-project), at
[`mlir/utils/vscode`](https://github.com/llvm/llvm-project/tree/main/mlir/utils/vscode).
As such, contributions should follow the
[normal LLVM guidelines](https://llvm.org/docs/Contributing.html), with code
reviews sent to
[phabricator](https://llvm.org/docs/Contributing.html#how-to-submit-a-patch).

When developing or deploying this extension within the LLVM monorepo, a few
extra setup steps are required:

- Copy `mlir/utils/textmate/mlir.json` to the extension directory and rename to
  `grammar.json`.
- Copy `llvm/utils/textmate/tablegen.json` to the extension directory and rename
  to `tablegen-grammar.json`.
- Copy
  `https://mlir.llvm.org//LogoAssets/logo/PNG/full_color/mlir-identity-03.png`
  to the extension directory and rename to `icon.png`.

Please follow the existing code style when contributing to the extension, we
recommend to run `npm run format` before sending a patch.
