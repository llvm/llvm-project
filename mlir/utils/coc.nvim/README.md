# MLIR

Ported from [vscode-mlir](https://github.com/llvm/vscode-mlir).

Because some APIs of [vscode](github.com/microsoft/vscode) are missing in
[coc.nvim](https://github.com/neoclide/coc.nvim), disable some features
temporarily:

- command `mlir.viewPDLLOutput`: miss
  `vscode.workspace.openTextDocument({language, content})`
- custom editor `mlir.bytecode`: miss
  `vscode.window.registerCustomEditorProvider()`
- filesystem `mlir.bytecode-mlir`: miss `vscode.workspace.registerFileSystemProvider()`
- use `chokidar.watch()` to detect the changes of the paths of language
  servers: miss `vscode.Disposable.constructor()`

## Install

- [coc-marketplace](https://github.com/fannheyward/coc-marketplace)
- [npm](https://www.npmjs.com/package/coc-mlir)
- vim:

```vim
" command line
CocInstall coc-mlir
" or add the following code to your vimrc
let g:coc_global_extensions = ['coc-mlir', 'other coc-plugins']
```

---

Provides language IDE features for [MLIR](https://mlir.llvm.org/) related
languages: [MLIR](#mlir---mlir-textual-assembly-format),
[PDLL](#pdll---mlir-pdll-pattern-files), and [TableGen](#td---tablegen-files)

## `.mlir` - MLIR textual assembly format:

The MLIR extension adds language support for the
[MLIR textual assembly format](https://mlir.llvm.org/docs/LangRef/):

### Features

- Syntax highlighting for `.mlir` files and `mlir` markdown blocks
- go-to-definition and cross references
- Detailed information when hovering over IR entities
- Outline and navigation of symbols and symbol tables
- Code completion
- Live parser and verifier diagnostics

#### Diagnostics

The language server actively runs verification on the IR as you type, showing
any generated diagnostics in-place.

![IMG](https://mlir.llvm.org/mlir-lsp-server/diagnostics.png)

##### Automatically insert `expected-` diagnostic checks

MLIR provides
[infrastructure](https://mlir.llvm.org/docs/Diagnostics/#sourcemgr-diagnostic-verifier-handler)
for checking expected diagnostics, which is heavily utilized when defining IR
parsing and verification. The language server provides code actions for
automatically inserting the checks for diagnostics it knows about.

![IMG](https://mlir.llvm.org/mlir-lsp-server/diagnostics_action.gif)

#### Code completion

The language server provides suggestions as you type, offering completions for
dialect constructs (such as attributes, operations, and types), block names, SSA
value names, keywords, and more.

![IMG](https://mlir.llvm.org/mlir-lsp-server/code_complete.gif)

#### Cross-references

Cross references allow for navigating the use/def chains of SSA values (i.e.
operation results and block arguments), [Symbols](../SymbolsAndSymbolTables.md),
and Blocks.

##### Find definition

Jump to the definition of the IR entity under the cursor. A few examples are
shown below:

- SSA Values

![SSA](https://mlir.llvm.org/mlir-lsp-server/goto_def_ssa.gif)

- Symbol References

![Symbols](https://mlir.llvm.org/mlir-lsp-server/goto_def_symbol.gif)

The definition of an operation will also take into account the source location
attached, allowing for navigating into the source file that generated the
operation.

![External Locations](https://mlir.llvm.org/mlir-lsp-server/goto_def_external.gif)

##### Find references

Show all references of the IR entity under the cursor.

![IMG](https://mlir.llvm.org/mlir-lsp-server/find_references.gif)

#### Hover

Hover over an IR entity to see more information about it. The exact information
displayed is dependent on the type of IR entity under the cursor. For example,
hovering over an `Operation` may show its generic format.

![IMG](https://mlir.llvm.org/mlir-lsp-server/hover.png)

#### Navigation

The language server will also inform the editor about the structure of symbol
tables within the IR. This allows for jumping directly to the definition of a
symbol, such as a `func.func`, within the file.

![IMG](https://mlir.llvm.org/mlir-lsp-server/navigation.gif)

#### Bytecode Editing and Inspection

The language server provides support for interacting with MLIR bytecode files,
enabling IDEs to transparently view and edit bytecode files in the same way
as textual `.mlir` files.

![IMG](https://mlir.llvm.org/mlir-lsp-server/bytecode_edit.gif)

### Setup

#### `mlir-lsp-server`

The various `.mlir` language features require the
[`mlir-lsp-server` language server](https://mlir.llvm.org/docs/Tools/MLIRLSP/#mlir-lsp-language-server--mlir-lsp-server).
If `mlir-lsp-server` is not found within your workspace path, you must specify
the path of the server via the `mlir.server_path` setting. The path of the
server may be absolute or relative within your workspace.

## `.pdll` - MLIR PDLL pattern files:

The MLIR extension adds language support for the
[PDLL pattern language](https://mlir.llvm.org/docs/PDLL/).

### Features

- Syntax highlighting for `.pdll` files and `pdll` markdown blocks
- go-to-definition and cross references
- Types and documentation on hover
- Code completion and signature help
- View intermediate AST, MLIR, or C++ output

#### Diagnostics

The language server actively runs verification as you type, showing any
generated diagnostics in-place.

![IMG](https://mlir.llvm.org/mlir-pdll-lsp-server/diagnostics.png)

#### Code completion and signature help

The language server provides suggestions as you type based on what constraints,
rewrites, dialects, operations, etc are available in this context. The server
also provides information about the structure of constraint and rewrite calls,
operations, and more as you fill them in.

![IMG](https://mlir.llvm.org/mlir-pdll-lsp-server/code_complete.gif)

#### Cross-references

Cross references allow for navigating the code base.

##### Find definition

Jump to the definition of a symbol under the cursor:

![IMG](https://mlir.llvm.org/mlir-pdll-lsp-server/goto_def.gif)

If ODS information is available, we can also jump to the definition of operation
names and more:

![IMG](https://mlir.llvm.org/mlir-pdll-lsp-server/goto_def_ods.gif)

##### Find references

Show all references of the symbol under the cursor.

![IMG](https://mlir.llvm.org/mlir-pdll-lsp-server/find_references.gif)

#### Hover

Hover over a symbol to see more information about it, such as its type,
documentation, and more.

![IMG](https://mlir.llvm.org/mlir-pdll-lsp-server/hover.png)

If ODS information is available, we can also show information directly from the
operation definitions:

![IMG](https://mlir.llvm.org/mlir-pdll-lsp-server/hover_ods.png)

#### Navigation

The language server will also inform the editor about the structure of symbols
within the IR.

![IMG](https://mlir.llvm.org/mlir-pdll-lsp-server/navigation.gif)

#### View intermediate output

The language server provides support for introspecting various intermediate
stages of compilation, such as the AST, the `.mlir` containing the generated
PDL, and the generated C++ glue. This is a custom LSP extension, and is not
necessarily provided by all IDE clients.

![IMG](https://mlir.llvm.org/mlir-pdll-lsp-server/view_output.gif)

#### Inlay hints

The language server provides additional information inline with the source code.
Editors usually render this using read-only virtual text snippets interspersed
with code. Hints may be shown for:

- types of local variables
- names of operand and result groups
- constraint and rewrite arguments

![IMG](https://mlir.llvm.org/mlir-pdll-lsp-server/inlay_hints.png)

### Setup

#### `mlir-pdll-lsp-server`

The various `.pdll` language features require the
[`mlir-pdll-lsp-server` language server](https://mlir.llvm.org/docs/Tools/MLIRLSP/#pdll-lsp-language-server--mlir-pdll-lsp-server).
If `mlir-pdll-lsp-server` is not found within your workspace path, you must
specify the path of the server via the `mlir.pdll_server_path` setting. The path
of the server may be absolute or relative within your workspace.

#### Project setup

To properly understand and interact with `.pdll` files, the language server must
understand how the project is built (compile flags).
[`pdll_compile_commands.yml` files](https://mlir.llvm.org/docs/Tools/MLIRLSP/#compilation-database)
related to your project should be provided to ensure files are properly
processed. These files can usually be generated by the build system, and the
server will attempt to find them within your `build/` directory. If not
available in or a unique location, additional `pdll_compile_commands.yml` files
may be specified via the `mlir.pdll_compilation_databases` setting. The paths of
these databases may be absolute or relative within your workspace.

## `.td` - TableGen files:

The MLIR extension adds language support for the
[TableGen language](https://llvm.org/docs/TableGen/ProgRef.html).

### Features

- Syntax highlighting for `.td` files and `tablegen` markdown blocks
- go-to-definition and cross references
- Types and documentation on hover

#### Diagnostics

The language server actively runs verification as you type, showing any
generated diagnostics in-place.

![IMG](https://mlir.llvm.org/tblgen-lsp-server/diagnostics.png)

#### Cross-references

Cross references allow for navigating the code base.

##### Find definition

Jump to the definition of a symbol under the cursor:

![IMG](https://mlir.llvm.org/tblgen-lsp-server/goto_def.gif)

##### Find references

Show all references of the symbol under the cursor.

![IMG](https://mlir.llvm.org/tblgen-lsp-server/find_references.gif)

#### Hover

Hover over a symbol to see more information about it, such as its type,
documentation, and more.

![IMG](https://mlir.llvm.org/tblgen-lsp-server/hover_def.png)

Hovering over an overridden field will also show you information such as
documentation from the base value:

![IMG](https://mlir.llvm.org/tblgen-lsp-server/hover_field.png)

### Setup

#### `tblgen-lsp-server`

The various `.td` language features require the
[`tblgen-lsp-server` language server](https://mlir.llvm.org/docs/Tools/MLIRLSP/#tablegen-lsp-language-server--tblgen-lsp-server).
If `tblgen-lsp-server` is not found within your workspace path, you must specify
the path of the server via the `mlir.tablegen_server_path` setting. The path of
the server may be absolute or relative within your workspace.

#### Project setup

To properly understand and interact with `.td` files, the language server must
understand how the project is built (compile flags).
[`tablegen_compile_commands.yml` files](https://mlir.llvm.org/docs/Tools/MLIRLSP/#compilation-database-1)
related to your project should be provided to ensure files are properly
processed. These files can usually be generated by the build system, and the
server will attempt to find them within your `build/` directory. If not
available in or a unique location, additional `tablegen_compile_commands.yml`
files may be specified via the `mlir.tablegen_compilation_databases` setting.
The paths of these databases may be absolute or relative within your workspace.

## Contributing

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

