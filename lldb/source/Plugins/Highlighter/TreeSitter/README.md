# Tree-sitter Syntax Highlighting

This directory contains Highlighter plugins that use
[Tree-sitter](https://tree-sitter.github.io/tree-sitter/).

Each plugin contains a vendored copy of the corresponding grammar in the
`tree-sitter-<language>` sub-directory, consisting of the following files:

- `grammar.js`: the grammar used by the Tree-sitter command line tool to generate the parser.
- `highlights.scm`: the syntax highlight query.
- `scanner.c`: an optional scanner.
- `LICENSE`: the license for the grammar.

## Supported Languages

- Swift based on [swift-tree-sitter](https://github.com/tree-sitter/swift-tree-sitter) 0.9.0
