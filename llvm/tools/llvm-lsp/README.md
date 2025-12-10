# LLVM LSP server

## Build

```bash
cmake -S llvm -B buildR -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
ninja -j 6 -C buildR llvm-lsp-server
```
Or
```bash
cmake -S llvm -B buildRA -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
ninja -j 6 -C buildRA llvm-lsp-server
```

## Features

This LSP server is built to the Language Server Protocol Specification 3.17. It provides several standard features to enhance the development experience.

---

### Standard Capabilities

The server supports the following standard LSP capabilities:

* `textDocumentSync.openClose`: Synchronizes document content with the server.
* `referencesProvider`: Finds all references to a symbol.
* `documentsSymbolProvider`: Provides a tree of document symbols, enables breadcrums navigation in the editor.
