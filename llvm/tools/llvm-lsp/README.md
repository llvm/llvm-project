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

This LSP server is built to the Language Server Protocol Specification 3.17. It provides several standard and custom features to enhance the development experience.

---

### Standard Capabilities

The server supports the following standard LSP capabilities:

* `textDocumentSync.openClose`: Synchronizes document content with the server.
* `referencesProvider`: Finds all references to a symbol.
* `codeActionProvider`: Provides code actions, such as quick fixes and refactorings. This server uses it to provide CFG views.
* `documentsSymbolProvider`: Provides a tree of document symbols, enables breadcrums navigation in the editor.

---

### Custom Methods

In addition to the standard capabilities, the server exposes several custom methods tailored for LLVM development.

#### `llvm/getCfg`

This method generates and returns an SVG representation of the Control Flow Graph (CFG) for the function at a specified position.

**Parameters**

```ts
interface GetCfgParams {
    /**
     * The URI of the file for which the CFG is requested.
     */
    uri: string;
    /**
     * The cursor's position. The CFG is generated for the function where the cursor is located.
     */
    position: Position;
}
```

**Response**

```ts
interface CFG {
    /**
     * URI of the SVG file containing the CFG.
     */
    uri: string;
    /**
     * The ID of the node corresponding to the basic block where the cursor was located.
     */
    node_id: string;
    /**
     * The name of the function for which the CFG was generated.
     */
    function: string;
}
```

---

#### `llvm/bbLocation`

This method retrieves the location of a basic block within the source code, identified by its node ID from a generated CFG.

**Parameters**

```ts
interface BbLocationParams {
    /**
     * The URI of the SVG file containing the CFG.
     */
    uri: string;
    /**
     * The ID of the node representing the basic block.
     */
    node_id: string;
}
```

**Response**

```ts
interface BbLocation {
    /**
     * The URI of the `.ll` file containing the basic block.
     */
    uri: string;
    /**
     * The range of the basic block corresponding to the node ID.
     */
    range: Range;
}
```
