# LLVM Development VS Code Extension

This VS Code extension provides a comprehensive suite of tools for working with LLVM projects. It includes syntax highlighting, LIT integration, and a custom LLVM IR visualizer with LSP-backed commands.

---

## Features

### Syntax Highlighting
- **LLVM IR (.ll)** — syntax highlighting translated from `llvm/utils/vim/syntax/llvm.vim`
- **TableGen (.td)** — syntax highlighting from `llvm/utils/textmate`

### LIT Test Integration
- Pattern matchers for LIT test output (`$llvm-lit`, `$llvm-filecheck`)
- VS Code Tasks to run LIT on the current file:
  - `Terminal` → `Run Task` → `llvm-lit`

---

## Installation

### Prerequisites

```bash
sudo apt-get install nodejs-dev node-gyp npm
sudo npm install -g typescript npx vsce
```

### Install From Source

```bash
cd <extensions-installation-folder>
cp -r llvm/utils/vscode/llvm .
cd llvm
npm install
npm run vscode:prepublish
```

📌 `<extensions-installation-folder>` is OS dependent. See:  
https://code.visualstudio.com/docs/editor/extension-gallery#_where-are-extensions-installed

### Install From Package (.vsix)

1. Package the extension:  
   https://code.visualstudio.com/api/working-with-extensions/publishing-extension#usage  
2. Install the `.vsix`:  
   https://code.visualstudio.com/docs/editor/extension-gallery#_install-from-a-vsix

---

## Setup

Set the following in your VS Code settings:

```json
"cmake.buildDirectory": "<your-cmake-build-dir>",
"llvm.server_path": "<path-to-llvm-lsp-server>"
```

If `"llvm.server_path"` is not set, the extension will search for `llvm-lsp-server` in your system `PATH`.

Resources:
- [VS Code User Settings](https://code.visualstudio.com/docs/getstarted/settings)
- [CMake Tools: buildDirectory](https://vector-of-bool.github.io/docs/vscode-cmake-tools/settings.html#cmake-builddirectory)

---

## Development

### Build & Debug

```bash
npm install
npm run compile
```

Alternatively:
1. Open `package.json` in VS Code.
2. Click the `Debug` button next to any script under the `scripts` section.
3. Open `src/extension.ts`, press `F5` (Debug: Start Debugging).
4. A new window titled `[Extension Development Host]` will launch.

### Debugging LSP Communication

In the Extension Development Host:
- Open the **Output** pane (`Ctrl+Shift+U`)
- Select `llvm-lsp-server` from the dropdown
- Make sure the setting `llvm.trace.server` is set to `"messages"` or `"verbose"`

---

## Project Structure

### `package.json`
Metadata and configuration:
- Extension name, version, engines, activation events, etc.
- Contributions:
  - `languages`, `commands`, `menus`, `configuration`

### `src/` — TypeScript sources
- `extension.ts`
  - Entry point: creates `OutputChannel`, `LLVMContext`, and registers commands
- `llvmContext.ts`
  - `WorkspaceFolderContext`: manages `LanguageClient` per workspace
  - `LLVMContext`: manages lifecycle, subscriptions, and per-folder context
