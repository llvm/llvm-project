; RUN: not llc -mtriple wasm32-unknown-unknown-wasm %s 2>&1 | FileCheck %s

$f = comdat any
@f = global i32 0, comdat
; CHECK: LLVM ERROR: WebAssembly doesn't support COMDATs, 'f' cannot be lowered.
