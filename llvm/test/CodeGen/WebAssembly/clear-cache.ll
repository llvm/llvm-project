; RUN: not --crash llc < %s -asm-verbose=false 2>&1 | FileCheck %s

target triple = "wasm32-unknown-unknown"

; CHECK: LLVM ERROR: llvm.clear_cache is not supported on wasm
define void @clear_cache(ptr %begin, ptr %end) {
entry:
  call void @llvm.clear_cache(ptr %begin, ptr %end)
  ret void
}

declare void @llvm.clear_cache(ptr, ptr)
