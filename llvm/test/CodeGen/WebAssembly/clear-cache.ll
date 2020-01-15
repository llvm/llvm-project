; RUN: not --crash llc < %s -asm-verbose=false 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: LLVM ERROR: llvm.clear_cache is not supported on wasm
define void @clear_cache(i8* %begin, i8* %end) {
entry:
  call void @llvm.clear_cache(i8* %begin, i8* %end)
  ret void
}

declare void @llvm.clear_cache(i8*, i8*)
