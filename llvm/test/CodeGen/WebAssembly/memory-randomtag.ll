; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test memory tagging builtin on wasm64

target triple = "wasm64-unknown-unknown"

; CHECK-LABEL: memory_randomtag:
; CHECK-NEXT: .functype memory_randomtag (i64) -> (i64)
; CHECK-NEXT: local.get	0
; CHECK-NEXT: memory.randomtag	
; CHECK-NEXT: end_function
define ptr @memory_randomtag(ptr %p) {
  %1 = call ptr @llvm.wasm.memory.randomtag(ptr %p)
  ret ptr %1
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.wasm.memory.randomtag(ptr)
