; RUN: llc < %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Check that we do not crash when attempting to lower away
; global_dtors without a definition.

@llvm.global_dtors = external global [2 x { i32, void ()*, i8* }]
