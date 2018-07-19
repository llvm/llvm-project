; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that we don't emit anything declaring a non-executable stack,
; because wasm's stack is always non-executable.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; CHECK-NOT: .note.GNU-stack
