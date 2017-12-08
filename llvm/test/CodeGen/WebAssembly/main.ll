; RUN: llc < %s -asm-verbose=false -wasm-temporary-workarounds=false | FileCheck %s

; Test main functions with alternate signatures.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

define void @main() {
  ret void
}

; CHECK-LABEL: .L__original_main:
; CHECK-NEXT: end_function

; CHECK-LABEL: main:
; CHECK-NEXT: .param i32, i32
; CHECK-NEXT: .result i32
; CHECK:      call .L__original_main@FUNCTION
