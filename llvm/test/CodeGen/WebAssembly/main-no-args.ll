; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test main functions with alternate signatures.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define i32 @main() {
  ret i32 0
}

; CHECK-LABEL: __original_main:
; CHECK-NEXT: .functype __original_main () -> (i32)
; CHECK-NEXT: i32.const 0
; CHECK-NEXT: end_function

; CHECK-LABEL: main:
; CHECK-NEXT: .functype main (i32, i32) -> (i32)
; CHECK:      call __original_main
