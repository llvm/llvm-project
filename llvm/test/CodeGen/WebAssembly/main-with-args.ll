; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that main function with expected signature is not wrapped

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define i32 @main(i32 %a, i8** %b) {
  ret i32 0
}

; CHECK-LABEL: main:
; CHECK-NEXT: .functype main (i32, i32) -> (i32)

; CHECK-NOT: __original_main:
