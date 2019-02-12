; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test main functions with alternate signatures.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @main()

define i32 @foo() {
  %t = call i32 @main()
  ret i32 %t
}

; CHECK-LABEL: foo:
; CHECK-NEXT:    .functype foo () -> (i32)
; CHECK-NEXT:    call __original_main@FUNCTION
; CHECK-NEXT:    end_function
