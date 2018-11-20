; RUN: llc < %s -asm-verbose=false -wasm-temporary-workarounds=false | FileCheck %s

; Test main functions with alternate signatures.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @main()

define void @foo() {
  call void @main()
  ret void
}

; CHECK-NOT:   __original_main
; CHECK-LABEL: foo:
; CHECK-NEXT:    .functype foo () -> ()
; CHECK-NEXT:    call main@FUNCTION
; CHECK-NEXT:    end_function
; CHECK-NOT:   __original_main
