; RUN: llc < %s -asm-verbose=false -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Test that switches are lowered correctly in the presence of an
; unreachable default branch target.

; CHECK-LABEL: foo:
; CHECK-NEXT:    .functype foo (i32) -> (i32)
; CHECK-NEXT:    block
; CHECK-NEXT:    block
; CHECK-NEXT:    local.get 0
; CHECK-NEXT:    br_table {0, 1, 0}
; CHECK-NEXT:  .LBB0_1:
; CHECK-NEXT:    end_block
; CHECK-NEXT:    i32.const 0
; CHECK-NEXT:    return
; CHECK-NEXT:  .LBB0_2:
; CHECK-NEXT:    end_block
; CHECK-NEXT:    i32.const 1
; CHECK-NEXT:    end_function
define i32 @foo(i32 %x) {
entry:
  switch i32 %x, label %unreachable [
    i32 0, label %bb0
    i32 1, label %bb1
  ]

bb0:
  ret i32 0

bb1:
  ret i32 1

unreachable:
  unreachable
}
