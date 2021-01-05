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

; CHECK-LABEL: split:
; CHECK:    .functype split (i32) -> ()
; CHECK:    block
; CHECK:    br_if 0
; CHECK:    block
; CHECK:    block
; CHECK:    br_table {1, 1, 0}
; CHECK: .LBB1_2
; CHECK:    end_block
; CHECK:    br_table {0, 0, 0, 0, 0, 0, 0, 0}
; CHECK: .LBB1_3
; CHECK:    end_block
; CHECK:    unreachable
; CHECK: .LBB1_4
; CHECK:    end_block
; CHECK:    end_function
define void @split(i8 %c) {
entry:
  switch i8 %c, label %sw.default [
    i8 114, label %return
    i8 103, label %sw.bb1
    i8 98, label %sw.bb2
    i8 97, label %sw.bb3
    i8 48, label %sw.bb4
    i8 49, label %sw.bb5
  ]

sw.bb1:
  unreachable

sw.bb2:
  unreachable

sw.bb3:
  unreachable

sw.bb4:
  unreachable

sw.bb5:
  unreachable

sw.default:
  unreachable

return:
  ret void
}
