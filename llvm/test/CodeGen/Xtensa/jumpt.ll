; RUN: llc -mtriple=xtensa -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

define void @switch_4_xtensa(i32 %in, ptr %out) nounwind {
; CHECK: .literal_position
; CHECK-NEXT:  .LCPI0_0, .LJTI0_0
; CHECK-LABEL: switch_4_xtensa:
; CHECK:       # %bb.0:
; CHECK-NEXT:  addi a9, a2, -1
; CHECK-NEXT:  movi a8, 3
; CHECK-NEXT:  bltu a8, a9, .LBB0_6
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:  l32r a10, .LCPI0_0
; CHECK-NEXT:  addx4 a9, a9, a10
; CHECK-NEXT:  l32i a9, a9, 0
; CHECK-NEXT:  jx a9
; CHECK:       ret

entry:
  switch i32 %in, label %exit [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
   i32 4, label %bb4
  ]
bb1:
  store i32 4, ptr %out
  br label %exit
bb2:
  store i32 3, ptr %out
  br label %exit
bb3:
  store i32 2, ptr %out
  br label %exit
bb4:
  store i32 1, ptr %out
  br label %exit
exit:
  ret void
}
