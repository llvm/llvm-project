; RUN: llc -mtriple=xtensa -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

define i64 @lshl_64(i64 %x, i64 %y) nounwind {
; CHECK-LABEL: lshl_64:
; CHECK:         ssl a4
; CHECK-NEXT:    src a3, a3, a2
; CHECK-NEXT:    addi a8, a4, -32
; CHECK-NEXT:    ssl a8
; CHECK-NEXT:    sll a10, a2
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    blt a8, a9, .LBB0_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a3, a10, a10
; CHECK-NEXT:  .LBB0_2:
; CHECK-NEXT:    ssl a4
; CHECK-NEXT:    sll a2, a2
; CHECK-NEXT:    blt a8, a9, .LBB0_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a2, a9, a9
; CHECK-NEXT:  .LBB0_4:
; CHECK-NEXT:    ret
  %c = shl i64 %x, %y
  ret i64 %c
}

define i64 @lshr_64(i64 %x, i64 %y) nounwind {
; CHECK-LABEL: lshr_64:
; CHECK:         ssr a4
; CHECK-NEXT:    src a2, a3, a2
; CHECK-NEXT:    addi a8, a4, -32
; CHECK-NEXT:    ssr a8
; CHECK-NEXT:    srl a10, a3
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    blt a8, a9, .LBB1_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a10, a10
; CHECK-NEXT:  .LBB1_2:
; CHECK-NEXT:    ssr a4
; CHECK-NEXT:    srl a3, a3
; CHECK-NEXT:    blt a8, a9, .LBB1_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a3, a9, a9
; CHECK-NEXT:  .LBB1_4:
; CHECK-NEXT:    ret
  %c = lshr i64 %x, %y
  ret i64 %c
}

define i64 @ashr_64(i64 %x, i64 %y) nounwind {
; CHECK-LABEL: ashr_64:
; CHECK:         ssr a4
; CHECK-NEXT:    src a2, a3, a2
; CHECK-NEXT:    addi a9, a4, -32
; CHECK-NEXT:    ssr a9
; CHECK-NEXT:    sra a8, a3
; CHECK-NEXT:    movi a10, 0
; CHECK-NEXT:    blt a9, a10, .LBB2_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB2_2:
; CHECK-NEXT:    ssr a4
; CHECK-NEXT:    sra a8, a3
; CHECK-NEXT:    blt a9, a10, .LBB2_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    srai a8, a3, 31
; CHECK-NEXT:  .LBB2_4:
; CHECK-NEXT:    or a3, a8, a8
; CHECK-NEXT:    ret
  %c = ashr i64 %x, %y
  ret i64 %c
}
