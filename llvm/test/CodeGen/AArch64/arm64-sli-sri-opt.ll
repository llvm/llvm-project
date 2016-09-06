; RUN: llc < %s -aarch64-shift-insert-generation=true -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define void @testLeftGood(<16 x i8> %src1, <16 x i8> %src2, <16 x i8>* %dest) nounwind {
; CHECK-LABEL: testLeftGood:
; CHECK: sli.16b v0, v1, #3
  %and.i = and <16 x i8> %src1, <i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252>
  %vshl_n = shl <16 x i8> %src2, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  %result = or <16 x i8> %and.i, %vshl_n
  store <16 x i8> %result, <16 x i8>* %dest, align 16
  ret void
}

define void @testLeftBad(<16 x i8> %src1, <16 x i8> %src2, <16 x i8>* %dest) nounwind {
; CHECK-LABEL: testLeftBad:
; CHECK-NOT: sli
  %and.i = and <16 x i8> %src1, <i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165>
  %vshl_n = shl <16 x i8> %src2, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %result = or <16 x i8> %and.i, %vshl_n
  store <16 x i8> %result, <16 x i8>* %dest, align 16
  ret void
}

define void @testRightGood(<16 x i8> %src1, <16 x i8> %src2, <16 x i8>* %dest) nounwind {
; CHECK-LABEL: testRightGood:
; CHECK: sri.16b v0, v1, #3
  %and.i = and <16 x i8> %src1, <i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252, i8 252>
  %vshl_n = lshr <16 x i8> %src2, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  %result = or <16 x i8> %and.i, %vshl_n
  store <16 x i8> %result, <16 x i8>* %dest, align 16
  ret void
}

define void @testRightBad(<16 x i8> %src1, <16 x i8> %src2, <16 x i8>* %dest) nounwind {
; CHECK-LABEL: testRightBad:
; CHECK-NOT: sri
  %and.i = and <16 x i8> %src1, <i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165, i8 165>
  %vshl_n = lshr <16 x i8> %src2, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %result = or <16 x i8> %and.i, %vshl_n
  store <16 x i8> %result, <16 x i8>* %dest, align 16
  ret void
}
