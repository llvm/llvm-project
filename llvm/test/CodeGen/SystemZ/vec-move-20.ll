; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a vector which is built with elements from two loads replicates the
; load with most elements having its value.

; CHECK:      vlef
; CHECK-NOT:  vlvgf

define void @update(ptr %src1, ptr %src2, ptr %dst) {
bb:
  %tmp = load i32, ptr %src1
  %tmp1 = load i32, ptr %src2
  %tmp2 = insertelement <4 x i32> undef, i32 %tmp, i32 0
  %tmp3 = insertelement <4 x i32> %tmp2, i32 %tmp1, i32 1
  %tmp4 = insertelement <4 x i32> %tmp3, i32 %tmp1, i32 2
  %tmp5 = insertelement <4 x i32> %tmp4, i32 %tmp1, i32 3
  store <4 x i32> %tmp5, ptr %dst
  ret void
}
