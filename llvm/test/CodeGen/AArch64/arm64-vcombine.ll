; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

; LowerCONCAT_VECTORS() was reversing the order of two parts.
; rdar://11558157
; rdar://11559553
define <16 x i8> @test(<16 x i8> %q0, <16 x i8> %q1, i8* nocapture %dest) nounwind {
entry:
; CHECK-LABEL: test:
; CHECK: ins.d v0[1], v1[0]
  %0 = bitcast <16 x i8> %q0 to <2 x i64>
  %shuffle.i = shufflevector <2 x i64> %0, <2 x i64> undef, <1 x i32> zeroinitializer
  %1 = bitcast <16 x i8> %q1 to <2 x i64>
  %shuffle.i4 = shufflevector <2 x i64> %1, <2 x i64> undef, <1 x i32> zeroinitializer
  %shuffle.i3 = shufflevector <1 x i64> %shuffle.i, <1 x i64> %shuffle.i4, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i3 to <16 x i8>
  ret <16 x i8> %2
}
