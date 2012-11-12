; RUN: llc < %s -mcpu=corei7 -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

define void @foo(<4 x float> %in, <4 x i8>* %out) {
  %t0 = fptosi <4 x float> %in to <4 x i32>
  %t1 = trunc <4 x i32> %t0 to <4 x i16>
  %t2 = shufflevector <4 x i16> %t1, <4 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %t3 = trunc <8 x i16> %t2 to <8 x i8>
  %t4 = shufflevector <8 x i8> %t3, <8 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %t5 = insertelement <4 x i8> %t4, i8 -1, i32 3
  store <4 x i8> %t5, <4 x i8>* %out
  ret void
; CHECK: foo
; CHECK: cvttps2dq
; CHECK-NOT: pextrd
; CHECK: pshufb
; CHECK: ret
}
