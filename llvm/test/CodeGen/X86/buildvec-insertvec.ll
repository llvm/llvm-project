; RUN: llc < %s -mcpu=corei7 -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

define void @foo(<3 x float> %in, <4 x i8>* nocapture %out) nounwind {
  %t0 = fptoui <3 x float> %in to <3 x i8>
  %t1 = shufflevector <3 x i8> %t0, <3 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  %t2 = insertelement <4 x i8> %t1, i8 -1, i32 3
  store <4 x i8> %t2, <4 x i8>* %out, align 4
  ret void
; CHECK: foo
; CHECK: cvttps2dq
; CHECK-NOT: pextrd
; CHECK: pinsrd
; CHECK-NEXT: pshufb
; CHECK: ret
}
