; RUN: llc < %s -march=x86 -mattr=+sse,-sse2 | FileCheck %s
; PR2484

define <4 x float> @f4523(<4 x float> %a,<4 x float> %b) nounwind {
entry:
; CHECK: shufps $-28, %xmm
%shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4,i32
5,i32 2,i32 3>
ret <4 x float> %shuffle
}
