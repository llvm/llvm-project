; RUN: llc < %s -mtriple=i386-apple-darwin -mattr=+sse2 -mcpu=nehalem | FileCheck %s
; rdar://7434544

define <2 x i64> @t2() nounwind {
entry:
; CHECK: t2:
; CHECK: pshufd	$85, (%esp), %xmm0
  %array = alloca [8 x float], align 4
  %arrayidx = getelementptr inbounds [8 x float]* %array, i32 0, i32 1
  %tmp2 = load float* %arrayidx
  %vecinit = insertelement <4 x float> undef, float %tmp2, i32 0
  %vecinit5 = insertelement <4 x float> %vecinit, float %tmp2, i32 1
  %vecinit7 = insertelement <4 x float> %vecinit5, float %tmp2, i32 2
  %vecinit9 = insertelement <4 x float> %vecinit7, float %tmp2, i32 3
  %0 = bitcast <4 x float> %vecinit9 to <2 x i64>
  ret <2 x i64> %0
}
