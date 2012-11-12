; RUN: llc < %s -march=x86 -mattr=+sse2 | grep movq

define <4 x float> @a(<4 x float> %a, float* nocapture %p) nounwind readonly {
entry:
	%tmp1 = load float* %p
	%vecins = insertelement <4 x float> undef, float %tmp1, i32 0
	%add.ptr = getelementptr float* %p, i32 1
	%tmp5 = load float* %add.ptr
	%vecins7 = insertelement <4 x float> %vecins, float %tmp5, i32 1
	ret <4 x float> %vecins7
}

