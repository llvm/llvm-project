; RUN: llc < %s -march=x86 -mattr=+sse2,-sse41 | grep movss | count 1
; RUN: llc < %s -march=x86 -mattr=+sse2,-sse41 | not grep pinsrw

define void @test(<4 x float>* %F, i32 %I) nounwind {
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=1]
	%f = sitofp i32 %I to float		; <float> [#uses=1]
	%tmp1 = insertelement <4 x float> %tmp, float %f, i32 0		; <<4 x float>> [#uses=2]
	%tmp18 = fadd <4 x float> %tmp1, %tmp1		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp18, <4 x float>* %F
	ret void
}

define void @test2(<4 x float>* %F, i32 %I, float %g) nounwind {
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=1]
	%f = sitofp i32 %I to float		; <float> [#uses=1]
	%tmp1 = insertelement <4 x float> %tmp, float %f, i32 2		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp1, <4 x float>* %F
	ret void
}
