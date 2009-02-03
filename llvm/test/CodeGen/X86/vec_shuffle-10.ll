; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | \
; RUN:   grep unpcklps | count 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | \
; RUN:   grep unpckhps | count 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | \
; RUN:   not grep {sub.*esp}

define void @test(<4 x float>* %res, <4 x float>* %A, <4 x float>* %B) {
	%tmp = load <4 x float>* %B		; <<4 x float>> [#uses=2]
	%tmp3 = load <4 x float>* %A		; <<4 x float>> [#uses=2]
	%tmp.upgrd.1 = extractelement <4 x float> %tmp3, i32 0		; <float> [#uses=1]
	%tmp7 = extractelement <4 x float> %tmp, i32 0		; <float> [#uses=1]
	%tmp8 = extractelement <4 x float> %tmp3, i32 1		; <float> [#uses=1]
	%tmp9 = extractelement <4 x float> %tmp, i32 1		; <float> [#uses=1]
	%tmp10 = insertelement <4 x float> undef, float %tmp.upgrd.1, i32 0		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp10, float %tmp7, i32 1		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float %tmp8, i32 2		; <<4 x float>> [#uses=1]
	%tmp13 = insertelement <4 x float> %tmp12, float %tmp9, i32 3		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp13, <4 x float>* %res
	ret void
}

define void @test2(<4 x float> %X, <4 x float>* %res) {
	%tmp5 = shufflevector <4 x float> %X, <4 x float> undef, <4 x i32> < i32 2, i32 6, i32 3, i32 7 >		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp5, <4 x float>* %res
	ret void
}
