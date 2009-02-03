; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -o %t -f
; RUN: grep punpck %t | count 2
; RUN: not grep pextrw %t

define <4 x i32> @test(i8** %ptr) {
entry:
	%tmp = load i8** %ptr		; <i8*> [#uses=1]
	%tmp.upgrd.1 = bitcast i8* %tmp to float*		; <float*> [#uses=1]
	%tmp.upgrd.2 = load float* %tmp.upgrd.1		; <float> [#uses=1]
	%tmp.upgrd.3 = insertelement <4 x float> undef, float %tmp.upgrd.2, i32 0		; <<4 x float>> [#uses=1]
	%tmp9 = insertelement <4 x float> %tmp.upgrd.3, float 0.000000e+00, i32 1		; <<4 x float>> [#uses=1]
	%tmp10 = insertelement <4 x float> %tmp9, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	%tmp21 = bitcast <4 x float> %tmp11 to <16 x i8>		; <<16 x i8>> [#uses=1]
	%tmp22 = shufflevector <16 x i8> %tmp21, <16 x i8> zeroinitializer, <16 x i32> < i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23 >		; <<16 x i8>> [#uses=1]
	%tmp31 = bitcast <16 x i8> %tmp22 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp.upgrd.4 = shufflevector <8 x i16> zeroinitializer, <8 x i16> %tmp31, <8 x i32> < i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11 >		; <<8 x i16>> [#uses=1]
	%tmp36 = bitcast <8 x i16> %tmp.upgrd.4 to <4 x i32>		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp36
}
