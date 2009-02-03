; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -o %t -f
; RUN: grep movlhps   %t | count 1
; RUN: grep movq      %t | count 1
; RUN: grep movsd     %t | count 1

define <4 x float> @test1(float %a, float %b) nounwind {
	%tmp = insertelement <4 x float> zeroinitializer, float %a, i32 0		; <<4 x float>> [#uses=1]
	%tmp6 = insertelement <4 x float> %tmp, float 0.000000e+00, i32 1		; <<4 x float>> [#uses=1]
	%tmp8 = insertelement <4 x float> %tmp6, float %b, i32 2		; <<4 x float>> [#uses=1]
	%tmp9 = insertelement <4 x float> %tmp8, float 0.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	ret <4 x float> %tmp9
}

define <4 x float> @test2(float %a, float %b) nounwind {
	%tmp = insertelement <4 x float> zeroinitializer, float %a, i32 0		; <<4 x float>> [#uses=1]
	%tmp7 = insertelement <4 x float> %tmp, float %b, i32 1		; <<4 x float>> [#uses=1]
	%tmp8 = insertelement <4 x float> %tmp7, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	%tmp9 = insertelement <4 x float> %tmp8, float 0.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	ret <4 x float> %tmp9
}

define <2 x i64> @test3(i32 %a, i32 %b) nounwind {
	%tmp = insertelement <4 x i32> zeroinitializer, i32 %a, i32 0		; <<4 x i32>> [#uses=1]
	%tmp6 = insertelement <4 x i32> %tmp, i32 %b, i32 1		; <<4 x i32>> [#uses=1]
	%tmp8 = insertelement <4 x i32> %tmp6, i32 0, i32 2		; <<4 x i32>> [#uses=1]
	%tmp10 = insertelement <4 x i32> %tmp8, i32 0, i32 3		; <<4 x i32>> [#uses=1]
	%tmp11 = bitcast <4 x i32> %tmp10 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp11
}
