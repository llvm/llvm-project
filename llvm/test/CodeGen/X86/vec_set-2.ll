; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movss | count 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movd | count 1

define <4 x float> @test1(float %a) nounwind {
	%tmp = insertelement <4 x float> zeroinitializer, float %a, i32 0		; <<4 x float>> [#uses=1]
	%tmp5 = insertelement <4 x float> %tmp, float 0.000000e+00, i32 1		; <<4 x float>> [#uses=1]
	%tmp6 = insertelement <4 x float> %tmp5, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	%tmp7 = insertelement <4 x float> %tmp6, float 0.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	ret <4 x float> %tmp7
}

define <2 x i64> @test(i32 %a) nounwind {
	%tmp = insertelement <4 x i32> zeroinitializer, i32 %a, i32 0		; <<8 x i16>> [#uses=1]
	%tmp6 = insertelement <4 x i32> %tmp, i32 0, i32 1		; <<8 x i32>> [#uses=1]
	%tmp8 = insertelement <4 x i32> %tmp6, i32 0, i32 2		; <<8 x i32>> [#uses=1]
	%tmp10 = insertelement <4 x i32> %tmp8, i32 0, i32 3		; <<8 x i32>> [#uses=1]
	%tmp19 = bitcast <4 x i32> %tmp10 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp19
}
