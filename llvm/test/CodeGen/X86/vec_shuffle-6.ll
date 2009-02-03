; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -o %t -f
; RUN: grep movapd %t | count 1
; RUN: grep movaps %t | count 1
; RUN: grep movups %t | count 2

target triple = "i686-apple-darwin"
@x = global [4 x i32] [ i32 1, i32 2, i32 3, i32 4 ]		; <[4 x i32]*> [#uses=4]

define <2 x i64> @test1() {
	%tmp = load i32* getelementptr ([4 x i32]* @x, i32 0, i32 0)		; <i32> [#uses=1]
	%tmp3 = load i32* getelementptr ([4 x i32]* @x, i32 0, i32 1)		; <i32> [#uses=1]
	%tmp5 = load i32* getelementptr ([4 x i32]* @x, i32 0, i32 2)		; <i32> [#uses=1]
	%tmp7 = load i32* getelementptr ([4 x i32]* @x, i32 0, i32 3)		; <i32> [#uses=1]
	%tmp.upgrd.1 = insertelement <4 x i32> undef, i32 %tmp, i32 0		; <<4 x i32>> [#uses=1]
	%tmp13 = insertelement <4 x i32> %tmp.upgrd.1, i32 %tmp3, i32 1		; <<4 x i32>> [#uses=1]
	%tmp14 = insertelement <4 x i32> %tmp13, i32 %tmp5, i32 2		; <<4 x i32>> [#uses=1]
	%tmp15 = insertelement <4 x i32> %tmp14, i32 %tmp7, i32 3		; <<4 x i32>> [#uses=1]
	%tmp16 = bitcast <4 x i32> %tmp15 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp16
}

define <4 x float> @test2(i32 %dummy, float %a, float %b, float %c, float %d) {
	%tmp = insertelement <4 x float> undef, float %a, i32 0		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp, float %b, i32 1		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float %c, i32 2		; <<4 x float>> [#uses=1]
	%tmp13 = insertelement <4 x float> %tmp12, float %d, i32 3		; <<4 x float>> [#uses=1]
	ret <4 x float> %tmp13
}

define <4 x float> @test3(float %a, float %b, float %c, float %d) {
	%tmp = insertelement <4 x float> undef, float %a, i32 0		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp, float %b, i32 1		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float %c, i32 2		; <<4 x float>> [#uses=1]
	%tmp13 = insertelement <4 x float> %tmp12, float %d, i32 3		; <<4 x float>> [#uses=1]
	ret <4 x float> %tmp13
}

define <2 x double> @test4(double %a, double %b) {
	%tmp = insertelement <2 x double> undef, double %a, i32 0		; <<2 x double>> [#uses=1]
	%tmp7 = insertelement <2 x double> %tmp, double %b, i32 1		; <<2 x double>> [#uses=1]
	ret <2 x double> %tmp7
}
