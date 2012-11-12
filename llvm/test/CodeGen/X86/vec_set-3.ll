; RUN: llc < %s -march=x86 -mattr=+sse2 -mcpu=penryn -o %t
; RUN: grep pshufd %t | count 2

define <4 x float> @test(float %a) nounwind {
        %tmp = insertelement <4 x float> zeroinitializer, float %a, i32 1               ; <<4 x float>> [#uses=1]
        %tmp5 = insertelement <4 x float> %tmp, float 0.000000e+00, i32 2               ; <<4 x float>> [#uses=1]
        %tmp6 = insertelement <4 x float> %tmp5, float 0.000000e+00, i32 3              ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp6
}

define <2 x i64> @test2(i32 %a) nounwind {
        %tmp7 = insertelement <4 x i32> zeroinitializer, i32 %a, i32 2          ; <<4 x i32>> [#uses=1]
        %tmp9 = insertelement <4 x i32> %tmp7, i32 0, i32 3             ; <<4 x i32>> [#uses=1]
        %tmp10 = bitcast <4 x i32> %tmp9 to <2 x i64>           ; <<2 x i64>> [#uses=1]
        ret <2 x i64> %tmp10
}

