; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   grep mul | count 2

define <4 x float> @test(<4 x float> %V) {
        %Y = mul <4 x float> %V, < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >                ; <<4 x float>> [#uses=1]
        %Z = mul <4 x float> %Y, < float 1.000000e+00, float 2.000000e+05, float -3.000000e+00, float 4.000000e+00 >               ; <<4 x float>> [#uses=1]
        ret <4 x float> %Z
}

