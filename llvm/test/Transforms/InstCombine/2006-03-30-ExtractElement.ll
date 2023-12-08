; RUN: opt < %s -passes=instcombine -disable-output

define float @test(<4 x float> %V) {
        %V2 = insertelement <4 x float> %V, float 1.000000e+00, i32 3           ; <<4 x float>> [#uses=1]
        %R = extractelement <4 x float> %V2, i32 2              ; <float> [#uses=1]
        ret float %R
}

