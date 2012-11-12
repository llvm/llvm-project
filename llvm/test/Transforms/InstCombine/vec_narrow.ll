; RUN: opt < %s -instcombine -S | grep "fadd float"


define float @test(<4 x float> %A, <4 x float> %B, float %f) {
        %C = insertelement <4 x float> %A, float %f, i32 0               ; <%V> [#uses=1]
        %D = fadd <4 x float> %C, %B              ; <%V> [#uses=1]
        %E = extractelement <4 x float> %D, i32 0                ; <float> [#uses=1]
        ret float %E
}

