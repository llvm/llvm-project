; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   grep {add float}

        %V = type <4 x float>

define float @test(%V %A, %V %B, float %f) {
        %C = insertelement %V %A, float %f, i32 0               ; <%V> [#uses=1]
        %D = add %V %C, %B              ; <%V> [#uses=1]
        %E = extractelement %V %D, i32 0                ; <float> [#uses=1]
        ret float %E
}

