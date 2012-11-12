; RUN: opt < %s -instcombine -S | grep sub
; RUN: opt < %s -instcombine -S | grep add

define <4 x float> @test(<4 x float> %tmp26, <4 x float> %tmp53) {
        ; (X+Y)-Y != X for fp vectors.
        %tmp64 = fadd <4 x float> %tmp26, %tmp53         ; <<4 x float>> [#uses=1]
        %tmp75 = fsub <4 x float> %tmp64, %tmp53         ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp75
}
