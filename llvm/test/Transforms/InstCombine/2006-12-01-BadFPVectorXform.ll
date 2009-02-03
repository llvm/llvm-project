; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep sub
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep add

define <4 x float> @test(<4 x float> %tmp26, <4 x float> %tmp53) {
        ; (X+Y)-Y != X for fp vectors.
        %tmp64 = add <4 x float> %tmp26, %tmp53         ; <<4 x float>> [#uses=1]
        %tmp75 = sub <4 x float> %tmp64, %tmp53         ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp75
}
