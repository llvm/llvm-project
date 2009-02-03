; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   grep shufflevec | count 1
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   not grep insertelement
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   not grep extractelement
; END.

define <4 x float> @test(<4 x float> %tmp, <4 x float> %tmp1) {
        %tmp4 = extractelement <4 x float> %tmp, i32 1          ; <float> [#uses=1]
        %tmp2 = extractelement <4 x float> %tmp, i32 3          ; <float> [#uses=1]
        %tmp1.upgrd.1 = extractelement <4 x float> %tmp1, i32 0         ; <float> [#uses=1]
        %tmp128 = insertelement <4 x float> undef, float %tmp4, i32 0           ; <<4 x float>> [#uses=1]
        %tmp130 = insertelement <4 x float> %tmp128, float undef, i32 1         ; <<4 x float>> [#uses=1]
        %tmp132 = insertelement <4 x float> %tmp130, float %tmp2, i32 2         ; <<4 x float>> [#uses=1]
        %tmp134 = insertelement <4 x float> %tmp132, float %tmp1.upgrd.1, i32 3         ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp134
}

