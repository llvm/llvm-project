; RUN: llc -verify-machineinstrs < %s -march=ppc32 -mcpu=g5 | grep test:
; RUN: llc -verify-machineinstrs < %s -march=ppc32 -mcpu=g5 | not grep vperm

define void @test(<4 x float>* %tmp2.i) {
        %tmp2.i.upgrd.1 = load <4 x float>, <4 x float>* %tmp2.i             ; <<4 x float>> [#uses=4]
        %xFloat0.48 = extractelement <4 x float> %tmp2.i.upgrd.1, i32 0      ; <float> [#uses=1]
        %inFloat0.49 = insertelement <4 x float> undef, float %xFloat0.48, i32 0              ; <<4 x float>> [#uses=1]
        %xFloat1.50 = extractelement <4 x float> %tmp2.i.upgrd.1, i32 1      ; <float> [#uses=1]
        %inFloat1.52 = insertelement <4 x float> %inFloat0.49, float %xFloat1.50, i32 1               ; <<4 x float>> [#uses=1]
        %xFloat2.53 = extractelement <4 x float> %tmp2.i.upgrd.1, i32 2      ; <float> [#uses=1]
        %inFloat2.55 = insertelement <4 x float> %inFloat1.52, float %xFloat2.53, i32 2               ; <<4 x float>> [#uses=1]
        %xFloat3.56 = extractelement <4 x float> %tmp2.i.upgrd.1, i32 3      ; <float> [#uses=1]
        %inFloat3.58 = insertelement <4 x float> %inFloat2.55, float %xFloat3.56, i32 3               ; <<4 x float>> [#uses=1]
        store <4 x float> %inFloat3.58, <4 x float>* %tmp2.i
        ret void
}

