; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -o %t -f
; RUN: grep movss    %t | count 1
; RUN: grep movups   %t | count 1
; RUN: grep shufps   %t | count 1

define <4 x float> @test(float %a, float %b, float %c) nounwind {
        %tmp = insertelement <4 x float> zeroinitializer, float %a, i32 1               ; <<4 x float>> [#uses=1]
        %tmp8 = insertelement <4 x float> %tmp, float %b, i32 2         ; <<4 x float>> [#uses=1]
        %tmp10 = insertelement <4 x float> %tmp8, float %c, i32 3               ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp10
}

