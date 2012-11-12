; RUN: llc < %s -march=x86 -mattr=sse | grep movaps
; Test that the load is NOT folded into the intrinsic, which would zero the top
; elts of the loaded vector.

target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8.7.2"

define <4 x float> @test(<4 x float> %A, <4 x float>* %B) nounwind {
        %BV = load <4 x float>* %B              ; <<4 x float>> [#uses=1]
        %tmp28 = tail call <4 x float> @llvm.x86.sse.sub.ss( <4 x float> %A, <4 x float> %BV )       ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp28
}

declare <4 x float> @llvm.x86.sse.sub.ss(<4 x float>, <4 x float>)

