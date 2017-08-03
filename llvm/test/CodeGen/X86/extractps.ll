; RUN: llc < %s -march=x86 -mcpu=penryn > %t
; RUN: not grep movd %t
; RUN: grep "movss	%xmm" %t | count 1
; RUN: grep "extractps	\$1, %xmm0, " %t | count 1
; PR2647

@0 = external global float, align 16         ; <float*>:0 [#uses=2]

define internal void @""() nounwind {
        load float, float* @0, align 16                ; <float>:1 [#uses=1]
        insertelement <4 x float> undef, float %1, i32 0                ; <<4 x float>>:2 [#uses=1]
        call <4 x float> @llvm.x86.sse.rsqrt.ss( <4 x float> %2 )              ; <<4 x float>>:3 [#uses=1]
        extractelement <4 x float> %3, i32 0            ; <float>:4 [#uses=1]
        store float %4, float* @0, align 16
        ret void
}
define internal void @""() nounwind {
        load float, float* @0, align 16                ; <float>:1 [#uses=1]
        insertelement <4 x float> undef, float %1, i32 1                ; <<4 x float>>:2 [#uses=1]
        call <4 x float> @llvm.x86.sse.rsqrt.ss( <4 x float> %2 )              ; <<4 x float>>:3 [#uses=1]
        extractelement <4 x float> %3, i32 1            ; <float>:4 [#uses=1]
        store float %4, float* @0, align 16
        ret void
}

declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone

