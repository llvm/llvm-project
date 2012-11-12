; RUN: llc < %s -march=cellspu -enable-unsafe-fp-math > %t1.s
; RUN: grep fa %t1.s | count 2
; RUN: grep fs %t1.s | count 2
; RUN: grep fm %t1.s | count 6
; RUN: grep fma %t1.s | count 2
; RUN: grep fms %t1.s | count 2
; RUN: grep fnms %t1.s | count 3
;
; This file includes standard floating point arithmetic instructions
; NOTE fdiv is tested separately since it is a compound operation
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define float @fp_add(float %arg1, float %arg2) {
        %A = fadd float %arg1, %arg2     ; <float> [#uses=1]
        ret float %A
}

define <4 x float> @fp_add_vec(<4 x float> %arg1, <4 x float> %arg2) {
        %A = fadd <4 x float> %arg1, %arg2       ; <<4 x float>> [#uses=1]
        ret <4 x float> %A
}

define float @fp_sub(float %arg1, float %arg2) {
        %A = fsub float %arg1,  %arg2    ; <float> [#uses=1]
        ret float %A
}

define <4 x float> @fp_sub_vec(<4 x float> %arg1, <4 x float> %arg2) {
        %A = fsub <4 x float> %arg1,  %arg2      ; <<4 x float>> [#uses=1]
        ret <4 x float> %A
}

define float @fp_mul(float %arg1, float %arg2) {
        %A = fmul float %arg1,  %arg2    ; <float> [#uses=1]
        ret float %A
}

define <4 x float> @fp_mul_vec(<4 x float> %arg1, <4 x float> %arg2) {
        %A = fmul <4 x float> %arg1,  %arg2      ; <<4 x float>> [#uses=1]
        ret <4 x float> %A
}

define float @fp_mul_add(float %arg1, float %arg2, float %arg3) {
        %A = fmul float %arg1,  %arg2    ; <float> [#uses=1]
        %B = fadd float %A, %arg3        ; <float> [#uses=1]
        ret float %B
}

define <4 x float> @fp_mul_add_vec(<4 x float> %arg1, <4 x float> %arg2, <4 x float> %arg3) {
        %A = fmul <4 x float> %arg1,  %arg2      ; <<4 x float>> [#uses=1]
        %B = fadd <4 x float> %A, %arg3  ; <<4 x float>> [#uses=1]
        ret <4 x float> %B
}

define float @fp_mul_sub(float %arg1, float %arg2, float %arg3) {
        %A = fmul float %arg1,  %arg2    ; <float> [#uses=1]
        %B = fsub float %A, %arg3        ; <float> [#uses=1]
        ret float %B
}

define <4 x float> @fp_mul_sub_vec(<4 x float> %arg1, <4 x float> %arg2, <4 x float> %arg3) {
        %A = fmul <4 x float> %arg1,  %arg2      ; <<4 x float>> [#uses=1]
        %B = fsub <4 x float> %A, %arg3  ; <<4 x float>> [#uses=1]
        ret <4 x float> %B
}

; Test the straightforward way of getting fnms
; c - a * b
define float @fp_neg_mul_sub_1(float %arg1, float %arg2, float %arg3) {
        %A = fmul float %arg1,  %arg2
        %B = fsub float %arg3, %A
        ret float %B
}

; Test another way of getting fnms
; - ( a *b -c ) = c - a * b
define float @fp_neg_mul_sub_2(float %arg1, float %arg2, float %arg3) {
        %A = fmul float %arg1,  %arg2
        %B = fsub float %A, %arg3
        %C = fsub float -0.0, %B
        ret float %C
}

define <4 x float> @fp_neg_mul_sub_vec(<4 x float> %arg1, <4 x float> %arg2, <4 x float> %arg3) {
        %A = fmul <4 x float> %arg1,  %arg2
        %B = fsub <4 x float> %A, %arg3
        %D = fsub <4 x float> < float -0.0, float -0.0, float -0.0, float -0.0 >, %B
        ret <4 x float> %D
}
