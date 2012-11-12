; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep frest    %t1.s | count 2 
; RUN: grep -w fi    %t1.s | count 2 
; RUN: grep -w fm    %t1.s | count 2
; RUN: grep fma      %t1.s | count 2 
; RUN: grep fnms     %t1.s | count 4
; RUN: grep cgti     %t1.s | count 2
; RUN: grep selb     %t1.s | count 2
;
; This file includes standard floating point arithmetic instructions
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define float @fdiv32(float %arg1, float %arg2) {
        %A = fdiv float %arg1,  %arg2
        ret float %A
}

define <4 x float> @fdiv_v4f32(<4 x float> %arg1, <4 x float> %arg2) {
        %A = fdiv <4 x float> %arg1,  %arg2
        ret <4 x float> %A
}
