; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep csflt %t1.s | count 5
; RUN: grep cuflt %t1.s | count 1
; RUN: grep xshw  %t1.s | count 2
; RUN: grep xsbh  %t1.s | count 1
; RUN: grep and   %t1.s | count 2
; RUN: grep andi  %t1.s | count 1
; RUN: grep ila   %t1.s | count 1

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define float @sitofp_i32(i32 %arg1) {
        %A = sitofp i32 %arg1 to float          ; <float> [#uses=1]
        ret float %A
}

define float @uitofp_u32(i32 %arg1) {
        %A = uitofp i32 %arg1 to float          ; <float> [#uses=1]
        ret float %A
}

define float @sitofp_i16(i16 %arg1) {
        %A = sitofp i16 %arg1 to float          ; <float> [#uses=1]
        ret float %A
}

define float @uitofp_i16(i16 %arg1) {
        %A = uitofp i16 %arg1 to float          ; <float> [#uses=1]
        ret float %A
}

define float @sitofp_i8(i8 %arg1) {
        %A = sitofp i8 %arg1 to float           ; <float> [#uses=1]
        ret float %A
}

define float @uitofp_i8(i8 %arg1) {
        %A = uitofp i8 %arg1 to float           ; <float> [#uses=1]
        ret float %A
}
