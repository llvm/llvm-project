; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep mpy     %t1.s | count 44
; RUN: grep mpyu    %t1.s | count 4
; RUN: grep mpyh    %t1.s | count 10
; RUN: grep mpyhh   %t1.s | count 2
; RUN: grep rotma   %t1.s | count 12
; RUN: grep rotmahi %t1.s | count 4
; RUN: grep and     %t1.s | count 2
; RUN: grep selb    %t1.s | count 6
; RUN: grep fsmbi   %t1.s | count 4
; RUN: grep shli    %t1.s | count 4
; RUN: grep shlhi   %t1.s | count 4
; RUN: grep ila     %t1.s | count 2
; RUN: grep xsbh    %t1.s | count 4
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

; 32-bit multiply instruction generation:
define <4 x i32> @mpy_v4i32_1(<4 x i32> %arg1, <4 x i32> %arg2) {
entry:
        %A = mul <4 x i32> %arg1, %arg2
        ret <4 x i32> %A
}

define <4 x i32> @mpy_v4i32_2(<4 x i32> %arg1, <4 x i32> %arg2) {
entry:
        %A = mul <4 x i32> %arg2, %arg1
        ret <4 x i32> %A
}

define <8 x i16> @mpy_v8i16_1(<8 x i16> %arg1, <8 x i16> %arg2) {
entry:
        %A = mul <8 x i16> %arg1, %arg2
        ret <8 x i16> %A
}

define <8 x i16> @mpy_v8i16_2(<8 x i16> %arg1, <8 x i16> %arg2) {
entry:
        %A = mul <8 x i16> %arg2, %arg1
        ret <8 x i16> %A
}

define <16 x i8> @mul_v16i8_1(<16 x i8> %arg1, <16 x i8> %arg2) {
entry:
        %A = mul <16 x i8> %arg2, %arg1
        ret <16 x i8> %A
}

define <16 x i8> @mul_v16i8_2(<16 x i8> %arg1, <16 x i8> %arg2) {
entry:
        %A = mul <16 x i8> %arg1, %arg2
        ret <16 x i8> %A
}

define i32 @mul_i32_1(i32 %arg1, i32 %arg2) {
entry:
        %A = mul i32 %arg2, %arg1
        ret i32 %A
}

define i32 @mul_i32_2(i32 %arg1, i32 %arg2) {
entry:
        %A = mul i32 %arg1, %arg2
        ret i32 %A
}

define i16 @mul_i16_1(i16 %arg1, i16 %arg2) {
entry:
        %A = mul i16 %arg2, %arg1
        ret i16 %A
}

define i16 @mul_i16_2(i16 %arg1, i16 %arg2) {
entry:
        %A = mul i16 %arg1, %arg2
        ret i16 %A
}

define i8 @mul_i8_1(i8 %arg1, i8 %arg2) {
entry:
        %A = mul i8 %arg2, %arg1
        ret i8 %A
}

define i8 @mul_i8_2(i8 %arg1, i8 %arg2) {
entry:
        %A = mul i8 %arg1, %arg2
        ret i8 %A
}
