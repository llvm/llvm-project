; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep eqv  %t1.s | count 18
; RUN: grep xshw %t1.s | count 6
; RUN: grep xsbh %t1.s | count 3
; RUN: grep andi %t1.s | count 3

; Test the 'eqv' instruction, whose boolean expression is:
; (a & b) | (~a & ~b), which simplifies to
; (a & b) | ~(a | b)
; Alternatively, a ^ ~b, which the compiler will also match.

; ModuleID = 'eqv.bc'
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define <4 x i32> @equiv_v4i32_1(<4 x i32> %arg1, <4 x i32> %arg2) {
        %A = and <4 x i32> %arg1, %arg2
        %B = or <4 x i32> %arg1, %arg2
        %Bnot = xor <4 x i32> %B, < i32 -1, i32 -1, i32 -1, i32 -1 >
        %C = or <4 x i32> %A, %Bnot
        ret <4 x i32> %C
}

define <4 x i32> @equiv_v4i32_2(<4 x i32> %arg1, <4 x i32> %arg2) {
        %B = or <4 x i32> %arg1, %arg2          ; <<4 x i32>> [#uses=1]
        %Bnot = xor <4 x i32> %B, < i32 -1, i32 -1, i32 -1, i32 -1 >            ; <<4 x i32>> [#uses=1]
        %A = and <4 x i32> %arg1, %arg2         ; <<4 x i32>> [#uses=1]
        %C = or <4 x i32> %A, %Bnot             ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %C
}

define <4 x i32> @equiv_v4i32_3(<4 x i32> %arg1, <4 x i32> %arg2) {
        %B = or <4 x i32> %arg1, %arg2          ; <<4 x i32>> [#uses=1]
        %A = and <4 x i32> %arg1, %arg2         ; <<4 x i32>> [#uses=1]
        %Bnot = xor <4 x i32> %B, < i32 -1, i32 -1, i32 -1, i32 -1 >            ; <<4 x i32>> [#uses=1]
        %C = or <4 x i32> %A, %Bnot             ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %C
}

define <4 x i32> @equiv_v4i32_4(<4 x i32> %arg1, <4 x i32> %arg2) {
        %arg2not = xor <4 x i32> %arg2, < i32 -1, i32 -1, i32 -1, i32 -1 >
        %C = xor <4 x i32> %arg1, %arg2not
        ret <4 x i32> %C
}

define i32 @equiv_i32_1(i32 %arg1, i32 %arg2) {
        %A = and i32 %arg1, %arg2               ; <i32> [#uses=1]
        %B = or i32 %arg1, %arg2                ; <i32> [#uses=1]
        %Bnot = xor i32 %B, -1                  ; <i32> [#uses=1]
        %C = or i32 %A, %Bnot                   ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @equiv_i32_2(i32 %arg1, i32 %arg2) {
        %B = or i32 %arg1, %arg2                ; <i32> [#uses=1]
        %Bnot = xor i32 %B, -1                  ; <i32> [#uses=1]
        %A = and i32 %arg1, %arg2               ; <i32> [#uses=1]
        %C = or i32 %A, %Bnot                   ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @equiv_i32_3(i32 %arg1, i32 %arg2) {
        %B = or i32 %arg1, %arg2                ; <i32> [#uses=1]
        %A = and i32 %arg1, %arg2               ; <i32> [#uses=1]
        %Bnot = xor i32 %B, -1                  ; <i32> [#uses=1]
        %C = or i32 %A, %Bnot                   ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @equiv_i32_4(i32 %arg1, i32 %arg2) {
        %arg2not = xor i32 %arg2, -1
        %C = xor i32 %arg1, %arg2not
        ret i32 %C
}

define i32 @equiv_i32_5(i32 %arg1, i32 %arg2) {
        %arg1not = xor i32 %arg1, -1
        %C = xor i32 %arg2, %arg1not
        ret i32 %C
}

define signext i16 @equiv_i16_1(i16 signext %arg1, i16 signext %arg2)  {
        %A = and i16 %arg1, %arg2               ; <i16> [#uses=1]
        %B = or i16 %arg1, %arg2                ; <i16> [#uses=1]
        %Bnot = xor i16 %B, -1                  ; <i16> [#uses=1]
        %C = or i16 %A, %Bnot                   ; <i16> [#uses=1]
        ret i16 %C
}

define signext i16 @equiv_i16_2(i16 signext %arg1, i16 signext %arg2) {
        %B = or i16 %arg1, %arg2                ; <i16> [#uses=1]
        %Bnot = xor i16 %B, -1                  ; <i16> [#uses=1]
        %A = and i16 %arg1, %arg2               ; <i16> [#uses=1]
        %C = or i16 %A, %Bnot                   ; <i16> [#uses=1]
        ret i16 %C
}

define signext i16 @equiv_i16_3(i16 signext %arg1, i16 signext %arg2)  {
        %B = or i16 %arg1, %arg2                ; <i16> [#uses=1]
        %A = and i16 %arg1, %arg2               ; <i16> [#uses=1]
        %Bnot = xor i16 %B, -1                  ; <i16> [#uses=1]
        %C = or i16 %A, %Bnot                   ; <i16> [#uses=1]
        ret i16 %C
}

define signext i8 @equiv_i8_1(i8 signext %arg1, i8 signext %arg2)  {
        %A = and i8 %arg1, %arg2                ; <i8> [#uses=1]
        %B = or i8 %arg1, %arg2         ; <i8> [#uses=1]
        %Bnot = xor i8 %B, -1                   ; <i8> [#uses=1]
        %C = or i8 %A, %Bnot                    ; <i8> [#uses=1]
        ret i8 %C
}

define signext i8 @equiv_i8_2(i8 signext %arg1, i8 signext %arg2)  {
        %B = or i8 %arg1, %arg2         ; <i8> [#uses=1]
        %Bnot = xor i8 %B, -1                   ; <i8> [#uses=1]
        %A = and i8 %arg1, %arg2                ; <i8> [#uses=1]
        %C = or i8 %A, %Bnot                    ; <i8> [#uses=1]
        ret i8 %C
}

define signext i8 @equiv_i8_3(i8 signext %arg1, i8 signext %arg2)  {
        %B = or i8 %arg1, %arg2         ; <i8> [#uses=1]
        %A = and i8 %arg1, %arg2                ; <i8> [#uses=1]
        %Bnot = xor i8 %B, -1                   ; <i8> [#uses=1]
        %C = or i8 %A, %Bnot                    ; <i8> [#uses=1]
        ret i8 %C
}

define zeroext i8 @equiv_u8_1(i8 zeroext %arg1, i8 zeroext %arg2)  {
        %A = and i8 %arg1, %arg2                ; <i8> [#uses=1]
        %B = or i8 %arg1, %arg2         ; <i8> [#uses=1]
        %Bnot = xor i8 %B, -1                   ; <i8> [#uses=1]
        %C = or i8 %A, %Bnot                    ; <i8> [#uses=1]
        ret i8 %C
}

define zeroext i8 @equiv_u8_2(i8 zeroext %arg1, i8 zeroext %arg2)  {
        %B = or i8 %arg1, %arg2         ; <i8> [#uses=1]
        %Bnot = xor i8 %B, -1                   ; <i8> [#uses=1]
        %A = and i8 %arg1, %arg2                ; <i8> [#uses=1]
        %C = or i8 %A, %Bnot                    ; <i8> [#uses=1]
        ret i8 %C
}

define zeroext i8 @equiv_u8_3(i8 zeroext %arg1, i8 zeroext %arg2)  {
        %B = or i8 %arg1, %arg2         ; <i8> [#uses=1]
        %A = and i8 %arg1, %arg2                ; <i8> [#uses=1]
        %Bnot = xor i8 %B, -1                   ; <i8> [#uses=1]
        %C = or i8 %A, %Bnot                    ; <i8> [#uses=1]
        ret i8 %C
}
