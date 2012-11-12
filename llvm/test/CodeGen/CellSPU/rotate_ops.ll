; RUN: llc < %s -march=cellspu -o %t1.s
; RUN: grep rot          %t1.s | count 86
; RUN: grep roth         %t1.s | count 8
; RUN: grep roti.*5      %t1.s | count 1
; RUN: grep roti.*27     %t1.s | count 1
; RUN: grep rothi.*5      %t1.s | count 2
; RUN: grep rothi.*11     %t1.s | count 1
; RUN: grep rothi.*,.3    %t1.s | count 1
; RUN: grep andhi        %t1.s | count 4
; RUN: grep shlhi        %t1.s | count 4
; RUN: cat %t1.s | FileCheck %s

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

; Vector rotates are not currently supported in gcc or llvm assembly. These are
; not tested.

; 32-bit rotates:
define i32 @rotl32_1a(i32 %arg1, i8 %arg2) {
        %tmp1 = zext i8 %arg2 to i32    ; <i32> [#uses=1]
        %B = shl i32 %arg1, %tmp1       ; <i32> [#uses=1]
        %arg22 = sub i8 32, %arg2       ; <i8> [#uses=1]
        %tmp2 = zext i8 %arg22 to i32   ; <i32> [#uses=1]
        %C = lshr i32 %arg1, %tmp2      ; <i32> [#uses=1]
        %D = or i32 %B, %C              ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @rotl32_1b(i32 %arg1, i16 %arg2) {
        %tmp1 = zext i16 %arg2 to i32   ; <i32> [#uses=1]
        %B = shl i32 %arg1, %tmp1       ; <i32> [#uses=1]
        %arg22 = sub i16 32, %arg2      ; <i8> [#uses=1]
        %tmp2 = zext i16 %arg22 to i32  ; <i32> [#uses=1]
        %C = lshr i32 %arg1, %tmp2      ; <i32> [#uses=1]
        %D = or i32 %B, %C              ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @rotl32_2(i32 %arg1, i32 %arg2) {
        %B = shl i32 %arg1, %arg2       ; <i32> [#uses=1]
        %tmp1 = sub i32 32, %arg2       ; <i32> [#uses=1]
        %C = lshr i32 %arg1, %tmp1      ; <i32> [#uses=1]
        %D = or i32 %B, %C              ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @rotl32_3(i32 %arg1, i32 %arg2) {
        %tmp1 = sub i32 32, %arg2       ; <i32> [#uses=1]
        %B = shl i32 %arg1, %arg2       ; <i32> [#uses=1]
        %C = lshr i32 %arg1, %tmp1      ; <i32> [#uses=1]
        %D = or i32 %B, %C              ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @rotl32_4(i32 %arg1, i32 %arg2) {
        %tmp1 = sub i32 32, %arg2       ; <i32> [#uses=1]
        %C = lshr i32 %arg1, %tmp1      ; <i32> [#uses=1]
        %B = shl i32 %arg1, %arg2       ; <i32> [#uses=1]
        %D = or i32 %B, %C              ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @rotr32_1(i32 %A, i8 %Amt) {
        %tmp1 = zext i8 %Amt to i32     ; <i32> [#uses=1]
        %B = lshr i32 %A, %tmp1         ; <i32> [#uses=1]
        %Amt2 = sub i8 32, %Amt         ; <i8> [#uses=1]
        %tmp2 = zext i8 %Amt2 to i32    ; <i32> [#uses=1]
        %C = shl i32 %A, %tmp2          ; <i32> [#uses=1]
        %D = or i32 %B, %C              ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @rotr32_2(i32 %A, i8 %Amt) {
        %Amt2 = sub i8 32, %Amt         ; <i8> [#uses=1]
        %tmp1 = zext i8 %Amt to i32     ; <i32> [#uses=1]
        %B = lshr i32 %A, %tmp1         ; <i32> [#uses=1]
        %tmp2 = zext i8 %Amt2 to i32    ; <i32> [#uses=1]
        %C = shl i32 %A, %tmp2          ; <i32> [#uses=1]
        %D = or i32 %B, %C              ; <i32> [#uses=1]
        ret i32 %D
}

; Rotate left with immediate
define i32 @rotli32(i32 %A) {
        %B = shl i32 %A, 5              ; <i32> [#uses=1]
        %C = lshr i32 %A, 27            ; <i32> [#uses=1]
        %D = or i32 %B, %C              ; <i32> [#uses=1]
        ret i32 %D
}

; Rotate right with immediate
define i32 @rotri32(i32 %A) {
        %B = lshr i32 %A, 5             ; <i32> [#uses=1]
        %C = shl i32 %A, 27             ; <i32> [#uses=1]
        %D = or i32 %B, %C              ; <i32> [#uses=1]
        ret i32 %D
}

; 16-bit rotates:
define i16 @rotr16_1(i16 %arg1, i8 %arg) {
        %tmp1 = zext i8 %arg to i16             ; <i16> [#uses=1]
        %B = lshr i16 %arg1, %tmp1              ; <i16> [#uses=1]
        %arg2 = sub i8 16, %arg                 ; <i8> [#uses=1]
        %tmp2 = zext i8 %arg2 to i16            ; <i16> [#uses=1]
        %C = shl i16 %arg1, %tmp2               ; <i16> [#uses=1]
        %D = or i16 %B, %C                      ; <i16> [#uses=1]
        ret i16 %D
}

define i16 @rotr16_2(i16 %arg1, i16 %arg) {
        %B = lshr i16 %arg1, %arg       ; <i16> [#uses=1]
        %tmp1 = sub i16 16, %arg        ; <i16> [#uses=1]
        %C = shl i16 %arg1, %tmp1       ; <i16> [#uses=1]
        %D = or i16 %B, %C              ; <i16> [#uses=1]
        ret i16 %D
}

define i16 @rotli16(i16 %A) {
        %B = shl i16 %A, 5              ; <i16> [#uses=1]
        %C = lshr i16 %A, 11            ; <i16> [#uses=1]
        %D = or i16 %B, %C              ; <i16> [#uses=1]
        ret i16 %D
}

define i16 @rotri16(i16 %A) {
        %B = lshr i16 %A, 5             ; <i16> [#uses=1]
        %C = shl i16 %A, 11             ; <i16> [#uses=1]
        %D = or i16 %B, %C              ; <i16> [#uses=1]
        ret i16 %D
}

define i8 @rotl8(i8 %A, i8 %Amt) {
        %B = shl i8 %A, %Amt            ; <i8> [#uses=1]
        %Amt2 = sub i8 8, %Amt          ; <i8> [#uses=1]
        %C = lshr i8 %A, %Amt2          ; <i8> [#uses=1]
        %D = or i8 %B, %C               ; <i8> [#uses=1]
        ret i8 %D
}

define i8 @rotr8(i8 %A, i8 %Amt) {
        %B = lshr i8 %A, %Amt           ; <i8> [#uses=1]
        %Amt2 = sub i8 8, %Amt          ; <i8> [#uses=1]
        %C = shl i8 %A, %Amt2           ; <i8> [#uses=1]
        %D = or i8 %B, %C               ; <i8> [#uses=1]
        ret i8 %D
}

define i8 @rotli8(i8 %A) {
        %B = shl i8 %A, 5               ; <i8> [#uses=1]
        %C = lshr i8 %A, 3              ; <i8> [#uses=1]
        %D = or i8 %B, %C               ; <i8> [#uses=1]
        ret i8 %D
}

define i8 @rotri8(i8 %A) {
        %B = lshr i8 %A, 5              ; <i8> [#uses=1]
        %C = shl i8 %A, 3               ; <i8> [#uses=1]
        %D = or i8 %B, %C               ; <i8> [#uses=1]
        ret i8 %D
}

define <2 x float> @test1(<4 x float> %param )
{
; CHECK: test1
; CHECK: shufb
  %el = extractelement <4 x float> %param, i32 1
  %vec1 = insertelement <1 x float> undef, float %el, i32 0
  %rv = shufflevector <1 x float> %vec1, <1 x float> undef, <2 x i32><i32 0,i32 0>
; CHECK: bi $lr
  ret <2 x float> %rv
} 
