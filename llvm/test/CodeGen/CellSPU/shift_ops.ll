; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep "shlh	"  %t1.s | count 10
; RUN: grep "shlhi	"  %t1.s | count 3
; RUN: grep "shl	"  %t1.s | count 10
; RUN: grep "shli	"  %t1.s | count 3
; RUN: grep "xshw	"  %t1.s | count 5
; RUN: grep "and	"  %t1.s | count 15
; RUN: grep "andi	"  %t1.s | count 4
; RUN: grep "rotmi	"  %t1.s | count 4
; RUN: grep "rotqmbyi	"  %t1.s | count 1
; RUN: grep "rotqmbii	"  %t1.s | count 2
; RUN: grep "rotqmby	"  %t1.s | count 1
; RUN: grep "rotqmbi	"  %t1.s | count 2
; RUN: grep "rotqbyi	"  %t1.s | count 1
; RUN: grep "rotqbii	"  %t1.s | count 2
; RUN: grep "rotqbybi	"  %t1.s | count 1
; RUN: grep "sfi	"  %t1.s | count 6
; RUN: cat %t1.s | FileCheck %s

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

; Shift left i16 via register, note that the second operand to shl is promoted
; to a 32-bit type:

define i16 @shlh_i16_1(i16 %arg1, i16 %arg2) {
        %A = shl i16 %arg1, %arg2
        ret i16 %A
}

define i16 @shlh_i16_2(i16 %arg1, i16 %arg2) {
        %A = shl i16 %arg2, %arg1
        ret i16 %A
}

define signext i16 @shlh_i16_3(i16 signext %arg1, i16 signext %arg2) {
        %A = shl i16 %arg1, %arg2
        ret i16 %A
}

define signext i16 @shlh_i16_4(i16 signext %arg1, i16 signext %arg2) {
        %A = shl i16 %arg2, %arg1
        ret i16 %A
}

define zeroext i16 @shlh_i16_5(i16 zeroext %arg1, i16 zeroext %arg2)  {
        %A = shl i16 %arg1, %arg2
        ret i16 %A
}

define zeroext i16 @shlh_i16_6(i16 zeroext %arg1, i16 zeroext %arg2) {
        %A = shl i16 %arg2, %arg1
        ret i16 %A
}

; Shift left i16 with immediate:
define i16 @shlhi_i16_1(i16 %arg1) {
        %A = shl i16 %arg1, 12
        ret i16 %A
}

; Should not generate anything other than the return, arg1 << 0 = arg1
define i16 @shlhi_i16_2(i16 %arg1) {
        %A = shl i16 %arg1, 0
        ret i16 %A
}

define i16 @shlhi_i16_3(i16 %arg1) {
        %A = shl i16 16383, %arg1
        ret i16 %A
}

; Should generate 0, 0 << arg1 = 0
define i16 @shlhi_i16_4(i16 %arg1) {
        %A = shl i16 0, %arg1
        ret i16 %A
}

define signext i16 @shlhi_i16_5(i16 signext %arg1)  {
        %A = shl i16 %arg1, 12
        ret i16 %A
}

; Should not generate anything other than the return, arg1 << 0 = arg1
define signext i16 @shlhi_i16_6(i16 signext %arg1) {
        %A = shl i16 %arg1, 0
        ret i16 %A
}

define signext i16 @shlhi_i16_7(i16 signext %arg1) {
        %A = shl i16 16383, %arg1
        ret i16 %A
}

; Should generate 0, 0 << arg1 = 0
define signext i16 @shlhi_i16_8(i16 signext %arg1)  {
        %A = shl i16 0, %arg1
        ret i16 %A
}

define zeroext i16 @shlhi_i16_9(i16 zeroext %arg1)  {
        %A = shl i16 %arg1, 12
        ret i16 %A
}

; Should not generate anything other than the return, arg1 << 0 = arg1
define zeroext i16 @shlhi_i16_10(i16 zeroext %arg1)  {
        %A = shl i16 %arg1, 0
        ret i16 %A
}

define zeroext i16 @shlhi_i16_11(i16 zeroext %arg1)  {
        %A = shl i16 16383, %arg1
        ret i16 %A
}

; Should generate 0, 0 << arg1 = 0
define zeroext i16 @shlhi_i16_12(i16 zeroext %arg1)  {
        %A = shl i16 0, %arg1
        ret i16 %A
}

; Shift left i32 via register, note that the second operand to shl is promoted
; to a 32-bit type:

define i32 @shl_i32_1(i32 %arg1, i32 %arg2) {
        %A = shl i32 %arg1, %arg2
        ret i32 %A
}

define i32 @shl_i32_2(i32 %arg1, i32 %arg2) {
        %A = shl i32 %arg2, %arg1
        ret i32 %A
}

define signext i32 @shl_i32_3(i32 signext %arg1, i32 signext %arg2)  {
        %A = shl i32 %arg1, %arg2
        ret i32 %A
}

define signext i32 @shl_i32_4(i32 signext %arg1, i32 signext %arg2)  {
        %A = shl i32 %arg2, %arg1
        ret i32 %A
}

define zeroext i32 @shl_i32_5(i32 zeroext %arg1, i32 zeroext %arg2)  {
        %A = shl i32 %arg1, %arg2
        ret i32 %A
}

define zeroext i32 @shl_i32_6(i32 zeroext %arg1, i32 zeroext %arg2)  {
        %A = shl i32 %arg2, %arg1
        ret i32 %A
}

; Shift left i32 with immediate:
define i32 @shli_i32_1(i32 %arg1) {
        %A = shl i32 %arg1, 12
        ret i32 %A
}

; Should not generate anything other than the return, arg1 << 0 = arg1
define i32 @shli_i32_2(i32 %arg1) {
        %A = shl i32 %arg1, 0
        ret i32 %A
}

define i32 @shli_i32_3(i32 %arg1) {
        %A = shl i32 16383, %arg1
        ret i32 %A
}

; Should generate 0, 0 << arg1 = 0
define i32 @shli_i32_4(i32 %arg1) {
        %A = shl i32 0, %arg1
        ret i32 %A
}

define signext i32 @shli_i32_5(i32 signext %arg1)  {
        %A = shl i32 %arg1, 12
        ret i32 %A
}

; Should not generate anything other than the return, arg1 << 0 = arg1
define signext i32 @shli_i32_6(i32 signext %arg1) {
        %A = shl i32 %arg1, 0
        ret i32 %A
}

define signext i32 @shli_i32_7(i32 signext %arg1)  {
        %A = shl i32 16383, %arg1
        ret i32 %A
}

; Should generate 0, 0 << arg1 = 0
define signext i32 @shli_i32_8(i32 signext %arg1) {
        %A = shl i32 0, %arg1
        ret i32 %A
}

define zeroext i32 @shli_i32_9(i32 zeroext %arg1)  {
        %A = shl i32 %arg1, 12
        ret i32 %A
}

; Should not generate anything other than the return, arg1 << 0 = arg1
define zeroext i32 @shli_i32_10(i32 zeroext %arg1)  {
        %A = shl i32 %arg1, 0
        ret i32 %A
}

define zeroext i32 @shli_i32_11(i32 zeroext %arg1) {
        %A = shl i32 16383, %arg1
        ret i32 %A
}

; Should generate 0, 0 << arg1 = 0
define zeroext i32 @shli_i32_12(i32 zeroext %arg1) {
        %A = shl i32 0, %arg1
        ret i32 %A
}

;; i64 shift left

define i64 @shl_i64_1(i64 %arg1) {
	%A = shl i64 %arg1, 9
	ret i64 %A
}

define i64 @shl_i64_2(i64 %arg1) {
	%A = shl i64 %arg1, 3
	ret i64 %A
}

define i64 @shl_i64_3(i64 %arg1, i32 %shift) {
	%1 = zext i32 %shift to i64
	%2 = shl i64 %arg1, %1
	ret i64 %2
}

;; i64 shift right logical (shift 0s from the right)

define i64 @lshr_i64_1(i64 %arg1) {
	%1 = lshr i64 %arg1, 9
	ret i64 %1
}

define i64 @lshr_i64_2(i64 %arg1) {
	%1 = lshr i64 %arg1, 3
	ret i64 %1
}

define i64 @lshr_i64_3(i64 %arg1, i32 %shift) {
	%1 = zext i32 %shift to i64
	%2 = lshr i64 %arg1, %1
	ret i64 %2
}

;; i64 shift right arithmetic (shift 1s from the right)

define i64 @ashr_i64_1(i64 %arg) {
	%1 = ashr i64 %arg, 9
	ret i64 %1
}

define i64 @ashr_i64_2(i64 %arg) {
	%1 = ashr i64 %arg, 3
	ret i64 %1
}

define i64 @ashr_i64_3(i64 %arg1, i32 %shift) {
	%1 = zext i32 %shift to i64
	%2 = ashr i64 %arg1, %1
	ret i64 %2
}

define i32 @hi32_i64(i64 %arg) {
	%1 = lshr i64 %arg, 32
	%2 = trunc i64 %1 to i32
	ret i32 %2
}

; some random tests
define i128 @test_lshr_i128( i128 %val ) {
 	;CHECK: test_lshr_i128
	;CHECK: sfi
	;CHECK: rotqmbi
	;CHECK: rotqmbybi
	;CHECK: bi $lr
	%rv = lshr i128 %val, 64
	ret i128 %rv
}

;Vector shifts
define <2 x i32> @shl_v2i32(<2 x i32> %val, <2 x i32> %sh) {
;CHECK: shl
;CHECK: bi $lr
	%rv = shl <2 x i32> %val, %sh
	ret <2 x i32> %rv
}

define <4 x i32> @shl_v4i32(<4 x i32> %val, <4 x i32> %sh) {
;CHECK: shl
;CHECK: bi $lr
	%rv = shl <4 x i32> %val, %sh
	ret <4 x i32> %rv
}

define <8 x i16> @shl_v8i16(<8 x i16> %val, <8 x i16> %sh) {
;CHECK: shlh
;CHECK: bi $lr
	%rv = shl <8 x i16> %val, %sh
	ret <8 x i16> %rv
}

define <4 x i32> @lshr_v4i32(<4 x i32> %val, <4 x i32> %sh) {
;CHECK: rotm
;CHECK: bi $lr
	%rv = lshr <4 x i32> %val, %sh
	ret <4 x i32> %rv
}

define <8 x i16> @lshr_v8i16(<8 x i16> %val, <8 x i16> %sh) {
;CHECK: sfhi
;CHECK: rothm
;CHECK: bi $lr
	%rv = lshr <8 x i16> %val, %sh
	ret <8 x i16> %rv
}

define <4 x i32> @ashr_v4i32(<4 x i32> %val, <4 x i32> %sh) {
;CHECK: rotma
;CHECK: bi $lr
	%rv = ashr <4 x i32> %val, %sh
	ret <4 x i32> %rv
}

define <8 x i16> @ashr_v8i16(<8 x i16> %val, <8 x i16> %sh) {
;CHECK: sfhi
;CHECK: rotmah
;CHECK: bi $lr
	%rv = ashr <8 x i16> %val, %sh
	ret <8 x i16> %rv
}

define <2 x i64> @special_const() {
  ret <2 x i64> <i64 4294967295, i64 4294967295>
}
