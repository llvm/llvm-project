; REQUIRES: asserts
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -stats 2>&1 | \
; RUN:   grep "Number of machine instrs printed" | grep 12

define i16 @Trans16Bit(i32 %srcA, i32 %srcB, i32 %alpha) {
	%tmp1 = shl i32 %srcA, 15		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 32505856		; <i32> [#uses=1]
	%tmp4 = and i32 %srcA, 31775		; <i32> [#uses=1]
	%tmp5 = or i32 %tmp2, %tmp4		; <i32> [#uses=1]
	%tmp7 = shl i32 %srcB, 15		; <i32> [#uses=1]
	%tmp8 = and i32 %tmp7, 32505856		; <i32> [#uses=1]
	%tmp10 = and i32 %srcB, 31775		; <i32> [#uses=1]
	%tmp11 = or i32 %tmp8, %tmp10		; <i32> [#uses=1]
	%tmp14 = mul i32 %tmp5, %alpha		; <i32> [#uses=1]
	%tmp16 = sub i32 32, %alpha		; <i32> [#uses=1]
	%tmp18 = mul i32 %tmp11, %tmp16		; <i32> [#uses=1]
	%tmp19 = add i32 %tmp18, %tmp14		; <i32> [#uses=2]
	%tmp21 = lshr i32 %tmp19, 5		; <i32> [#uses=1]
	%tmp21.upgrd.1 = trunc i32 %tmp21 to i16		; <i16> [#uses=1]
	%tmp = and i16 %tmp21.upgrd.1, 31775		; <i16> [#uses=1]
	%tmp23 = lshr i32 %tmp19, 20		; <i32> [#uses=1]
	%tmp23.upgrd.2 = trunc i32 %tmp23 to i16		; <i16> [#uses=1]
	%tmp24 = and i16 %tmp23.upgrd.2, 992		; <i16> [#uses=1]
	%tmp25 = or i16 %tmp, %tmp24		; <i16> [#uses=1]
	ret i16 %tmp25
}
