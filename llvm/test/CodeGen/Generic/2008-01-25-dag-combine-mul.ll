; RUN: llc < %s
; rdar://5707064

; XCore default subtarget does not support 8-byte alignment on stack.
; XFAIL: xcore

define i32 @f(i16* %pc) {
entry:
	%acc = alloca i64, align 8		; <i64*> [#uses=4]
	%tmp97 = load i64, i64* %acc, align 8		; <i64> [#uses=1]
	%tmp98 = and i64 %tmp97, 4294967295		; <i64> [#uses=1]
	%tmp99 = load i64, i64* null, align 8		; <i64> [#uses=1]
	%tmp100 = and i64 %tmp99, 4294967295		; <i64> [#uses=1]
	%tmp101 = mul i64 %tmp98, %tmp100		; <i64> [#uses=1]
	%tmp103 = lshr i64 %tmp101, 0		; <i64> [#uses=1]
	%tmp104 = load i64, i64* %acc, align 8		; <i64> [#uses=1]
	%.cast105 = zext i32 32 to i64		; <i64> [#uses=1]
	%tmp106 = lshr i64 %tmp104, %.cast105		; <i64> [#uses=1]
	%tmp107 = load i64, i64* null, align 8		; <i64> [#uses=1]
	%tmp108 = and i64 %tmp107, 4294967295		; <i64> [#uses=1]
	%tmp109 = mul i64 %tmp106, %tmp108		; <i64> [#uses=1]
	%tmp112 = add i64 %tmp109, 0		; <i64> [#uses=1]
	%tmp116 = add i64 %tmp112, 0		; <i64> [#uses=1]
	%tmp117 = add i64 %tmp103, %tmp116		; <i64> [#uses=1]
	%tmp118 = load i64, i64* %acc, align 8		; <i64> [#uses=1]
	%tmp120 = lshr i64 %tmp118, 0		; <i64> [#uses=1]
	%tmp121 = load i64, i64* null, align 8		; <i64> [#uses=1]
	%tmp123 = lshr i64 %tmp121, 0		; <i64> [#uses=1]
	%tmp124 = mul i64 %tmp120, %tmp123		; <i64> [#uses=1]
	%tmp126 = shl i64 %tmp124, 0		; <i64> [#uses=1]
	%tmp127 = add i64 %tmp117, %tmp126		; <i64> [#uses=1]
	store i64 %tmp127, i64* %acc, align 8
	ret i32 0
}
