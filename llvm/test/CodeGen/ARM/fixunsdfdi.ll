; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o /dev/null
; RUN: llc -mtriple=arm-eabi -mattr=vfp2 %s -o - | FileCheck %s

define hidden i64 @__fixunsdfdi(double %x) nounwind readnone {
entry:
	%x14 = bitcast double %x to i64		; <i64> [#uses=1]
	br i1 true, label %bb3, label %bb10

bb3:		; preds = %entry
	br i1 true, label %bb5, label %bb7

bb5:		; preds = %bb3
	%u.in.mask = and i64 %x14, -4294967296		; <i64> [#uses=1]
	%.ins = or i64 0, %u.in.mask		; <i64> [#uses=1]
	%0 = bitcast i64 %.ins to double		; <double> [#uses=1]
	%1 = fsub double %x, %0		; <double> [#uses=1]
	%2 = fptosi double %1 to i32		; <i32> [#uses=1]
	%3 = add i32 %2, 0		; <i32> [#uses=1]
	%4 = zext i32 %3 to i64		; <i64> [#uses=1]
	%5 = shl i64 %4, 32		; <i64> [#uses=1]
	%6 = or i64 %5, 0		; <i64> [#uses=1]
	ret i64 %6

bb7:		; preds = %bb3
	ret i64 0

bb10:		; preds = %entry
	ret i64 0
}

; CHECK-NOT: vstr.64

