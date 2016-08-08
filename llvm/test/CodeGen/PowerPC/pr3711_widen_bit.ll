; RUN: llc -verify-machineinstrs < %s -march=ppc32 -mcpu=g5

; Test that causes a abort in expanding a bit convert due to a missing support
; for widening.

define i32 @main() nounwind {
entry:
	br i1 icmp ne (i32 trunc (i64 bitcast (<2 x i32> <i32 2, i32 2> to i64) to i32), i32 2), label %bb, label %bb1

bb:		; preds = %entry
	tail call void @abort() noreturn nounwind
	unreachable

bb1:		; preds = %entry
	ret i32 0
}

declare void @abort() noreturn nounwind
