; RUN: opt < %s -sccp -S | grep "ret i32 1"

; This function definitely returns 1, even if we don't know the direction
; of the branch.

define i32 @foo() {
	br i1 undef, label %T, label %T
T:		; preds = %0, %0
	%X = add i32 0, 1		; <i32> [#uses=1]
	ret i32 %X
}

