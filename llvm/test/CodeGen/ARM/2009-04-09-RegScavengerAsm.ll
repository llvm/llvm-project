; RUN: llc < %s -march=arm
; PR3954

define void @foo(...) nounwind {
entry:
	%rr = alloca i32		; <i32*> [#uses=2]
	%0 = load i32* %rr		; <i32> [#uses=1]
	%1 = call i32 asm "nop", "=r,0"(i32 %0) nounwind		; <i32> [#uses=1]
	store i32 %1, i32* %rr
	br label %return

return:		; preds = %entry
	ret void
}
