; RUN: llc -mtriple=arm-eabi %s -o /dev/null
; PR3954

define void @foo(...) nounwind {
entry:
	%rr = alloca i32		; <ptr> [#uses=2]
	%0 = load i32, ptr %rr		; <i32> [#uses=1]
	%1 = call i32 asm "nop", "=r,0"(i32 %0) nounwind		; <i32> [#uses=1]
	store i32 %1, ptr %rr
	br label %return

return:		; preds = %entry
	ret void
}
