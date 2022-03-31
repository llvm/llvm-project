; RUN: opt < %s -passes=instcombine -S | grep " sub " | count 1
; PR2047

define i32 @l(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
	%b.neg = sub i32 0, %b		; <i32> [#uses=1]
	%c.neg = sub i32 0, %c		; <i32> [#uses=1]
	%sub4 = add i32 %c.neg, %b.neg		; <i32> [#uses=1]
	%sub6 = add i32 %sub4, %d		; <i32> [#uses=1]
	ret i32 %sub6
}
