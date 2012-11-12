; RUN: opt < %s -instcombine -S | grep srem
; PR3439

define i32 @a(i32 %x) nounwind {
entry:
	%rem = srem i32 %x, 2
	%and = and i32 %rem, 2
	ret i32 %and
}
