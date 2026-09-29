; RUN: opt < %s -passes=sccp -disable-output
; PR4277

define i32 @main() nounwind {
entry:
	%0 = tail call signext i8 (...) @sin() nounwind
	ret i32 0
}

declare signext i8 @sin(...)
