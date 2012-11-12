; RUN: opt < %s -prune-eh -S | not grep "ret i32"

declare void @noreturn() noreturn

define i32 @caller() {
	call void @noreturn( )
	ret i32 17
}

define i32 @caller2() {
	%T = call i32 @caller( )		; <i32> [#uses=1]
	ret i32 %T
}
