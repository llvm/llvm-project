; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

; This testcase failed to work because two variable sized allocas confused the
; local register allocator.

define i32 @main(i32 %X) {
	%A = alloca i32, i32 %X		; <ptr> [#uses=0]
	%B = alloca float, i32 %X		; <ptr> [#uses=0]
	ret i32 0
}

