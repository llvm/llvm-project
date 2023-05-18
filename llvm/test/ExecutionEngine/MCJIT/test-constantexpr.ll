; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

; This tests to make sure that we can evaluate weird constant expressions

@A = global i32 5		; <ptr> [#uses=1]
@B = global i32 6		; <ptr> [#uses=1]

define i32 @main() {
	%A = or i1 false, icmp slt (ptr @A, ptr @B)		; <i1> [#uses=0]
	ret i32 0
}

