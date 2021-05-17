; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

; This tests to make sure that we can evaluate weird constant expressions

@A = global i32 5		; <i32*> [#uses=1]
@B = global i32 6		; <i32*> [#uses=1]

define i32 @main() {
	%A = or i1 false, icmp slt (i32* @A, i32* @B)		; <i1> [#uses=0]
	ret i32 0
}

