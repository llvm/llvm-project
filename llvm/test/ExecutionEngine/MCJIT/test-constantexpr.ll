; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

; This tests to make sure that we can evaluate weird constant expressions

@A = global i32 5		; <ptr> [#uses=1]

define i32 @main() {
	%A = or i1 false, ptrtoint (ptr @A to i1)
	ret i32 0
}

