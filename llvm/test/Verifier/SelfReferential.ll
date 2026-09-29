; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: Only PHI nodes may reference their own value

; Test that self referential instructions are not allowed

define void @test() {
	%A = add i32 %A, 0		; <i32> [#uses=1]
	ret void
}

