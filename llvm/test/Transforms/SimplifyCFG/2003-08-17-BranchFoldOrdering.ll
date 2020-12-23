; This test checks to make sure that 'br X, Dest, Dest' is folded into
; 'br Dest'.  This can only happen after the 'Z' block is eliminated.  This is
; due to the fact that the SimplifyCFG function does not use
; the ConstantFoldTerminator function.

; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

; CHECK-NOT: br i1 %c2

declare void @noop()

define i32 @test(i1 %c1, i1 %c2) {
	call void @noop( )
	br i1 %c1, label %A, label %Y
A:		; preds = %0
	call void @noop( )
	br i1 %c2, label %Z, label %X
Z:		; preds = %A
	br label %X
X:		; preds = %Y, %Z, %A
	call void @noop( )
	ret i32 0
Y:		; preds = %0
	call void @noop( )
	br label %X
}

