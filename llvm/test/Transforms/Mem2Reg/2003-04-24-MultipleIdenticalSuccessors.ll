; Mem2reg used to only add one incoming value to a PHI node, even if it had
; multiple incoming edges from a block.
;
; RUN: opt < %s -mem2reg -disable-output

define i32 @test(i1 %c1, i1 %c2) {
	%X = alloca i32		; <i32*> [#uses=2]
	br i1 %c1, label %Exit, label %B2
B2:		; preds = %0
	store i32 2, i32* %X
	br i1 %c2, label %Exit, label %Exit
Exit:		; preds = %B2, %B2, %0
	%Y = load i32* %X		; <i32> [#uses=1]
	ret i32 %Y
}

