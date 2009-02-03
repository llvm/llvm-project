; This test checks to make sure phi nodes are updated properly
;
; RUN: llvm-as < %s | opt -tailduplicate -disable-output

define i32 @test(i1 %c, i32 %X, i32 %Y) {
	br label %L
L:		; preds = %F, %0
	%A = add i32 %X, %Y		; <i32> [#uses=1]
	br i1 %c, label %T, label %F
F:		; preds = %L
	br i1 %c, label %L, label %T
T:		; preds = %F, %L
	%V = phi i32 [ %A, %L ], [ 0, %F ]		; <i32> [#uses=1]
	ret i32 %V
}

