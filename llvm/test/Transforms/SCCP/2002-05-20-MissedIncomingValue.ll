; This test shows a case where SCCP is incorrectly eliminating the PHI node
; because it thinks it has a constant 0 value, when it really doesn't.

; RUN: opt < %s -sccp -S | grep phi

define i32 @test(i32 %A, i1 %c) {
bb1:
	br label %BB2
BB2:		; preds = %BB4, %bb1
	%V = phi i32 [ 0, %bb1 ], [ %A, %BB4 ]		; <i32> [#uses=1]
	br label %BB3
BB3:		; preds = %BB2
	br i1 %c, label %BB4, label %BB5
BB4:		; preds = %BB3
	br label %BB2
BB5:		; preds = %BB3
	ret i32 %V
}

