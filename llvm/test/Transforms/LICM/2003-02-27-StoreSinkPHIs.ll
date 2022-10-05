; LICM is adding stores before phi nodes.  bad.

; RUN: opt < %s -licm

define i1 @test(i1 %c) {
; <label>:0
	br i1 %c, label %Loop, label %Out
Loop:		; preds = %Loop, %0
	store i32 0, ptr null
	br i1 %c, label %Loop, label %Out
Out:		; preds = %Loop, %0
	%X = phi i1 [ %c, %0 ], [ true, %Loop ]		; <i1> [#uses=1]
	ret i1 %X
}

