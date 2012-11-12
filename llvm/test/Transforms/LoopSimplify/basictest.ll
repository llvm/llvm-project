; RUN: opt < %s -loop-simplify

; This function should get a preheader inserted before BB3, that is jumped
; to by BB1 & BB2
;

define void @test() {
	br i1 true, label %BB1, label %BB2
BB1:		; preds = %0
	br label %BB3
BB2:		; preds = %0
	br label %BB3
BB3:		; preds = %BB3, %BB2, %BB1
	br label %BB3
}

