; Test that hoisting is disabled for pointers of different types...
;
; RUN: opt < %s -passes=licm

define void @test(ptr %P) {
	br label %Loop
Loop:		; preds = %Loop, %0
	store i32 5, ptr %P
	store i8 4, ptr %P
	br i1 true, label %Loop, label %Out
Out:		; preds = %Loop
	ret void
}

