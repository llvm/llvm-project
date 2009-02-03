; Test that hoisting is disabled for pointers of different types...
;
; RUN: llvm-as < %s | opt -licm

define void @test(i32* %P) {
	br label %Loop
Loop:		; preds = %Loop, %0
	store i32 5, i32* %P
	%P2 = bitcast i32* %P to i8*		; <i8*> [#uses=1]
	store i8 4, i8* %P2
	br i1 true, label %Loop, label %Out
Out:		; preds = %Loop
	ret void
}

