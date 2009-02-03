; This testcase checks to make sure the sinker does not cause problems with
; critical edges.

; RUN: llvm-as < %s | opt -licm | llvm-dis | %prcontext add 1 | grep Exit

define void @test() {
Entry:
	br i1 false, label %Loop, label %Exit
Loop:		; preds = %Loop, %Entry
	%X = add i32 0, 1		; <i32> [#uses=1]
	br i1 false, label %Loop, label %Exit
Exit:		; preds = %Loop, %Entry
	%Y = phi i32 [ 0, %Entry ], [ %X, %Loop ]		; <i32> [#uses=0]
	ret void
}

