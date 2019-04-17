; This testcases makes sure that mem2reg can handle unreachable blocks.
; RUN: opt < %s -mem2reg

define i32 @test() {
	%X = alloca i32		; <i32*> [#uses=2]
	store i32 6, i32* %X
	br label %Loop
Loop:		; preds = %EndOfLoop, %0
	store i32 5, i32* %X
	br label %EndOfLoop
Unreachable:		; No predecessors!
	br label %EndOfLoop
EndOfLoop:		; preds = %Unreachable, %Loop
	br label %Loop
}

