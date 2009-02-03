; This testcase tests for a problem where LICM hoists 
; potentially trapping instructions when they are not guaranteed to execute.
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | %prcontext "IfUnEqual" 2 | grep div 

@X = global i32 0		; <i32*> [#uses=1]

declare void @foo()

define i32 @test(i1 %c) {
	%A = load i32* @X		; <i32> [#uses=2]
	br label %Loop
Loop:		; preds = %LoopTail, %0
	call void @foo( )
	br i1 %c, label %LoopTail, label %IfUnEqual
IfUnEqual:		; preds = %Loop
	%B1 = sdiv i32 4, %A		; <i32> [#uses=1]
	br label %LoopTail
LoopTail:		; preds = %IfUnEqual, %Loop
	%B = phi i32 [ 0, %Loop ], [ %B1, %IfUnEqual ]		; <i32> [#uses=1]
	br i1 %c, label %Loop, label %Out
Out:		; preds = %LoopTail
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}

