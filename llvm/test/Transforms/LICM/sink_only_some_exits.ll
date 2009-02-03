; This testcase checks to make sure we can sink values which are only live on
; some exits out of the loop, and that we can do so without breaking dominator
; info.
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | \
; RUN:   %prcontext add 1 | grep exit2:

define i32 @test(i1 %C1, i1 %C2, i32* %P, i32* %Q) {
Entry:
	br label %Loop
Loop:		; preds = %Cont, %Entry
	br i1 %C1, label %Cont, label %exit1
Cont:		; preds = %Loop
	%X = load i32* %P		; <i32> [#uses=2]
	store i32 %X, i32* %Q
	%V = add i32 %X, 1		; <i32> [#uses=1]
	br i1 %C2, label %Loop, label %exit2
exit1:		; preds = %Loop
	ret i32 0
exit2:		; preds = %Cont
	ret i32 %V
}

