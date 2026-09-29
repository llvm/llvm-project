; This testcase tests for a problem where LICM hoists loads out of a loop
; despite the fact that calls to unknown functions may modify what is being
; loaded from.  Basically if the load gets hoisted, the subtract gets turned
; into a constant zero.
;
; RUN: opt < %s -passes='loop-mssa(licm),gvn,instcombine' -S | grep load

@X = global i32 7		; <ptr> [#uses=2]

declare void @foo()

define i32 @test(i1 %c) {
	%A = load i32, ptr @X		; <i32> [#uses=1]
	br label %Loop
Loop:		; preds = %Loop, %0
	call void @foo( )
        ;; Should not hoist this load!
	%B = load i32, ptr @X		; <i32> [#uses=1]
	br i1 %c, label %Loop, label %Out
Out:		; preds = %Loop
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}
