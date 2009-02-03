; This testcase tests to make sure a trapping instruction is hoisted when
; it is guaranteed to execute.
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | %prcontext "test" 2 | grep div

@X = global i32 0		; <i32*> [#uses=1]

declare void @foo(i32)

define i32 @test(i1 %c) {
	%A = load i32* @X		; <i32> [#uses=2]
	br label %Loop
Loop:		; preds = %Loop, %0
        ;; Should have hoisted this div!
	%B = sdiv i32 4, %A		; <i32> [#uses=2]
	call void @foo( i32 %B )
	br i1 %c, label %Loop, label %Out
Out:		; preds = %Loop
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}
