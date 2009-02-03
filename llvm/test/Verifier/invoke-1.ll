; RUN: not llvm-as < %s |& grep {not verify as correct}
; PR1042

define i32 @foo() {
	%A = invoke i32 @foo( )
			to label %L unwind label %L		; <i32> [#uses=1]
L:		; preds = %0, %0
	ret i32 %A
}

