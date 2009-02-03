; RUN: not llvm-as %s -f |& grep {not verify as correct}
; PR1042

define i32 @foo() {
	br i1 false, label %L1, label %L2
L1:		; preds = %0
	%A = invoke i32 @foo( )
			to label %L unwind label %L		; <i32> [#uses=1]
L2:		; preds = %0
	br label %L
L:		; preds = %L2, %L1, %L1
	ret i32 %A
}

