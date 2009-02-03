; The PHI cannot be eliminated from this testcase, SCCP is mishandling invoke's!
; RUN: llvm-as < %s | opt -sccp | llvm-dis | grep phi

declare void @foo()

define i32 @test(i1 %cond) {
Entry:
	br i1 %cond, label %Inv, label %Cont
Inv:		; preds = %Entry
	invoke void @foo( )
			to label %Ok unwind label %Cont
Ok:		; preds = %Inv
	br label %Cont
Cont:		; preds = %Ok, %Inv, %Entry
	%X = phi i32 [ 0, %Entry ], [ 1, %Ok ], [ 0, %Inv ]		; <i32> [#uses=1]
	ret i32 %X
}

