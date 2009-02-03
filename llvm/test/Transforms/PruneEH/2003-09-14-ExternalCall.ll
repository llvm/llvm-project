; RUN: llvm-as < %s | opt -prune-eh | llvm-dis | grep invoke

declare void @External()

define void @foo() {
	invoke void @External( )
			to label %Cont unwind label %Cont
Cont:		; preds = %0, %0
	ret void
}

