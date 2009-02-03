; RUN: llvm-as < %s | opt -simplifycfg -disable-output

define i1 @foo() {
	%X = invoke i1 @foo( )
			to label %N unwind label %F		; <i1> [#uses=1]
F:		; preds = %0
	ret i1 false
N:		; preds = %0
	br i1 %X, label %A, label %B
A:		; preds = %N
	ret i1 true
B:		; preds = %N
	ret i1 true
}

