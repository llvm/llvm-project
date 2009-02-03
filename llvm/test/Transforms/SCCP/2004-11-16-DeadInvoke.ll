; RUN: llvm-as < %s | opt -sccp -disable-output

declare i32 @foo()

define void @caller() {
	br i1 true, label %T, label %F
F:		; preds = %0
	%X = invoke i32 @foo( )
			to label %T unwind label %T		; <i32> [#uses=0]
T:		; preds = %F, %F, %0
	ret void
}

