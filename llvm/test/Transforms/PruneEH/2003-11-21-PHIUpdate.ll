; RUN: llvm-as < %s | opt -prune-eh -disable-output

define internal void @callee() {
	ret void
}

define i32 @caller() {
; <label>:0
	invoke void @callee( )
			to label %E unwind label %E
E:		; preds = %0, %0
	%X = phi i32 [ 0, %0 ], [ 0, %0 ]		; <i32> [#uses=1]
	ret i32 %X
}

