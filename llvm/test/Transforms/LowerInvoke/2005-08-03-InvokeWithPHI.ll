; RUN: opt < %s -lowerinvoke -enable-correct-eh-support -disable-output

declare void @ll_listnext__listiterPtr()

define void @WorkTask.fn() {
block0:
	invoke void @ll_listnext__listiterPtr( )
			to label %block9 unwind label %block8_exception_handling
block8_exception_handling:		; preds = %block0
	ret void
block9:		; preds = %block0
	%w_2690 = phi { i32, i32 }* [ null, %block0 ]		; <{ i32, i32 }*> [#uses=1]
	%tmp.129 = getelementptr { i32, i32 }* %w_2690, i32 0, i32 1		; <i32*> [#uses=1]
	%v2769 = load i32* %tmp.129		; <i32> [#uses=0]
	ret void
}

