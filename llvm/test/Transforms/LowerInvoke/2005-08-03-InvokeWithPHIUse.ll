; RUN: opt < %s -lowerinvoke -enable-correct-eh-support -disable-output

declare fastcc i32 @ll_listnext__listiterPtr()

define fastcc i32 @WorkTask.fn() {
block0:
	%v2679 = invoke fastcc i32 @ll_listnext__listiterPtr( )
			to label %block9 unwind label %block8_exception_handling	; <i32> [#uses=1]
block8_exception_handling:		; preds = %block0
	ret i32 0
block9:		; preds = %block0
	%i_2689 = phi i32 [ %v2679, %block0 ]		; <i32> [#uses=1]
	ret i32 %i_2689
}

