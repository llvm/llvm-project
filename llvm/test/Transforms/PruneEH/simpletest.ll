; RUN: opt < %s -prune-eh -S | not grep invoke

declare void @nounwind() nounwind

define internal void @foo() {
	call void @nounwind()
	ret void
}

define i32 @caller() {
	invoke void @foo( )
			to label %Normal unwind label %Except

Normal:		; preds = %0
	ret i32 0

Except:		; preds = %0
	ret i32 1
}
