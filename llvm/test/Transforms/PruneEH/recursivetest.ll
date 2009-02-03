; RUN: llvm-as < %s | opt -prune-eh | llvm-dis | not grep invoke

define internal i32 @foo() {
	invoke i32 @foo( )
			to label %Normal unwind label %Except		; <i32>:1 [#uses=0]
Normal:		; preds = %0
	ret i32 12
Except:		; preds = %0
	ret i32 123
}

define i32 @caller() {
	invoke i32 @foo( )
			to label %Normal unwind label %Except		; <i32>:1 [#uses=0]
Normal:		; preds = %0
	ret i32 0
Except:		; preds = %0
	ret i32 1
}

