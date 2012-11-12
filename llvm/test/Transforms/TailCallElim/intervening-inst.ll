; This function contains intervening instructions which should be moved out of the way
; RUN: opt < %s -tailcallelim -S | not grep call

define i32 @Test(i32 %X) {
entry:
	%tmp.1 = icmp eq i32 %X, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %then.0, label %endif.0
then.0:		; preds = %entry
	%tmp.4 = add i32 %X, 1		; <i32> [#uses=1]
	ret i32 %tmp.4
endif.0:		; preds = %entry
	%tmp.10 = add i32 %X, -1		; <i32> [#uses=1]
	%tmp.8 = call i32 @Test( i32 %tmp.10 )		; <i32> [#uses=1]
	%DUMMY = add i32 %X, 1		; <i32> [#uses=0]
	ret i32 %tmp.8
}

