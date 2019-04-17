; REQUIRES: asserts
; This function contains two tail calls, which should be eliminated
; RUN: opt < %s -tailcallelim -verify-dom-info -stats -disable-output 2>&1 | grep "2 tailcallelim"

define i32 @Ack(i32 %M.1, i32 %N.1) {
entry:
	%tmp.1 = icmp eq i32 %M.1, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %then.0, label %endif.0
then.0:		; preds = %entry
	%tmp.4 = add i32 %N.1, 1		; <i32> [#uses=1]
	ret i32 %tmp.4
endif.0:		; preds = %entry
	%tmp.6 = icmp eq i32 %N.1, 0		; <i1> [#uses=1]
	br i1 %tmp.6, label %then.1, label %endif.1
then.1:		; preds = %endif.0
	%tmp.10 = add i32 %M.1, -1		; <i32> [#uses=1]
	%tmp.8 = call i32 @Ack( i32 %tmp.10, i32 1 )		; <i32> [#uses=1]
	ret i32 %tmp.8
endif.1:		; preds = %endif.0
	%tmp.13 = add i32 %M.1, -1		; <i32> [#uses=1]
	%tmp.17 = add i32 %N.1, -1		; <i32> [#uses=1]
	%tmp.14 = call i32 @Ack( i32 %M.1, i32 %tmp.17 )		; <i32> [#uses=1]
	%tmp.11 = call i32 @Ack( i32 %tmp.13, i32 %tmp.14 )		; <i32> [#uses=1]
	ret i32 %tmp.11
}

