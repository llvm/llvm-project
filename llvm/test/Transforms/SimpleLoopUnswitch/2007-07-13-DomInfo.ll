; RUN: opt < %s -simple-loop-unswitch -disable-output

define i32 @main(i32 %argc, i8** %argv) {
entry:
	%tmp1785365 = icmp ult i32 0, 100		; <i1> [#uses=1]
	br label %bb

bb:		; preds = %cond_true, %entry
	br i1 false, label %cond_true, label %cond_next

cond_true:		; preds = %bb
	br i1 %tmp1785365, label %bb, label %bb1788

cond_next:		; preds = %bb
	%iftmp.1.0 = select i1 false, i32 0, i32 0		; <i32> [#uses=1]
	br i1 false, label %cond_true47, label %cond_next74

cond_true47:		; preds = %cond_next
	%tmp53 = urem i32 %iftmp.1.0, 0		; <i32> [#uses=0]
	ret i32 0

cond_next74:		; preds = %cond_next
	ret i32 0

bb1788:		; preds = %cond_true
	ret i32 0
}
