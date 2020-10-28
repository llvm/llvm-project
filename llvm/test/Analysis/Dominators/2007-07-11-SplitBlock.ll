; RUN: opt < %s -loop-rotate -simple-loop-unswitch -disable-output

define i32 @stringSearch_Clib(i32 %count) {
entry:
	br i1 false, label %bb36, label %bb44

cond_true20:		; preds = %bb36
	%tmp33 = add i32 0, 0		; <i32> [#uses=1]
	br label %bb36

bb36:		; preds = %cond_true20, %entry
	%c.2 = phi i32 [ %tmp33, %cond_true20 ], [ 0, %entry ]		; <i32> [#uses=1]
	br i1 false, label %cond_true20, label %bb41

bb41:		; preds = %bb36
	%c.2.lcssa = phi i32 [ %c.2, %bb36 ]		; <i32> [#uses=0]
	ret i32 0

bb44:		; preds = %entry
	ret i32 0
}
