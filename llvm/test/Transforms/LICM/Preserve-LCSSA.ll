; RUN: opt < %s -loop-rotate -licm -loop-unswitch -disable-output -verify-loop-info -verify-dom-info

define i32 @stringSearch_Clib(i32 %count) {
entry:
	br i1 false, label %bb36, label %bb44

bb4:		; preds = %bb36
	br i1 false, label %cond_next, label %cond_true

cond_true:		; preds = %bb4
	ret i32 0

cond_next:		; preds = %bb4
	ret i32 0

bb36:		; preds = %bb41, %entry
	br i1 false, label %bb4, label %bb41

bb41:		; preds = %bb36
	%ttmp2 = icmp slt i32 0, %count		; <i1> [#uses=1]
	br i1 %ttmp2, label %bb36, label %bb44

bb44:		; preds = %bb41, %entry
	ret i32 0
}
