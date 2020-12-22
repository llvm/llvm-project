; Here we have a case where there are two loops and LICM is hoisting an
; instruction from one loop into the other loop!  This is obviously bad and
; happens because preheader insertion doesn't insert a preheader for this
; case... bad.

; RUN: opt < %s -licm -loop-deletion -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | \
; RUN:   not grep "br "

define i32 @main(i32 %argc) {
; <label>:0
	br label %bb5
bb5:		; preds = %bb5, %0
	%I = phi i32 [ 0, %0 ], [ %I2, %bb5 ]		; <i32> [#uses=1]
	%I2 = add i32 %I, 1		; <i32> [#uses=2]
	%c = icmp eq i32 %I2, 10		; <i1> [#uses=1]
	br i1 %c, label %bb5, label %bb8
bb8:		; preds = %bb8, %bb5
	%cann-indvar = phi i32 [ 0, %bb8 ], [ 0, %bb5 ]		; <i32> [#uses=0]
	%X = add i32 %argc, %argc		; <i32> [#uses=1]
	br i1 false, label %bb8, label %bb10
bb10:		; preds = %bb8
	ret i32 %X
}

