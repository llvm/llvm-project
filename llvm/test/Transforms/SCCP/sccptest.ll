; This is the test case taken from appel's book that illustrates a hard case
; that SCCP gets right. BB3 should be completely eliminated.
;
; RUN: llvm-as < %s | opt -sccp -constprop -dce -simplifycfg | \
; RUN:   llvm-dis | not grep BB3

define i32 @testfunction(i32 %i0, i32 %j0) {
BB1:
	br label %BB2
BB2:		; preds = %BB7, %BB1
	%j2 = phi i32 [ %j4, %BB7 ], [ 1, %BB1 ]		; <i32> [#uses=2]
	%k2 = phi i32 [ %k4, %BB7 ], [ 0, %BB1 ]		; <i32> [#uses=4]
	%kcond = icmp slt i32 %k2, 100		; <i1> [#uses=1]
	br i1 %kcond, label %BB3, label %BB4
BB3:		; preds = %BB2
	%jcond = icmp slt i32 %j2, 20		; <i1> [#uses=1]
	br i1 %jcond, label %BB5, label %BB6
BB4:		; preds = %BB2
	ret i32 %j2
BB5:		; preds = %BB3
	%k3 = add i32 %k2, 1		; <i32> [#uses=1]
	br label %BB7
BB6:		; preds = %BB3
	%k5 = add i32 %k2, 1		; <i32> [#uses=1]
	br label %BB7
BB7:		; preds = %BB6, %BB5
	%j4 = phi i32 [ 1, %BB5 ], [ %k2, %BB6 ]		; <i32> [#uses=1]
	%k4 = phi i32 [ %k3, %BB5 ], [ %k5, %BB6 ]		; <i32> [#uses=1]
	br label %BB2
}

