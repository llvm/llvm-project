; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -disable-output
; PR3016
; Dead use caused invariant violation.

define i32 @func_105(i1 %tmp5, i1 %tmp7) nounwind {
BB:
	br i1 true, label %BB2, label %BB1

BB1:		; preds = %BB
	br label %BB2

BB2:		; preds = %BB1, %BB
	%tmp3 = phi i1 [ true, %BB ], [ false, %BB1 ]		; <i1> [#uses=1]
	br label %BB9

BB9:		; preds = %BB11, %BB2
	%tmp10 = phi i32 [ 0, %BB2 ], [ %tmp12, %BB11 ]		; <i32> [#uses=1]
	br i1 %tmp5, label %BB11, label %BB13

BB11:		; preds = %BB13, %BB9
	%tmp12 = phi i32 [ 0, %BB13 ], [ %tmp10, %BB9 ]		; <i32> [#uses=2]
	br i1 %tmp3, label %BB9, label %BB20

BB13:		; preds = %BB13, %BB9
	%tmp14 = phi i32 [ 0, %BB9 ], [ %tmp14, %BB13 ]		; <i32> [#uses=1]
	br i1 %tmp7, label %BB13, label %BB11

BB20:		; preds = %BB11
	ret i32 %tmp12
}
