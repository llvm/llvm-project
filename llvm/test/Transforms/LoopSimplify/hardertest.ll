; RUN: opt < %s -passes=loop-simplify

define void @foo(i1 %C) {
	br i1 %C, label %T, label %F
T:		; preds = %0
	br label %Loop
F:		; preds = %0
	br label %Loop
Loop:		; preds = %L2, %Loop, %F, %T
	%Val = phi i32 [ 0, %T ], [ 1, %F ], [ 2, %Loop ], [ 3, %L2 ]		; <i32> [#uses=0]
	br i1 %C, label %Loop, label %L2
L2:		; preds = %Loop
	br label %Loop
}

