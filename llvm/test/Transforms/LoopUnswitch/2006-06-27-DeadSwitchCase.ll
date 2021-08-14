; RUN: opt < %s -loop-unswitch -enable-new-pm=0 -verify-memoryssa -disable-output

define void @init_caller_save() {
entry:
	br label %cond_true78
cond_next20:		; preds = %cond_true64
	br label %bb31
bb31:		; preds = %cond_true64, %cond_true64, %cond_next20
	%iftmp.29.1 = phi i32 [ 0, %cond_next20 ], [ 0, %cond_true64 ], [ 0, %cond_true64 ]		; <i32> [#uses=0]
	br label %bb54
bb54:		; preds = %cond_true78, %bb31
	br i1 false, label %bb75, label %cond_true64
cond_true64:		; preds = %bb54
	switch i32 %i.0.0, label %cond_next20 [
		 i32 17, label %bb31
		 i32 18, label %bb31
	]
bb75:		; preds = %bb54
	%tmp74.0 = add i32 %i.0.0, 1		; <i32> [#uses=1]
	br label %cond_true78
cond_true78:		; preds = %bb75, %entry
	%i.0.0 = phi i32 [ 0, %entry ], [ %tmp74.0, %bb75 ]		; <i32> [#uses=2]
	br label %bb54
}

