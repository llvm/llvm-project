; Do not promote null value because it may be unsafe to do so.
; RUN: opt < %s -passes=licm -S | not grep promoted

define i32 @f(i32 %foo, i32 %bar, i32 %com) {
entry:
	%tmp2 = icmp eq i32 %foo, 0		; <i1> [#uses=1]
	br i1 %tmp2, label %cond_next, label %cond_true

cond_true:		; preds = %entry
	br label %return

cond_next:		; preds = %entry
	br label %bb

bb:		; preds = %bb15, %cond_next
	switch i32 %bar, label %bb15 [
		 i32 1, label %bb6
	]

bb6:		; preds = %bb
	%tmp8 = icmp eq i32 %com, 0		; <i1> [#uses=1]
	br i1 %tmp8, label %cond_next14, label %cond_true11

cond_true11:		; preds = %bb6
	br label %return

cond_next14:		; preds = %bb6
	store i8 0, ptr null
	br label %bb15

bb15:		; preds = %cond_next14, %bb
	br label %bb

return:		; preds = %cond_true11, %cond_true
	%storemerge = phi i32 [ 0, %cond_true ], [ undef, %cond_true11 ]		; <i32> [#uses=1]
	ret i32 %storemerge
}

define i32 @kdMain() {
entry:
	%tmp1 = call i32 @f( i32 0, i32 1, i32 1 )		; <i32> [#uses=0]
	call void @exit( i32 0 )
	unreachable
}

declare void @exit(i32)
