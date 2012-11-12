; RUN: opt < %s -analyze -scalar-evolution
; PR2602

define i32 @a() nounwind  {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%w.0 = phi i32 [ 0, %entry ], [ %tmp, %bb ]		; <i32> [#uses=2]
	%e.0 = phi i32 [ 0, %entry ], [ %e.1, %bb ]		; <i32> [#uses=2]
	%w.1 = add i32 0, %w.0		; <i32>:0 [#uses=1]
	%tmp = add i32 %e.0, %w.0		; <i32>:1 [#uses=1]
	%e.1 = add i32 %e.0, 1		; <i32>:2 [#uses=1]
	%cond = icmp eq i32 %w.1, -1		; <i1>:3 [#uses=1]
	br i1 %cond, label %return, label %bb

return:		; preds = %bb
	ret i32 undef
}
