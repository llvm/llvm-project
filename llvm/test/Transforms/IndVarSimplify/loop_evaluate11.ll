; RUN: opt < %s -domfrontier -indvars -loop-deletion

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

define void @slap_sl_mem_create() nounwind {
entry:
	br label %bb15

bb15:		; preds = %bb15, %entry
	%order_end.0 = phi i32 [ 0, %entry ], [ %tmp, %bb15 ]		; <i32> [#uses=1]
	%tmp = add i32 %order_end.0, 1		; <i32> [#uses=2]
	br i1 undef, label %bb17, label %bb15

bb17:		; preds = %bb17, %bb15
	%order_start.0 = phi i32 [ %tmp1, %bb17 ], [ 0, %bb15 ]		; <i32> [#uses=2]
	%tmp1 = add i32 %order_start.0, 1		; <i32> [#uses=2]
	%tmp2 = icmp eq i32 undef, 0		; <i1> [#uses=1]
	br i1 %tmp2, label %bb18, label %bb17

bb18:		; preds = %bb17
	%tmp3 = sub i32 %tmp, %tmp1		; <i32> [#uses=0]
	br label %bb59

bb51:		; preds = %bb59
	%tmp4 = add i32 %order_start.0, 2		; <i32> [#uses=1]
	%tmp5 = add i32 %tmp4, undef		; <i32> [#uses=1]
	%tmp6 = lshr i32 undef, %tmp5		; <i32> [#uses=1]
	%tmp7 = icmp eq i32 %tmp6, 0		; <i1> [#uses=1]
	br i1 %tmp7, label %bb52, label %bb59

bb59:		; preds = %bb51, %bb18
	br label %bb51

bb52:		; preds = %bb51
	unreachable
}
