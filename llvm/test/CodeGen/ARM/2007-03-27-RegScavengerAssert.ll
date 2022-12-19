; RUN: llc < %s -mtriple=arm-linux-gnueabi
; PR1279

	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.u = type { [1 x i64] }

define fastcc void @find_reloads_address(ptr %loc) {
entry:
	%ad_addr = alloca ptr		; <ptr> [#uses=2]
	br i1 false, label %cond_next416, label %cond_true340

cond_true340:		; preds = %entry
	ret void

cond_next416:		; preds = %entry
	%tmp1085 = load ptr, ptr %ad_addr		; <ptr> [#uses=1]
	br i1 false, label %bb1084, label %cond_true418

cond_true418:		; preds = %cond_next416
	ret void

bb1084:		; preds = %cond_next416
	br i1 false, label %cond_true1092, label %cond_next1102

cond_true1092:		; preds = %bb1084
	%tmp1094 = getelementptr %struct.rtx_def, ptr %tmp1085, i32 0, i32 3		; <ptr> [#uses=1]
	%tmp1101 = load ptr, ptr %tmp1094		; <ptr> [#uses=1]
	store ptr %tmp1101, ptr %ad_addr
	br label %cond_next1102

cond_next1102:		; preds = %cond_true1092, %bb1084
	%loc_addr.0 = phi ptr [ %tmp1094, %cond_true1092 ], [ %loc, %bb1084 ]		; <ptr> [#uses=0]
	ret void
}
