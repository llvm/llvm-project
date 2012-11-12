; RUN: llc < %s -march=arm -mtriple=arm-linux-gnueabi
; PR1279

	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.u = type { [1 x i64] }

define fastcc void @find_reloads_address(%struct.rtx_def** %loc) {
entry:
	%ad_addr = alloca %struct.rtx_def*		; <%struct.rtx_def**> [#uses=2]
	br i1 false, label %cond_next416, label %cond_true340

cond_true340:		; preds = %entry
	ret void

cond_next416:		; preds = %entry
	%tmp1085 = load %struct.rtx_def** %ad_addr		; <%struct.rtx_def*> [#uses=1]
	br i1 false, label %bb1084, label %cond_true418

cond_true418:		; preds = %cond_next416
	ret void

bb1084:		; preds = %cond_next416
	br i1 false, label %cond_true1092, label %cond_next1102

cond_true1092:		; preds = %bb1084
	%tmp1094 = getelementptr %struct.rtx_def* %tmp1085, i32 0, i32 3		; <%struct.u*> [#uses=1]
	%tmp10981099 = bitcast %struct.u* %tmp1094 to %struct.rtx_def**		; <%struct.rtx_def**> [#uses=2]
	%tmp1101 = load %struct.rtx_def** %tmp10981099		; <%struct.rtx_def*> [#uses=1]
	store %struct.rtx_def* %tmp1101, %struct.rtx_def** %ad_addr
	br label %cond_next1102

cond_next1102:		; preds = %cond_true1092, %bb1084
	%loc_addr.0 = phi %struct.rtx_def** [ %tmp10981099, %cond_true1092 ], [ %loc, %bb1084 ]		; <%struct.rtx_def**> [#uses=0]
	ret void
}
