; RUN: llc < %s -mtriple=arm-linux-gnueabi
; PR1279

	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32 }
	%struct.arm_stack_offsets = type { i32, i32, i32, i32, i32 }
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, i32, %struct.location_t, i32, i8*, %struct.rtx_def** }
	%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i8, i32, i64, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, i32, %struct.var_refs_queue*, i32, i32, %struct.rtvec_def*, %struct.tree_node*, i32, i32, i32, %struct.machine_function*, i32, i32, i8, i8, %struct.language_function*, %struct.rtx_def*, i32, i32, i32, i32, %struct.location_t, %struct.varray_head_tag*, %struct.tree_node*, i8, i8, i8 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { i8*, i32 }
	%struct.machine_function = type { %struct.rtx_def*, i32, i32, i32, %struct.arm_stack_offsets, i32, i32, i32, [14 x %struct.rtx_def*] }
	%struct.rtvec_def = type { i32, [1 x %struct.rtx_def*] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.temp_slot = type opaque
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %union.tree_ann_d*, i8, i8, i8, i8, i8 }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, %struct.tree_node*, i8, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, i32, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.u = type { [1 x i64] }
	%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_head_tag = type { i32, i32, i32, i8*, %struct.u }
	%union.tree_ann_d = type opaque
@str469 = external global [42 x i8]		; <[42 x i8]*> [#uses=0]
@__FUNCTION__.24265 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]

declare void @fancy_abort()

define fastcc void @fold_builtin_bitop() {
entry:
	br i1 false, label %cond_true105, label %UnifiedReturnBlock

cond_true105:		; preds = %entry
	br i1 false, label %cond_true134, label %UnifiedReturnBlock

cond_true134:		; preds = %cond_true105
	switch i32 0, label %bb479 [
		 i32 378, label %bb313
		 i32 380, label %bb313
		 i32 381, label %bb313
		 i32 383, label %bb366
		 i32 385, label %bb366
		 i32 386, label %bb366
		 i32 403, label %bb250
		 i32 405, label %bb250
		 i32 406, label %bb250
		 i32 434, label %bb464
		 i32 436, label %bb464
		 i32 437, label %bb464
		 i32 438, label %bb441
		 i32 440, label %bb441
		 i32 441, label %bb441
	]

bb250:		; preds = %cond_true134, %cond_true134, %cond_true134
	ret void

bb313:		; preds = %cond_true134, %cond_true134, %cond_true134
	ret void

bb366:		; preds = %cond_true134, %cond_true134, %cond_true134
	ret void

bb441:		; preds = %cond_true134, %cond_true134, %cond_true134
	ret void

bb457:		; preds = %bb464, %bb457
	%tmp459 = add i64 0, 1		; <i64> [#uses=1]
	br i1 false, label %bb474.preheader, label %bb457

bb464:		; preds = %cond_true134, %cond_true134, %cond_true134
	br i1 false, label %bb474.preheader, label %bb457

bb474.preheader:		; preds = %bb464, %bb457
	%result.5.ph = phi i64 [ 0, %bb464 ], [ %tmp459, %bb457 ]		; <i64> [#uses=1]
	br label %bb474

bb467:		; preds = %bb474
	%indvar.next586 = add i64 %indvar585, 1		; <i64> [#uses=1]
	br label %bb474

bb474:		; preds = %bb467, %bb474.preheader
	%indvar585 = phi i64 [ 0, %bb474.preheader ], [ %indvar.next586, %bb467 ]		; <i64> [#uses=2]
	br i1 false, label %bb476, label %bb467

bb476:		; preds = %bb474
	%result.5 = add i64 %indvar585, %result.5.ph		; <i64> [#uses=0]
	ret void

bb479:		; preds = %cond_true134
	tail call void @fancy_abort( )
	unreachable

UnifiedReturnBlock:		; preds = %cond_true105, %entry
	ret void
}
