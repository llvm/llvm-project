; RUN: llc < %s -mtriple=arm-linux-gnueabi
; PR1279

	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32 }
	%struct.arm_stack_offsets = type { i32, i32, i32, i32, i32 }
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, ptr, ptr, ptr, i32, %struct.location_t, i32, ptr, ptr }
	%struct.expr_status = type { i32, i32, i32, ptr, ptr, ptr }
	%struct.function = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, ptr, %struct.CUMULATIVE_ARGS, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8, i32, i64, ptr, ptr, ptr, ptr, ptr, i32, ptr, i32, i32, ptr, ptr, i32, i32, i32, ptr, i32, i32, i8, i8, ptr, ptr, i32, i32, i32, i32, %struct.location_t, ptr, ptr, i8, i8, i8 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { ptr, i32 }
	%struct.machine_function = type { ptr, i32, i32, i32, %struct.arm_stack_offsets, i32, i32, i32, [14 x ptr] }
	%struct.rtvec_def = type { i32, [1 x ptr] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { ptr, ptr, ptr }
	%struct.temp_slot = type opaque
	%struct.tree_common = type { ptr, ptr, ptr, i8, i8, i8, i8, i8 }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, ptr, i8, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, %struct.tree_decl_u2, ptr, ptr, i64, ptr }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u2 = type { ptr }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.u = type { [1 x i64] }
	%struct.var_refs_queue = type { ptr, i32, i32, ptr }
	%struct.varasm_status = type opaque
	%struct.varray_head_tag = type { i32, i32, i32, ptr, %struct.u }
	%union.tree_ann_d = type opaque
@str469 = external global [42 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.24265 = external global [19 x i8]		; <ptr> [#uses=0]

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
