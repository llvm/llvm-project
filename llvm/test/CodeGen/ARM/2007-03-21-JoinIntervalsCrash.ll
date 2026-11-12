; RUN: llc < %s -mtriple=arm-linux-gnueabi
; PR1257

	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32 }
	%struct.arm_stack_offsets = type { i32, i32, i32, i32, i32 }
	%struct.c_arg_info = type { ptr, ptr, ptr, ptr, i8 }
	%struct.c_language_function = type { %struct.stmt_tree_s }
	%struct.c_switch = type opaque
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, ptr, ptr, ptr, i32, %struct.location_t, i32, ptr, ptr }
	%struct.expr_status = type { i32, i32, i32, ptr, ptr, ptr }
	%struct.function = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, ptr, %struct.CUMULATIVE_ARGS, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8, i32, i64, ptr, ptr, ptr, ptr, ptr, i32, ptr, i32, i32, ptr, ptr, i32, i32, i32, ptr, i32, i32, i8, i8, ptr, ptr, i32, i32, i32, i32, %struct.location_t, ptr, ptr, i8, i8, i8 }
	%struct.ht_identifier = type { ptr, i32, i32 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type { i8 }
	%struct.language_function = type { %struct.c_language_function, ptr, ptr, ptr, ptr, i32, i32, i32, i32 }
	%struct.location_t = type { ptr, i32 }
	%struct.machine_function = type { ptr, i32, i32, i32, %struct.arm_stack_offsets, i32, i32, i32, [14 x ptr] }
	%struct.rtvec_def = type { i32, [1 x ptr] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { ptr, ptr, ptr }
	%struct.stmt_tree_s = type { ptr, i32 }
	%struct.temp_slot = type opaque
	%struct.tree_common = type { ptr, ptr, ptr, i8, i8, i8, i8, i8 }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, ptr, i8, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, %struct.tree_decl_u2, ptr, ptr, i64, ptr }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u2 = type { ptr }
	%struct.tree_identifier = type { %struct.tree_common, %struct.ht_identifier }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.u = type { [1 x i64] }
	%struct.var_refs_queue = type { ptr, i32, i32, ptr }
	%struct.varasm_status = type opaque
	%struct.varray_head_tag = type opaque
	%union.tree_ann_d = type opaque


define void @declspecs_add_type(i32 %spec.1) {
entry:
	%spec.1961 = zext i32 %spec.1 to i64		; <i64> [#uses=1]
	%spec.1961.adj = shl i64 %spec.1961, 32		; <i64> [#uses=1]
	%spec.1961.adj.ins = or i64 %spec.1961.adj, 0		; <i64> [#uses=2]
	%tmp10959 = lshr i64 %spec.1961.adj.ins, 32		; <i64> [#uses=2]
	%tmp1920 = inttoptr i64 %tmp10959 to ptr		; <ptr> [#uses=1]
	%tmp21 = getelementptr %struct.tree_common, ptr %tmp1920, i32 0, i32 3		; <ptr> [#uses=1]
	br i1 false, label %cond_next53, label %cond_true

cond_true:		; preds = %entry
	ret void

cond_next53:		; preds = %entry
	br i1 false, label %cond_true63, label %cond_next689

cond_true63:		; preds = %cond_next53
	ret void

cond_next689:		; preds = %cond_next53
	br i1 false, label %cond_false841, label %bb743

bb743:		; preds = %cond_next689
	ret void

cond_false841:		; preds = %cond_next689
	br i1 false, label %cond_true851, label %cond_true918

cond_true851:		; preds = %cond_false841
	tail call void @lookup_name( )
	br i1 false, label %bb866, label %cond_next856

cond_next856:		; preds = %cond_true851
	ret void

bb866:		; preds = %cond_true851
	%tmp874 = load i32, ptr %tmp21		; <i32> [#uses=1]
	%tmp876877 = trunc i32 %tmp874 to i8		; <i8> [#uses=1]
	icmp eq i8 %tmp876877, 1		; <i1>:0 [#uses=1]
	br i1 %0, label %cond_next881, label %cond_true878

cond_true878:		; preds = %bb866
	unreachable

cond_next881:		; preds = %bb866
	%tmp884885 = inttoptr i64 %tmp10959 to ptr		; <ptr> [#uses=1]
	%tmp887 = getelementptr %struct.tree_identifier, ptr %tmp884885, i32 0, i32 1, i32 0		; <ptr> [#uses=1]
	%tmp888 = load ptr, ptr %tmp887		; <ptr> [#uses=1]
	tail call void (i32, ...) @error( i32 undef, ptr %tmp888 )
	ret void

cond_true918:		; preds = %cond_false841
	%tmp920957 = trunc i64 %spec.1961.adj.ins to i32		; <i32> [#uses=0]
	ret void
}

declare void @error(i32, ...)

declare void @lookup_name()
