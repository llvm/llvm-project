; RUN: llc < %s -mtriple=arm-linux-gnueabi
; PR1257

	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32 }
	%struct.arm_stack_offsets = type { i32, i32, i32, i32, i32 }
	%struct.c_arg_info = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i8 }
	%struct.c_language_function = type { %struct.stmt_tree_s }
	%struct.c_switch = type opaque
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, i32, %struct.location_t, i32, i8*, %struct.rtx_def** }
	%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i8, i32, i64, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, i32, %struct.var_refs_queue*, i32, i32, %struct.rtvec_def*, %struct.tree_node*, i32, i32, i32, %struct.machine_function*, i32, i32, i8, i8, %struct.language_function*, %struct.rtx_def*, i32, i32, i32, i32, %struct.location_t, %struct.varray_head_tag*, %struct.tree_node*, i8, i8, i8 }
	%struct.ht_identifier = type { i8*, i32, i32 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type { i8 }
	%struct.language_function = type { %struct.c_language_function, %struct.tree_node*, %struct.tree_node*, %struct.c_switch*, %struct.c_arg_info*, i32, i32, i32, i32 }
	%struct.location_t = type { i8*, i32 }
	%struct.machine_function = type { %struct.rtx_def*, i32, i32, i32, %struct.arm_stack_offsets, i32, i32, i32, [14 x %struct.rtx_def*] }
	%struct.rtvec_def = type { i32, [1 x %struct.rtx_def*] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.stmt_tree_s = type { %struct.tree_node*, i32 }
	%struct.temp_slot = type opaque
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %union.tree_ann_d*, i8, i8, i8, i8, i8 }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, %struct.tree_node*, i8, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, i32, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_identifier = type { %struct.tree_common, %struct.ht_identifier }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.u = type { [1 x i64] }
	%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_head_tag = type opaque
	%union.tree_ann_d = type opaque


define void @declspecs_add_type(i32 %spec.1) {
entry:
	%spec.1961 = zext i32 %spec.1 to i64		; <i64> [#uses=1]
	%spec.1961.adj = shl i64 %spec.1961, 32		; <i64> [#uses=1]
	%spec.1961.adj.ins = or i64 %spec.1961.adj, 0		; <i64> [#uses=2]
	%tmp10959 = lshr i64 %spec.1961.adj.ins, 32		; <i64> [#uses=2]
	%tmp1920 = inttoptr i64 %tmp10959 to %struct.tree_common*		; <%struct.tree_common*> [#uses=1]
	%tmp21 = getelementptr %struct.tree_common* %tmp1920, i32 0, i32 3		; <i8*> [#uses=1]
	%tmp2122 = bitcast i8* %tmp21 to i32*		; <i32*> [#uses=1]
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
	%tmp874 = load i32* %tmp2122		; <i32> [#uses=1]
	%tmp876877 = trunc i32 %tmp874 to i8		; <i8> [#uses=1]
	icmp eq i8 %tmp876877, 1		; <i1>:0 [#uses=1]
	br i1 %0, label %cond_next881, label %cond_true878

cond_true878:		; preds = %bb866
	unreachable

cond_next881:		; preds = %bb866
	%tmp884885 = inttoptr i64 %tmp10959 to %struct.tree_identifier*		; <%struct.tree_identifier*> [#uses=1]
	%tmp887 = getelementptr %struct.tree_identifier* %tmp884885, i32 0, i32 1, i32 0		; <i8**> [#uses=1]
	%tmp888 = load i8** %tmp887		; <i8*> [#uses=1]
	tail call void (i32, ...)* @error( i32 undef, i8* %tmp888 )
	ret void

cond_true918:		; preds = %cond_false841
	%tmp920957 = trunc i64 %spec.1961.adj.ins to i32		; <i32> [#uses=0]
	ret void
}

declare void @error(i32, ...)

declare void @lookup_name()
