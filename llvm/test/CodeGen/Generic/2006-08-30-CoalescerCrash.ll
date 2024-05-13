; RUN: llc < %s	
%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.VEC_edge = type { i32, i32, [1 x ptr] }
	%struct._obstack_chunk = type { ptr, ptr, [4 x i8] }
	%struct.basic_block_def = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, [2 x ptr], ptr, ptr, ptr, ptr, i64, i32, i32, i32, i32 }
	%struct.bb_ann_d = type { ptr, i8, ptr }
	%struct.bitmap_element_def = type { ptr, ptr, i32, [4 x i32] }
	%struct.bitmap_head_def = type { ptr, ptr, i32, ptr }
	%struct.bitmap_obstack = type { ptr, ptr, %struct.obstack }
	%struct.cost_pair = type { ptr, i32, ptr }
	%struct.dataflow_d = type { ptr, [2 x ptr] }
	%struct.def_operand_ptr = type { ptr }
	%struct.def_optype_d = type { i32, [1 x %struct.def_operand_ptr] }
	%struct.edge_def = type { ptr, ptr, %struct.edge_def_insns, ptr, ptr, i32, i32, i64, i32 }
	%struct.edge_def_insns = type { ptr }
	%struct.edge_prediction = type { ptr, ptr, i32, i32 }
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, ptr, ptr, ptr, i32, %struct.location_t, i32, ptr, ptr }
	%struct.et_node = type opaque
	%struct.expr_status = type { i32, i32, i32, ptr, ptr, ptr }
	%struct.function = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, ptr, %struct.CUMULATIVE_ARGS, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8, i32, i64, ptr, ptr, ptr, ptr, ptr, i32, ptr, i32, i32, ptr, ptr, i32, i32, i32, ptr, i32, i32, i1, i1, ptr, ptr, i32, i32, i32, i32, %struct.location_t, ptr, ptr, i8, i8, i8 }
	%struct.htab = type { ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, i32 }
	%struct.initial_value_struct = type opaque
	%struct.iv = type { ptr, ptr, ptr, ptr, i1, i1, i32 }
	%struct.iv_cand = type { i32, i1, i32, ptr, ptr, ptr, ptr, i32 }
	%struct.iv_use = type { i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr }
	%struct.ivopts_data = type { ptr, ptr, i32, ptr, ptr, i32, ptr, ptr, ptr, i1 }
	%struct.lang_decl = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { ptr, i32 }
	%struct.loop = type { i32, ptr, ptr, ptr, %struct.lpt_decision, i32, i32, ptr, i32, ptr, ptr, i32, ptr, i32, ptr, i32, ptr, i32, ptr, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, i1 }
	%struct.lpt_decision = type { i32, i32 }
	%struct.machine_function = type { ptr, ptr, ptr, i32, i32, i32, i32, i32 }
	%struct.nb_iter_bound = type { ptr, ptr, ptr, ptr }
	%struct.obstack = type { i32, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, i8 }
	%struct.reorder_block_def = type { ptr, ptr, ptr, ptr, ptr, i32, i32, i32 }
	%struct.rtvec_def = type { i32, [1 x ptr] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { ptr, ptr, ptr }
	%struct.simple_bitmap_def = type { i32, i32, i32, [1 x i64] }
	%struct.stack_local_entry = type opaque
	%struct.stmt_ann_d = type { %struct.tree_ann_common_d, i8, ptr, %struct.stmt_operands_d, ptr, ptr, i32 }
	%struct.stmt_operands_d = type { ptr, ptr, ptr, ptr, ptr }
	%struct.temp_slot = type opaque
	%struct.tree_ann_common_d = type { i32, ptr, ptr }
	%struct.tree_ann_d = type { %struct.stmt_ann_d }
	%struct.tree_common = type { ptr, ptr, ptr, i8, i8, i8, i8, i8 }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, ptr, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, %struct.tree_decl_u2, ptr, ptr, i64, ptr }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u2 = type { ptr }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.u = type { [1 x i64] }
	%struct.v_def_use_operand_type_t = type { ptr, ptr }
	%struct.v_may_def_optype_d = type { i32, [1 x %struct.v_def_use_operand_type_t] }
	%struct.var_refs_queue = type { ptr, i32, i32, ptr }
	%struct.varasm_status = type opaque
	%struct.varray_head_tag = type { i32, i32, i32, ptr, %struct.u }
	%struct.version_info = type { ptr, ptr, i1, i32, i1 }
	%struct.vuse_optype_d = type { i32, [1 x ptr] }

define i1 @determine_use_iv_cost(ptr %data, ptr %use, ptr %cand) {
entry:
	switch i32 0, label %bb91 [
		 i32 0, label %bb
		 i32 1, label %bb6
		 i32 3, label %cond_next135
	]

bb:		; preds = %entry
	ret i1 false

bb6:		; preds = %entry
	br i1 false, label %bb87, label %cond_next27

cond_next27:		; preds = %bb6
	br i1 false, label %cond_true30, label %cond_next55

cond_true30:		; preds = %cond_next27
	br i1 false, label %cond_next41, label %cond_true35

cond_true35:		; preds = %cond_true30
	ret i1 false

cond_next41:		; preds = %cond_true30
	%tmp44 = call i32 @force_var_cost( ptr %data, ptr null, ptr null )		; <i32> [#uses=2]
	%tmp46 = udiv i32 %tmp44, 5		; <i32> [#uses=1]
	call void @set_use_iv_cost( ptr %data, ptr %use, ptr %cand, i32 %tmp46, ptr null )
	%tmp44.off = add i32 %tmp44, -50000000		; <i32> [#uses=1]
	%tmp52 = icmp ugt i32 %tmp44.off, 4		; <i1> [#uses=1]
	%tmp52.upgrd.1 = zext i1 %tmp52 to i32		; <i32> [#uses=1]
	br label %bb87

cond_next55:		; preds = %cond_next27
	ret i1 false

bb87:		; preds = %cond_next41, %bb6
	%tmp2.0 = phi i32 [ %tmp52.upgrd.1, %cond_next41 ], [ 1, %bb6 ]		; <i32> [#uses=0]
	ret i1 false

bb91:		; preds = %entry
	ret i1 false

cond_next135:		; preds = %entry
	%tmp193 = call i1 @determine_use_iv_cost_generic( ptr %data, ptr %use, ptr %cand )		; <i1> [#uses=0]
	ret i1 false
}

declare void @set_use_iv_cost(ptr, ptr, ptr, i32, ptr)

declare i32 @force_var_cost(ptr, ptr, ptr)

declare i1 @determine_use_iv_cost_generic(ptr, ptr, ptr)
