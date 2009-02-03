; RUN: llvm-as < %s | llc	
%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.VEC_edge = type { i32, i32, [1 x %struct.edge_def*] }
	%struct._obstack_chunk = type { i8*, %struct._obstack_chunk*, [4 x i8] }
	%struct.basic_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.tree_node*, %struct.VEC_edge*, %struct.VEC_edge*, %struct.bitmap_head_def*, %struct.bitmap_head_def*, i8*, %struct.loop*, [2 x %struct.et_node*], %struct.basic_block_def*, %struct.basic_block_def*, %struct.reorder_block_def*, %struct.bb_ann_d*, i64, i32, i32, i32, i32 }
	%struct.bb_ann_d = type { %struct.tree_node*, i8, %struct.edge_prediction* }
	%struct.bitmap_element_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, [4 x i32] }
	%struct.bitmap_head_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, %struct.bitmap_obstack* }
	%struct.bitmap_obstack = type { %struct.bitmap_element_def*, %struct.bitmap_head_def*, %struct.obstack }
	%struct.cost_pair = type { %struct.iv_cand*, i32, %struct.bitmap_head_def* }
	%struct.dataflow_d = type { %struct.varray_head_tag*, [2 x %struct.tree_node*] }
	%struct.def_operand_ptr = type { %struct.tree_node** }
	%struct.def_optype_d = type { i32, [1 x %struct.def_operand_ptr] }
	%struct.edge_def = type { %struct.basic_block_def*, %struct.basic_block_def*, %struct.edge_def_insns, i8*, %struct.location_t*, i32, i32, i64, i32 }
	%struct.edge_def_insns = type { %struct.rtx_def* }
	%struct.edge_prediction = type { %struct.edge_prediction*, %struct.edge_def*, i32, i32 }
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, i32, %struct.location_t, i32, i8*, %struct.rtx_def** }
	%struct.et_node = type opaque
	%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i8, i32, i64, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, i32, %struct.var_refs_queue*, i32, i32, %struct.rtvec_def*, %struct.tree_node*, i32, i32, i32, %struct.machine_function*, i32, i32, i1, i1, %struct.language_function*, %struct.rtx_def*, i32, i32, i32, i32, %struct.location_t, %struct.varray_head_tag*, %struct.tree_node*, i8, i8, i8 }
	%struct.htab = type { i32 (i8*)*, i32 (i8*, i8*)*, void (i8*)*, i8**, i32, i32, i32, i32, i32, i8* (i32, i32)*, void (i8*)*, i8*, i8* (i8*, i32, i32)*, void (i8*, i8*)*, i32 }
	%struct.initial_value_struct = type opaque
	%struct.iv = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i1, i1, i32 }
	%struct.iv_cand = type { i32, i1, i32, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.iv*, i32 }
	%struct.iv_use = type { i32, i32, %struct.iv*, %struct.tree_node*, %struct.tree_node**, %struct.bitmap_head_def*, i32, %struct.cost_pair*, %struct.iv_cand* }
	%struct.ivopts_data = type { %struct.loop*, %struct.htab*, i32, %struct.version_info*, %struct.bitmap_head_def*, i32, %struct.varray_head_tag*, %struct.varray_head_tag*, %struct.bitmap_head_def*, i1 }
	%struct.lang_decl = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { i8*, i32 }
	%struct.loop = type { i32, %struct.basic_block_def*, %struct.basic_block_def*, %struct.basic_block_def*, %struct.lpt_decision, i32, i32, %struct.edge_def**, i32, %struct.basic_block_def*, %struct.basic_block_def*, i32, %struct.edge_def**, i32, %struct.edge_def**, i32, %struct.simple_bitmap_def*, i32, %struct.loop**, i32, %struct.loop*, %struct.loop*, %struct.loop*, %struct.loop*, i32, i8*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i32, %struct.tree_node*, %struct.tree_node*, %struct.nb_iter_bound*, %struct.edge_def*, i1 }
	%struct.lpt_decision = type { i32, i32 }
	%struct.machine_function = type { %struct.stack_local_entry*, i8*, %struct.rtx_def*, i32, i32, i32, i32, i32 }
	%struct.nb_iter_bound = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.nb_iter_bound* }
	%struct.obstack = type { i32, %struct._obstack_chunk*, i8*, i8*, i8*, i32, i32, %struct._obstack_chunk* (i8*, i32)*, void (i8*, %struct._obstack_chunk*)*, i8*, i8 }
	%struct.reorder_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.basic_block_def*, %struct.basic_block_def*, %struct.basic_block_def*, i32, i32, i32 }
	%struct.rtvec_def = type { i32, [1 x %struct.rtx_def*] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.simple_bitmap_def = type { i32, i32, i32, [1 x i64] }
	%struct.stack_local_entry = type opaque
	%struct.stmt_ann_d = type { %struct.tree_ann_common_d, i8, %struct.basic_block_def*, %struct.stmt_operands_d, %struct.dataflow_d*, %struct.bitmap_head_def*, i32 }
	%struct.stmt_operands_d = type { %struct.def_optype_d*, %struct.def_optype_d*, %struct.v_may_def_optype_d*, %struct.vuse_optype_d*, %struct.v_may_def_optype_d* }
	%struct.temp_slot = type opaque
	%struct.tree_ann_common_d = type { i32, i8*, %struct.tree_node* }
	%struct.tree_ann_d = type { %struct.stmt_ann_d }
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_ann_d*, i8, i8, i8, i8, i8 }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, %struct.tree_node*, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, i32, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.u = type { [1 x i64] }
	%struct.v_def_use_operand_type_t = type { %struct.tree_node*, %struct.tree_node* }
	%struct.v_may_def_optype_d = type { i32, [1 x %struct.v_def_use_operand_type_t] }
	%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_head_tag = type { i32, i32, i32, i8*, %struct.u }
	%struct.version_info = type { %struct.tree_node*, %struct.iv*, i1, i32, i1 }
	%struct.vuse_optype_d = type { i32, [1 x %struct.tree_node*] }

define i1 @determine_use_iv_cost(%struct.ivopts_data* %data, %struct.iv_use* %use, %struct.iv_cand* %cand) {
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
	%tmp44 = call i32 @force_var_cost( %struct.ivopts_data* %data, %struct.tree_node* null, %struct.bitmap_head_def** null )		; <i32> [#uses=2]
	%tmp46 = udiv i32 %tmp44, 5		; <i32> [#uses=1]
	call void @set_use_iv_cost( %struct.ivopts_data* %data, %struct.iv_use* %use, %struct.iv_cand* %cand, i32 %tmp46, %struct.bitmap_head_def* null )
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
	%tmp193 = call i1 @determine_use_iv_cost_generic( %struct.ivopts_data* %data, %struct.iv_use* %use, %struct.iv_cand* %cand )		; <i1> [#uses=0]
	ret i1 false
}

declare void @set_use_iv_cost(%struct.ivopts_data*, %struct.iv_use*, %struct.iv_cand*, i32, %struct.bitmap_head_def*)

declare i32 @force_var_cost(%struct.ivopts_data*, %struct.tree_node*, %struct.bitmap_head_def**)

declare i1 @determine_use_iv_cost_generic(%struct.ivopts_data*, %struct.iv_use*, %struct.iv_cand*)
