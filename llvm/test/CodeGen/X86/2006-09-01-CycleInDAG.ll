; RUN: llc < %s
target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8"
	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.VEC_edge = type { i32, i32, [1 x %struct.edge_def*] }
	%struct.VEC_tree = type { i32, i32, [1 x %struct.tree_node*] }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
	%struct._obstack_chunk = type { i8*, %struct._obstack_chunk*, [4 x i8] }
	%struct._var_map = type { %struct.partition_def*, i32*, i32*, %struct.tree_node**, i32, i32, i32* }
	%struct.basic_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.tree_node*, %struct.VEC_edge*, %struct.VEC_edge*, %struct.bitmap_head_def*, %struct.bitmap_head_def*, i8*, %struct.loop*, [2 x %struct.et_node*], %struct.basic_block_def*, %struct.basic_block_def*, %struct.reorder_block_def*, %struct.bb_ann_d*, i64, i32, i32, i32, i32 }
	%struct.bb_ann_d = type { %struct.tree_node*, i8, %struct.edge_prediction* }
	%struct.bitmap_element_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, [4 x i32] }
	%struct.bitmap_head_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, %struct.bitmap_obstack* }
	%struct.bitmap_iterator = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, i32 }
	%struct.bitmap_obstack = type { %struct.bitmap_element_def*, %struct.bitmap_head_def*, %struct.obstack }
	%struct.block_stmt_iterator = type { %struct.tree_stmt_iterator, %struct.basic_block_def* }
	%struct.coalesce_list_d = type { %struct._var_map*, %struct.partition_pair_d**, i1 }
	%struct.conflict_graph_def = type opaque
	%struct.dataflow_d = type { %struct.varray_head_tag*, [2 x %struct.tree_node*] }
	%struct.def_operand_ptr = type { %struct.tree_node** }
	%struct.def_optype_d = type { i32, [1 x %struct.def_operand_ptr] }
	%struct.die_struct = type opaque
	%struct.edge_def = type { %struct.basic_block_def*, %struct.basic_block_def*, %struct.edge_def_insns, i8*, %struct.__sbuf*, i32, i32, i64, i32 }
	%struct.edge_def_insns = type { %struct.rtx_def* }
	%struct.edge_iterator = type { i32, %struct.VEC_edge** }
	%struct.edge_prediction = type { %struct.edge_prediction*, %struct.edge_def*, i32, i32 }
	%struct.eh_status = type opaque
	%struct.elt_list = type opaque
	%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, i32, %struct.__sbuf, i32, i8*, %struct.rtx_def** }
	%struct.et_node = type opaque
	%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i8, i32, i64, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, i32, %struct.var_refs_queue*, i32, i32, %struct.rtvec_def*, %struct.tree_node*, i32, i32, i32, %struct.machine_function*, i32, i32, i1, i1, %struct.language_function*, %struct.rtx_def*, i32, i32, i32, i32, %struct.__sbuf, %struct.varray_head_tag*, %struct.tree_node*, i8, i8, i8 }
	%struct.ht_identifier = type { i8*, i32, i32 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.lang_type = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { i8*, i32 }
	%struct.loop = type opaque
	%struct.machine_function = type { i32, i32, i8*, i32, i32 }
	%struct.obstack = type { i32, %struct._obstack_chunk*, i8*, i8*, i8*, i32, i32, %struct._obstack_chunk* (i8*, i32)*, void (i8*, %struct._obstack_chunk*)*, i8*, i8 }
	%struct.partition_def = type { i32, [1 x %struct.partition_elem] }
	%struct.partition_elem = type { i32, %struct.partition_elem*, i32 }
	%struct.partition_pair_d = type { i32, i32, i32, %struct.partition_pair_d* }
	%struct.phi_arg_d = type { %struct.tree_node*, i1 }
	%struct.pointer_set_t = type opaque
	%struct.ptr_info_def = type { i8, %struct.bitmap_head_def*, %struct.tree_node* }
	%struct.real_value = type opaque
	%struct.reg_info_def = type opaque
	%struct.reorder_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.basic_block_def*, %struct.basic_block_def*, %struct.basic_block_def*, i32, i32, i32 }
	%struct.rtvec_def = type opaque
	%struct.rtx_def = type opaque
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.simple_bitmap_def = type { i32, i32, i32, [1 x i64] }
	%struct.ssa_op_iter = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.stmt_operands_d*, i1 }
	%struct.stmt_ann_d = type { %struct.tree_ann_common_d, i8, %struct.basic_block_def*, %struct.stmt_operands_d, %struct.dataflow_d*, %struct.bitmap_head_def*, i32 }
	%struct.stmt_operands_d = type { %struct.def_optype_d*, %struct.def_optype_d*, %struct.v_may_def_optype_d*, %struct.vuse_optype_d*, %struct.v_may_def_optype_d* }
	%struct.temp_slot = type opaque
	%struct.tree_ann_common_d = type { i32, i8*, %struct.tree_node* }
	%struct.tree_ann_d = type { %struct.stmt_ann_d }
	%struct.tree_binfo = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.VEC_tree*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.VEC_tree }
	%struct.tree_block = type { %struct.tree_common, i8, [3 x i8], %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_ann_d*, i8, i8, i8, i8, i8 }
	%struct.tree_complex = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_decl = type { %struct.tree_common, %struct.__sbuf, i32, %struct.tree_node*, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, i32, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u1_a = type { i32 }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_exp = type { %struct.tree_common, %struct.__sbuf*, i32, %struct.tree_node*, [1 x %struct.tree_node*] }
	%struct.tree_identifier = type { %struct.tree_common, %struct.ht_identifier }
	%struct.tree_int_cst = type { %struct.tree_common, %struct.tree_int_cst_lowhi }
	%struct.tree_int_cst_lowhi = type { i64, i64 }
	%struct.tree_list = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_live_info_d = type { %struct._var_map*, %struct.bitmap_head_def*, %struct.bitmap_head_def**, i32, %struct.bitmap_head_def** }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.tree_partition_associator_d = type { %struct.varray_head_tag*, %struct.varray_head_tag*, i32*, i32*, i32, i32, %struct._var_map* }
	%struct.tree_phi_node = type { %struct.tree_common, %struct.tree_node*, i32, i32, i32, %struct.basic_block_def*, %struct.dataflow_d*, [1 x %struct.phi_arg_d] }
	%struct.tree_real_cst = type { %struct.tree_common, %struct.real_value* }
	%struct.tree_ssa_name = type { %struct.tree_common, %struct.tree_node*, i32, %struct.ptr_info_def*, %struct.tree_node*, i8* }
	%struct.tree_statement_list = type { %struct.tree_common, %struct.tree_statement_list_node*, %struct.tree_statement_list_node* }
	%struct.tree_statement_list_node = type { %struct.tree_statement_list_node*, %struct.tree_statement_list_node*, %struct.tree_node* }
	%struct.tree_stmt_iterator = type { %struct.tree_statement_list_node*, %struct.tree_node* }
	%struct.tree_string = type { %struct.tree_common, i32, [1 x i8] }
	%struct.tree_type = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i32, i16, i8, i8, i32, %struct.tree_node*, %struct.tree_node*, %struct.tree_decl_u1_a, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_type* }
	%struct.tree_type_symtab = type { i32 }
	%struct.tree_value_handle = type { %struct.tree_common, %struct.value_set*, i32 }
	%struct.tree_vec = type { %struct.tree_common, i32, [1 x %struct.tree_node*] }
	%struct.tree_vector = type { %struct.tree_common, %struct.tree_node* }
	%struct.use_operand_ptr = type { %struct.tree_node** }
	%struct.use_optype_d = type { i32, [1 x %struct.def_operand_ptr] }
	%struct.v_def_use_operand_type_t = type { %struct.tree_node*, %struct.tree_node* }
	%struct.v_may_def_optype_d = type { i32, [1 x %struct.v_def_use_operand_type_t] }
	%struct.v_must_def_optype_d = type { i32, [1 x %struct.v_def_use_operand_type_t] }
	%struct.value_set = type opaque
	%struct.var_ann_d = type { %struct.tree_ann_common_d, i8, i8, %struct.tree_node*, %struct.varray_head_tag*, i32, i32, i32, %struct.tree_node*, %struct.tree_node* }
	%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_data = type { [1 x i64] }
	%struct.varray_head_tag = type { i32, i32, i32, i8*, %struct.varray_data }
	%struct.vuse_optype_d = type { i32, [1 x %struct.tree_node*] }
@basic_block_info = external global %struct.varray_head_tag*		; <%struct.varray_head_tag**> [#uses=1]

define void @calculate_live_on_entry_cond_true3632(%struct.varray_head_tag* %stack3023.6, i32* %tmp3629, %struct.VEC_edge*** %tmp3397.out) {
newFuncRoot:
	br label %cond_true3632

bb3502.exitStub:		; preds = %cond_true3632
	store %struct.VEC_edge** %tmp3397, %struct.VEC_edge*** %tmp3397.out
	ret void

cond_true3632:		; preds = %newFuncRoot
	%tmp3378 = load i32, i32* %tmp3629		; <i32> [#uses=1]
	%tmp3379 = add i32 %tmp3378, -1		; <i32> [#uses=1]
	%tmp3381 = getelementptr %struct.varray_head_tag, %struct.varray_head_tag* %stack3023.6, i32 0, i32 4		; <%struct.varray_data*> [#uses=1]
	%tmp3382 = bitcast %struct.varray_data* %tmp3381 to [1 x i32]*		; <[1 x i32]*> [#uses=1]
	%gep.upgrd.1 = zext i32 %tmp3379 to i64		; <i64> [#uses=1]
	%tmp3383 = getelementptr [1 x i32], [1 x i32]* %tmp3382, i32 0, i64 %gep.upgrd.1		; <i32*> [#uses=1]
	%tmp3384 = load i32, i32* %tmp3383		; <i32> [#uses=1]
	%tmp3387 = load i32, i32* %tmp3629		; <i32> [#uses=1]
	%tmp3388 = add i32 %tmp3387, -1		; <i32> [#uses=1]
	store i32 %tmp3388, i32* %tmp3629
	%tmp3391 = load %struct.varray_head_tag*, %struct.varray_head_tag** @basic_block_info		; <%struct.varray_head_tag*> [#uses=1]
	%tmp3393 = getelementptr %struct.varray_head_tag, %struct.varray_head_tag* %tmp3391, i32 0, i32 4		; <%struct.varray_data*> [#uses=1]
	%tmp3394 = bitcast %struct.varray_data* %tmp3393 to [1 x %struct.basic_block_def*]*		; <[1 x %struct.basic_block_def*]*> [#uses=1]
	%tmp3395 = getelementptr [1 x %struct.basic_block_def*], [1 x %struct.basic_block_def*]* %tmp3394, i32 0, i32 %tmp3384		; <%struct.basic_block_def**> [#uses=1]
	%tmp3396 = load %struct.basic_block_def*, %struct.basic_block_def** %tmp3395		; <%struct.basic_block_def*> [#uses=1]
	%tmp3397 = getelementptr %struct.basic_block_def, %struct.basic_block_def* %tmp3396, i32 0, i32 3		; <%struct.VEC_edge**> [#uses=1]
	br label %bb3502.exitStub
}
