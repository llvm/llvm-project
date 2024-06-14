; RUN: llc < %s
target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8"
	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.VEC_edge = type { i32, i32, [1 x ptr] }
	%struct.VEC_tree = type { i32, i32, [1 x ptr] }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ptr, i32 }
	%struct._obstack_chunk = type { ptr, ptr, [4 x i8] }
	%struct._var_map = type { ptr, ptr, ptr, ptr, i32, i32, ptr }
	%struct.basic_block_def = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, [2 x ptr], ptr, ptr, ptr, ptr, i64, i32, i32, i32, i32 }
	%struct.bb_ann_d = type { ptr, i8, ptr }
	%struct.bitmap_element_def = type { ptr, ptr, i32, [4 x i32] }
	%struct.bitmap_head_def = type { ptr, ptr, i32, ptr }
	%struct.bitmap_iterator = type { ptr, ptr, i32, i32 }
	%struct.bitmap_obstack = type { ptr, ptr, %struct.obstack }
	%struct.block_stmt_iterator = type { %struct.tree_stmt_iterator, ptr }
	%struct.coalesce_list_d = type { ptr, ptr, i1 }
	%struct.conflict_graph_def = type opaque
	%struct.dataflow_d = type { ptr, [2 x ptr] }
	%struct.def_operand_ptr = type { ptr }
	%struct.def_optype_d = type { i32, [1 x %struct.def_operand_ptr] }
	%struct.die_struct = type opaque
	%struct.edge_def = type { ptr, ptr, %struct.edge_def_insns, ptr, ptr, i32, i32, i64, i32 }
	%struct.edge_def_insns = type { ptr }
	%struct.edge_iterator = type { i32, ptr }
	%struct.edge_prediction = type { ptr, ptr, i32, i32 }
	%struct.eh_status = type opaque
	%struct.elt_list = type opaque
	%struct.emit_status = type { i32, i32, ptr, ptr, ptr, i32, %struct.__sbuf, i32, ptr, ptr }
	%struct.et_node = type opaque
	%struct.expr_status = type { i32, i32, i32, ptr, ptr, ptr }
	%struct.function = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, ptr, %struct.CUMULATIVE_ARGS, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8, i32, i64, ptr, ptr, ptr, ptr, ptr, i32, ptr, i32, i32, ptr, ptr, i32, i32, i32, ptr, i32, i32, i1, i1, ptr, ptr, i32, i32, i32, i32, %struct.__sbuf, ptr, ptr, i8, i8, i8 }
	%struct.ht_identifier = type { ptr, i32, i32 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.lang_type = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { ptr, i32 }
	%struct.loop = type opaque
	%struct.machine_function = type { i32, i32, ptr, i32, i32 }
	%struct.obstack = type { i32, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, i8 }
	%struct.partition_def = type { i32, [1 x %struct.partition_elem] }
	%struct.partition_elem = type { i32, ptr, i32 }
	%struct.partition_pair_d = type { i32, i32, i32, ptr }
	%struct.phi_arg_d = type { ptr, i1 }
	%struct.pointer_set_t = type opaque
	%struct.ptr_info_def = type { i8, ptr, ptr }
	%struct.real_value = type opaque
	%struct.reg_info_def = type opaque
	%struct.reorder_block_def = type { ptr, ptr, ptr, ptr, ptr, i32, i32, i32 }
	%struct.rtvec_def = type opaque
	%struct.rtx_def = type opaque
	%struct.sequence_stack = type { ptr, ptr, ptr }
	%struct.simple_bitmap_def = type { i32, i32, i32, [1 x i64] }
	%struct.ssa_op_iter = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, i1 }
	%struct.stmt_ann_d = type { %struct.tree_ann_common_d, i8, ptr, %struct.stmt_operands_d, ptr, ptr, i32 }
	%struct.stmt_operands_d = type { ptr, ptr, ptr, ptr, ptr }
	%struct.temp_slot = type opaque
	%struct.tree_ann_common_d = type { i32, ptr, ptr }
	%struct.tree_ann_d = type { %struct.stmt_ann_d }
	%struct.tree_binfo = type { %struct.tree_common, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %struct.VEC_tree }
	%struct.tree_block = type { %struct.tree_common, i8, [3 x i8], ptr, ptr, ptr, ptr, ptr, ptr }
	%struct.tree_common = type { ptr, ptr, ptr, i8, i8, i8, i8, i8 }
	%struct.tree_complex = type { %struct.tree_common, ptr, ptr }
	%struct.tree_decl = type { %struct.tree_common, %struct.__sbuf, i32, ptr, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, %struct.tree_decl_u2, ptr, ptr, i64, ptr }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u1_a = type { i32 }
	%struct.tree_decl_u2 = type { ptr }
	%struct.tree_exp = type { %struct.tree_common, ptr, i32, ptr, [1 x ptr] }
	%struct.tree_identifier = type { %struct.tree_common, %struct.ht_identifier }
	%struct.tree_int_cst = type { %struct.tree_common, %struct.tree_int_cst_lowhi }
	%struct.tree_int_cst_lowhi = type { i64, i64 }
	%struct.tree_list = type { %struct.tree_common, ptr, ptr }
	%struct.tree_live_info_d = type { ptr, ptr, ptr, i32, ptr }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.tree_partition_associator_d = type { ptr, ptr, ptr, ptr, i32, i32, ptr }
	%struct.tree_phi_node = type { %struct.tree_common, ptr, i32, i32, i32, ptr, ptr, [1 x %struct.phi_arg_d] }
	%struct.tree_real_cst = type { %struct.tree_common, ptr }
	%struct.tree_ssa_name = type { %struct.tree_common, ptr, i32, ptr, ptr, ptr }
	%struct.tree_statement_list = type { %struct.tree_common, ptr, ptr }
	%struct.tree_statement_list_node = type { ptr, ptr, ptr }
	%struct.tree_stmt_iterator = type { ptr, ptr }
	%struct.tree_string = type { %struct.tree_common, i32, [1 x i8] }
	%struct.tree_type = type { %struct.tree_common, ptr, ptr, ptr, ptr, i32, i16, i8, i8, i32, ptr, ptr, %struct.tree_decl_u1_a, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, ptr }
	%struct.tree_type_symtab = type { i32 }
	%struct.tree_value_handle = type { %struct.tree_common, ptr, i32 }
	%struct.tree_vec = type { %struct.tree_common, i32, [1 x ptr] }
	%struct.tree_vector = type { %struct.tree_common, ptr }
	%struct.use_operand_ptr = type { ptr }
	%struct.use_optype_d = type { i32, [1 x %struct.def_operand_ptr] }
	%struct.v_def_use_operand_type_t = type { ptr, ptr }
	%struct.v_may_def_optype_d = type { i32, [1 x %struct.v_def_use_operand_type_t] }
	%struct.v_must_def_optype_d = type { i32, [1 x %struct.v_def_use_operand_type_t] }
	%struct.value_set = type opaque
	%struct.var_ann_d = type { %struct.tree_ann_common_d, i8, i8, ptr, ptr, i32, i32, i32, ptr, ptr }
	%struct.var_refs_queue = type { ptr, i32, i32, ptr }
	%struct.varasm_status = type opaque
	%struct.varray_data = type { [1 x i64] }
	%struct.varray_head_tag = type { i32, i32, i32, ptr, %struct.varray_data }
	%struct.vuse_optype_d = type { i32, [1 x ptr] }
@basic_block_info = external global ptr		; <ptr> [#uses=1]

define void @calculate_live_on_entry_cond_true3632(ptr %stack3023.6, ptr %tmp3629, ptr %tmp3397.out) {
newFuncRoot:
	br label %cond_true3632

bb3502.exitStub:		; preds = %cond_true3632
	store ptr %tmp3397, ptr %tmp3397.out
	ret void

cond_true3632:		; preds = %newFuncRoot
	%tmp3378 = load i32, ptr %tmp3629		; <i32> [#uses=1]
	%tmp3379 = add i32 %tmp3378, -1		; <i32> [#uses=1]
	%tmp3381 = getelementptr %struct.varray_head_tag, ptr %stack3023.6, i32 0, i32 4		; <ptr> [#uses=1]
	%gep.upgrd.1 = zext i32 %tmp3379 to i64		; <i64> [#uses=1]
	%tmp3383 = getelementptr [1 x i32], ptr %tmp3381, i32 0, i64 %gep.upgrd.1		; <ptr> [#uses=1]
	%tmp3384 = load i32, ptr %tmp3383		; <i32> [#uses=1]
	%tmp3387 = load i32, ptr %tmp3629		; <i32> [#uses=1]
	%tmp3388 = add i32 %tmp3387, -1		; <i32> [#uses=1]
	store i32 %tmp3388, ptr %tmp3629
	%tmp3391 = load ptr, ptr @basic_block_info		; <ptr> [#uses=1]
	%tmp3393 = getelementptr %struct.varray_head_tag, ptr %tmp3391, i32 0, i32 4		; <ptr> [#uses=1]
	%tmp3395 = getelementptr [1 x ptr], ptr %tmp3393, i32 0, i32 %tmp3384		; <ptr> [#uses=1]
	%tmp3396 = load ptr, ptr %tmp3395		; <ptr> [#uses=1]
	%tmp3397 = getelementptr %struct.basic_block_def, ptr %tmp3396, i32 0, i32 3		; <ptr> [#uses=1]
	br label %bb3502.exitStub
}
