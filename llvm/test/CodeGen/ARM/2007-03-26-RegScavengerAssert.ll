; RUN: llvm-as < %s | llc -march=arm
; PR1266

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "arm-linux-gnueabi"
	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32 }
	%struct.FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct.FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i32, [52 x i8] }
	%struct.VEC_edge = type { i32, i32, [1 x %struct.edge_def*] }
	%struct.VEC_tree = type { i32, i32, [1 x %struct.tree_node*] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct.FILE*, i32 }
	%struct._obstack_chunk = type { i8*, %struct._obstack_chunk*, [4 x i8] }
	%struct.addr_diff_vec_flags = type { i8, i8, i8, i8 }
	%struct.arm_stack_offsets = type { i32, i32, i32, i32, i32 }
	%struct.attribute_spec = type { i8*, i32, i32, i8, i8, i8, %struct.tree_node* (%struct.tree_node**, %struct.tree_node*, %struct.tree_node*, i32, i8*)* }
	%struct.basic_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.tree_node*, %struct.VEC_edge*, %struct.VEC_edge*, %struct.bitmap_head_def*, %struct.bitmap_head_def*, i8*, %struct.loop*, [2 x %struct.et_node*], %struct.basic_block_def*, %struct.basic_block_def*, %struct.reorder_block_def*, %struct.bb_ann_d*, i64, i32, i32, i32, i32 }
	%struct.bb_ann_d = type { %struct.tree_node*, i8, %struct.edge_prediction* }
	%struct.bitmap_element_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, [4 x i32] }
	%struct.bitmap_head_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, %struct.bitmap_obstack* }
	%struct.bitmap_obstack = type { %struct.bitmap_element_def*, %struct.bitmap_head_def*, %struct.obstack }
	%struct.cgraph_edge = type { %struct.cgraph_node*, %struct.cgraph_node*, %struct.cgraph_edge*, %struct.cgraph_edge*, %struct.cgraph_edge*, %struct.cgraph_edge*, %struct.tree_node*, i8*, i8* }
	%struct.cgraph_global_info = type { %struct.cgraph_node*, i32, i8 }
	%struct.cgraph_local_info = type { i32, i8, i8, i8, i8, i8, i8, i8 }
	%struct.cgraph_node = type { %struct.tree_node*, %struct.cgraph_edge*, %struct.cgraph_edge*, %struct.cgraph_node*, %struct.cgraph_node*, %struct.cgraph_node*, %struct.cgraph_node*, %struct.cgraph_node*, %struct.cgraph_node*, %struct.cgraph_node*, i8*, %struct.cgraph_local_info, %struct.cgraph_global_info, %struct.cgraph_rtl_info, i32, i8, i8, i8, i8, i8 }
	%struct.cgraph_rtl_info = type { i32, i8, i8 }
	%struct.cl_perfunc_opts = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.cselib_val_struct = type opaque
	%struct.dataflow_d = type { %struct.varray_head_tag*, [2 x %struct.tree_node*] }
	%struct.def_operand_ptr = type { %struct.tree_node** }
	%struct.def_optype_d = type { i32, [1 x %struct.def_operand_ptr] }
	%struct.diagnostic_context = type { %struct.pretty_printer*, [8 x i32], i8, i8, i8, void (%struct.diagnostic_context*, %struct.diagnostic_info*)*, void (%struct.diagnostic_context*, %struct.diagnostic_info*)*, void (i8*, i8**)*, %struct.tree_node*, i32, i32 }
	%struct.diagnostic_info = type { %struct.text_info, %struct.location_t, i32 }
	%struct.die_struct = type opaque
	%struct.edge_def = type { %struct.basic_block_def*, %struct.basic_block_def*, %struct.edge_def_insns, i8*, %struct.location_t*, i32, i32, i64, i32 }
	%struct.edge_def_insns = type { %struct.rtx_def* }
	%struct.edge_prediction = type { %struct.edge_prediction*, %struct.edge_def*, i32, i32 }
	%struct.eh_status = type opaque
	%struct.elt_list = type opaque
	%struct.elt_t = type { %struct.tree_node*, %struct.tree_node* }
	%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, i32, %struct.location_t, i32, i8*, %struct.rtx_def** }
	%struct.et_node = type opaque
	%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i8, i32, i64, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, i32, %struct.var_refs_queue*, i32, i32, %struct.rtvec_def*, %struct.tree_node*, i32, i32, i32, %struct.machine_function*, i32, i32, i8, i8, %struct.language_function*, %struct.rtx_def*, i32, i32, i32, i32, %struct.location_t, %struct.varray_head_tag*, %struct.tree_node*, i8, i8, i8 }
	%struct.ggc_root_tab = type { i8*, i32, i32, void (i8*)*, void (i8*)* }
	%struct.gimplify_ctx = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.varray_head_tag*, %struct.htab*, i32, i8, i8 }
	%struct.gimplify_init_ctor_preeval_data = type { %struct.tree_node*, i32 }
	%struct.ht_identifier = type { i8*, i32, i32 }
	%struct.htab = type { i32 (i8*)*, i32 (i8*, i8*)*, void (i8*)*, i8**, i32, i32, i32, i32, i32, i8* (i32, i32)*, void (i8*)*, i8*, i8* (i8*, i32, i32)*, void (i8*, i8*)*, i32 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.lang_hooks = type { i8*, i32, i32 (i32)*, i32 (i32, i8**)*, void (%struct.diagnostic_context*)*, i32 (i32, i8*, i32)*, i8 (i8*, i32) zeroext *, i8 (i8**) zeroext *, i8 () zeroext *, void ()*, void ()*, void (i32)*, void ()*, i64 (%struct.tree_node*)*, %struct.tree_node* (%struct.tree_node*)*, %struct.rtx_def* (%struct.tree_node*, %struct.rtx_def*, i32, i32, %struct.rtx_def**)*, i32 (%struct.tree_node*)*, %struct.tree_node* (%struct.tree_node*)*, i32 (%struct.rtx_def*, %struct.tree_node*)*, void (%struct.tree_node*)*, i8 (%struct.tree_node*) zeroext *, %struct.tree_node* (%struct.tree_node*)*, void (%struct.tree_node*)*, void (%struct.tree_node*)*, i8 () zeroext *, i8, i8, void ()*, void (%struct.FILE*, %struct.tree_node*, i32)*, void (%struct.FILE*, %struct.tree_node*, i32)*, void (%struct.FILE*, %struct.tree_node*, i32)*, void (%struct.FILE*, %struct.tree_node*, i32)*, i8* (%struct.tree_node*, i32)*, i32 (%struct.tree_node*, %struct.tree_node*)*, %struct.tree_node* (%struct.tree_node*)*, void (%struct.diagnostic_context*, i8*)*, %struct.tree_node* (%struct.tree_node*)*, i64 (i64)*, %struct.attribute_spec*, %struct.attribute_spec*, %struct.attribute_spec*, i32 (%struct.tree_node*)*, %struct.lang_hooks_for_functions, %struct.lang_hooks_for_tree_inlining, %struct.lang_hooks_for_callgraph, %struct.lang_hooks_for_tree_dump, %struct.lang_hooks_for_decls, %struct.lang_hooks_for_types, i32 (%struct.tree_node**, %struct.tree_node**, %struct.tree_node**)*, %struct.tree_node* (%struct.tree_node*, %struct.tree_node*)*, %struct.tree_node* (i8*, %struct.tree_node*, i32, i32, i8*, %struct.tree_node*)* }
	%struct.lang_hooks_for_callgraph = type { %struct.tree_node* (%struct.tree_node**, i32*, %struct.tree_node*)*, void (%struct.tree_node*)* }
	%struct.lang_hooks_for_decls = type { i32 ()*, void (%struct.tree_node*)*, %struct.tree_node* (%struct.tree_node*)*, %struct.tree_node* ()*, i8 (%struct.tree_node*) zeroext *, void ()*, void (%struct.tree_node*)*, i8 (%struct.tree_node*) zeroext *, i8* (%struct.tree_node*)* }
	%struct.lang_hooks_for_functions = type { void (%struct.function*)*, void (%struct.function*)*, void (%struct.function*)*, void (%struct.function*)*, i8 (%struct.tree_node*) zeroext * }
	%struct.lang_hooks_for_tree_dump = type { i8 (i8*, %struct.tree_node*) zeroext *, i32 (%struct.tree_node*)* }
	%struct.lang_hooks_for_tree_inlining = type { %struct.tree_node* (%struct.tree_node**, i32*, %struct.tree_node* (%struct.tree_node**, i32*, i8*)*, i8*, %struct.pointer_set_t*)*, i32 (%struct.tree_node**)*, i32 (%struct.tree_node*)*, %struct.tree_node* (i8*, %struct.tree_node*)*, i32 (%struct.tree_node*, %struct.tree_node*)*, i32 (%struct.tree_node*)*, i8 (%struct.tree_node*, %struct.tree_node*) zeroext *, i32 (%struct.tree_node*)*, void (%struct.tree_node*)*, %struct.tree_node* (%struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i32)* }
	%struct.lang_hooks_for_types = type { %struct.tree_node* (i32)*, %struct.tree_node* (i32, i32)*, %struct.tree_node* (i32, i32)*, %struct.tree_node* (%struct.tree_node*)*, %struct.tree_node* (%struct.tree_node*)*, %struct.tree_node* (i32, %struct.tree_node*)*, %struct.tree_node* (%struct.tree_node*)*, void (%struct.tree_node*, i8*)*, void (%struct.tree_node*, %struct.tree_node*)*, %struct.tree_node* (%struct.tree_node*)*, i8 }
	%struct.lang_type = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { i8*, i32 }
	%struct.loop = type opaque
	%struct.machine_function = type { %struct.rtx_def*, i32, i32, i32, %struct.arm_stack_offsets, i32, i32, i32, [14 x %struct.rtx_def*] }
	%struct.mem_attrs = type { i64, %struct.tree_node*, %struct.rtx_def*, %struct.rtx_def*, i32 }
	%struct.obstack = type { i32, %struct._obstack_chunk*, i8*, i8*, i8*, i32, i32, %struct._obstack_chunk* (i8*, i32)*, void (i8*, %struct._obstack_chunk*)*, i8*, i8 }
	%struct.output_buffer = type { %struct.obstack, %struct.FILE*, i32, [128 x i8] }
	%struct.phi_arg_d = type { %struct.tree_node*, i8 }
	%struct.pointer_set_t = type opaque
	%struct.pretty_printer = type { %struct.output_buffer*, i8*, i32, i32, i32, i32, i32, i8 (%struct.pretty_printer*, %struct.text_info*) zeroext *, i8, i8 }
	%struct.ptr_info_def = type { i8, %struct.bitmap_head_def*, %struct.tree_node* }
	%struct.real_value = type { i8, [3 x i8], [5 x i32] }
	%struct.reg_attrs = type { %struct.tree_node*, i64 }
	%struct.reg_info_def = type opaque
	%struct.reorder_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.basic_block_def*, %struct.basic_block_def*, %struct.basic_block_def*, i32, i32, i32 }
	%struct.rtunion = type { i32 }
	%struct.rtvec_def = type { i32, [1 x %struct.rtx_def*] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.stmt_ann_d = type { %struct.tree_ann_common_d, i8, %struct.basic_block_def*, %struct.stmt_operands_d, %struct.dataflow_d*, %struct.bitmap_head_def*, i32 }
	%struct.stmt_operands_d = type { %struct.def_optype_d*, %struct.def_optype_d*, %struct.v_may_def_optype_d*, %struct.vuse_optype_d*, %struct.v_may_def_optype_d* }
	%struct.temp_slot = type opaque
	%struct.text_info = type { i8*, i8**, i32 }
	%struct.tree_ann_common_d = type { i32, i8*, %struct.tree_node* }
	%struct.tree_ann_d = type { %struct.stmt_ann_d }
	%struct.tree_binfo = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.VEC_tree*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.VEC_tree }
	%struct.tree_block = type { %struct.tree_common, i8, [3 x i8], %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_ann_d*, i8, i8, i8, i8, i8 }
	%struct.tree_complex = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, %struct.tree_node*, i8, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, i32, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u1_a = type { i32 }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_exp = type { %struct.tree_common, %struct.location_t*, i32, %struct.tree_node*, [1 x %struct.tree_node*] }
	%struct.tree_identifier = type { %struct.tree_common, %struct.ht_identifier }
	%struct.tree_int_cst = type { %struct.tree_common, %struct.tree_int_cst_lowhi }
	%struct.tree_int_cst_lowhi = type { i64, i64 }
	%struct.tree_list = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.tree_phi_node = type { %struct.tree_common, %struct.tree_node*, i32, i32, i32, %struct.basic_block_def*, %struct.dataflow_d*, [1 x %struct.phi_arg_d] }
	%struct.tree_real_cst = type { %struct.tree_common, %struct.real_value* }
	%struct.tree_ssa_name = type { %struct.tree_common, %struct.tree_node*, i32, %struct.ptr_info_def*, %struct.tree_node*, i8* }
	%struct.tree_statement_list = type { %struct.tree_common, %struct.tree_statement_list_node*, %struct.tree_statement_list_node* }
	%struct.tree_statement_list_node = type { %struct.tree_statement_list_node*, %struct.tree_statement_list_node*, %struct.tree_node* }
	%struct.tree_stmt_iterator = type { %struct.tree_statement_list_node*, %struct.tree_node* }
	%struct.tree_string = type { %struct.tree_common, i32, [1 x i8] }
	%struct.tree_type = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i32, i16, i8, i8, i32, %struct.tree_node*, %struct.tree_node*, %struct.rtunion, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_type* }
	%struct.tree_type_symtab = type { i32 }
	%struct.tree_value_handle = type { %struct.tree_common, %struct.value_set*, i32 }
	%struct.tree_vec = type { %struct.tree_common, i32, [1 x %struct.tree_node*] }
	%struct.tree_vector = type { %struct.tree_common, %struct.tree_node* }
	%struct.u = type { [1 x i64] }
	%struct.use_operand_ptr = type { %struct.tree_node** }
	%struct.use_optype_d = type { i32, [1 x %struct.def_operand_ptr] }
	%struct.v_def_use_operand_type_t = type { %struct.tree_node*, %struct.tree_node* }
	%struct.v_may_def_optype_d = type { i32, [1 x %struct.elt_t] }
	%struct.v_must_def_optype_d = type { i32, [1 x %struct.elt_t] }
	%struct.value_set = type opaque
	%struct.var_ann_d = type { %struct.tree_ann_common_d, i8, i8, %struct.tree_node*, %struct.varray_head_tag*, i32, i32, i32, %struct.tree_node*, %struct.tree_node* }
	%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_data = type { [1 x i64] }
	%struct.varray_head_tag = type { i32, i32, i32, i8*, %struct.u }
	%struct.vuse_optype_d = type { i32, [1 x %struct.tree_node*] }
@gt_pch_rs_gt_gimplify_h = external global [2 x %struct.ggc_root_tab]		; <[2 x %struct.ggc_root_tab]*> [#uses=0]
@tmp_var_id_num = external global i32		; <i32*> [#uses=0]
@gt_ggc_r_gt_gimplify_h = external global [1 x %struct.ggc_root_tab]		; <[1 x %struct.ggc_root_tab]*> [#uses=0]
@__FUNCTION__.19956 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@str = external global [42 x i8]		; <[42 x i8]*> [#uses=1]
@__FUNCTION__.19974 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@gimplify_ctxp = external global %struct.gimplify_ctx*		; <%struct.gimplify_ctx**> [#uses=0]
@cl_pf_opts = external global %struct.cl_perfunc_opts		; <%struct.cl_perfunc_opts*> [#uses=0]
@__FUNCTION__.20030 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@__FUNCTION__.20099 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@global_trees = external global [47 x %struct.tree_node*]		; <[47 x %struct.tree_node*]*> [#uses=0]
@tree_code_type = external global [0 x i32]		; <[0 x i32]*> [#uses=2]
@current_function_decl = external global %struct.tree_node*		; <%struct.tree_node**> [#uses=0]
@str1 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@str2 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@__FUNCTION__.20151 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@__FUNCTION__.20221 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@tree_code_length = external global [0 x i8]		; <[0 x i8]*> [#uses=0]
@__FUNCTION__.20435 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@__FUNCTION__.20496 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@cfun = external global %struct.function*		; <%struct.function**> [#uses=0]
@__FUNCTION__.20194 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@__FUNCTION__.19987 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@__FUNCTION__.20532 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@__FUNCTION__.20583 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@__FUNCTION__.20606 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@__FUNCTION__.20644 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@__FUNCTION__.20681 = external global [13 x i8]		; <[13 x i8]*> [#uses=0]
@__FUNCTION__.20700 = external global [13 x i8]		; <[13 x i8]*> [#uses=0]
@__FUNCTION__.21426 = external global [20 x i8]		; <[20 x i8]*> [#uses=0]
@__FUNCTION__.21471 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@__FUNCTION__.21962 = external global [27 x i8]		; <[27 x i8]*> [#uses=0]
@__FUNCTION__.22992 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@__FUNCTION__.23735 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@lang_hooks = external global %struct.lang_hooks		; <%struct.lang_hooks*> [#uses=0]
@__FUNCTION__.27383 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@__FUNCTION__.20776 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@__FUNCTION__.10672 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@str3 = external global [47 x i8]		; <[47 x i8]*> [#uses=0]
@str4 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@__FUNCTION__.20065 = external global [25 x i8]		; <[25 x i8]*> [#uses=0]
@__FUNCTION__.23256 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@__FUNCTION__.23393 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@__FUNCTION__.20043 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@__FUNCTION__.20729 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@__FUNCTION__.20563 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@__FUNCTION__.10663 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@__FUNCTION__.20367 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@__FUNCTION__.20342 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@input_location = external global %struct.location_t		; <%struct.location_t*> [#uses=0]
@__FUNCTION__.24510 = external global [27 x i8]		; <[27 x i8]*> [#uses=0]
@__FUNCTION__.25097 = external global [25 x i8]		; <[25 x i8]*> [#uses=0]
@__FUNCTION__.24705 = external global [26 x i8]		; <[26 x i8]*> [#uses=0]
@str5 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@__FUNCTION__.25136 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@__FUNCTION__.24450 = external global [31 x i8]		; <[31 x i8]*> [#uses=0]
@implicit_built_in_decls = external global [471 x %struct.tree_node*]		; <[471 x %struct.tree_node*]*> [#uses=0]
@__FUNCTION__.24398 = external global [31 x i8]		; <[31 x i8]*> [#uses=0]
@__FUNCTION__.26156 = external global [14 x i8]		; <[14 x i8]*> [#uses=1]
@unknown_location = external global %struct.location_t		; <%struct.location_t*> [#uses=0]
@__FUNCTION__.23038 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@str6 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@__FUNCTION__.25476 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@__FUNCTION__.22136 = external global [20 x i8]		; <[20 x i8]*> [#uses=1]
@__FUNCTION__.21997 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@__FUNCTION__.21247 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@built_in_decls = external global [471 x %struct.tree_node*]		; <[471 x %struct.tree_node*]*> [#uses=0]
@__FUNCTION__.21924 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@__FUNCTION__.21861 = external global [25 x i8]		; <[25 x i8]*> [#uses=0]
@global_dc = external global %struct.diagnostic_context*		; <%struct.diagnostic_context**> [#uses=0]
@__FUNCTION__.25246 = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@str7 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@stderr = external global %struct.FILE*		; <%struct.FILE**> [#uses=0]
@str8 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@str9 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@__FUNCTION__.27653 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@__FUNCTION__.27322 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@__FUNCTION__.27139 = external global [20 x i8]		; <[20 x i8]*> [#uses=0]
@__FUNCTION__.22462 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@str10 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@__FUNCTION__.25389 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@__FUNCTION__.25650 = external global [18 x i8]		; <[18 x i8]*> [#uses=0]
@str11 = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@str12 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@str13 = external global [44 x i8]		; <[44 x i8]*> [#uses=0]
@__FUNCTION__.27444 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@timevar_enable = external global i8		; <i8*> [#uses=0]
@__FUNCTION__.27533 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@flag_instrument_function_entry_exit = external global i32		; <i32*> [#uses=0]
@__FUNCTION__.25331 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@__FUNCTION__.20965 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@str14 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@__FUNCTION__.26053 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@__FUNCTION__.26004 = external global [20 x i8]		; <[20 x i8]*> [#uses=0]
@str15 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@__FUNCTION__.21584 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@str16 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@__FUNCTION__.25903 = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@__FUNCTION__.22930 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@__FUNCTION__.23832 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@str17 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@__FUNCTION__.24620 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@__FUNCTION__.24582 = external global [30 x i8]		; <[30 x i8]*> [#uses=0]
@__FUNCTION__.21382 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@__FUNCTION__.21117 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]


declare void @push_gimplify_context()

declare i32 @gimple_tree_hash(i8*)

declare i32 @iterative_hash_expr(%struct.tree_node*, i32)

declare i32 @gimple_tree_eq(i8*, i8*)

declare i32 @operand_equal_p(%struct.tree_node*, %struct.tree_node*, i32)

declare void @fancy_abort(i8*, i32, i8*)

declare i8* @xcalloc(i32, i32)

declare %struct.htab* @htab_create(i32, i32 (i8*)*, i32 (i8*, i8*)*, void (i8*)*)

declare void @free(i8*)

declare void @gimple_push_bind_expr(%struct.tree_node*)

declare void @gimple_pop_bind_expr()

declare %struct.tree_node* @gimple_current_bind_expr()

declare fastcc void @gimple_push_condition()

declare %struct.tree_node* @create_artificial_label()

declare %struct.tree_node* @build_decl_stat(i32, %struct.tree_node*, %struct.tree_node*)

declare void @tree_class_check_failed(%struct.tree_node*, i32, i8*, i32, i8*)

declare %struct.tree_node* @create_tmp_var_name(i8*)

declare i32 @strlen(i8*)

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare i32 @sprintf(i8*, i8*, ...)

declare %struct.tree_node* @get_identifier(i8*)

declare %struct.tree_node* @create_tmp_var_raw(%struct.tree_node*, i8*)

declare %struct.tree_node* @build_qualified_type(%struct.tree_node*, i32)

declare i8* @get_name(%struct.tree_node*)

declare void @tree_operand_check_failed(i32, i32, i8*, i32, i8*)

declare void @tree_check_failed(%struct.tree_node*, i8*, i32, i8*, ...)

declare void @declare_tmp_vars(%struct.tree_node*, %struct.tree_node*)

declare %struct.tree_node* @nreverse(%struct.tree_node*)

declare void @gimple_add_tmp_var(%struct.tree_node*)

declare void @record_vars(%struct.tree_node*)

declare %struct.tree_node* @create_tmp_var(%struct.tree_node*, i8*)

declare void @pop_gimplify_context(%struct.tree_node*)

declare void @htab_delete(%struct.htab*)

declare fastcc void @annotate_one_with_locus(%struct.tree_node*, i32, i32)

declare void @annotate_with_locus(%struct.tree_node*, i32, i32)

declare %struct.tree_node* @mostly_copy_tree_r(%struct.tree_node**, i32*, i8*)

declare %struct.tree_node* @copy_tree_r(%struct.tree_node**, i32*, i8*)

declare %struct.tree_node* @mark_decls_volatile_r(%struct.tree_node**, i32*, i8*)

declare %struct.tree_node* @copy_if_shared_r(%struct.tree_node**, i32*, i8*)

declare %struct.tree_node* @walk_tree(%struct.tree_node**, %struct.tree_node* (%struct.tree_node**, i32*, i8*)*, i8*, %struct.pointer_set_t*)

declare %struct.tree_node* @unmark_visited_r(%struct.tree_node**, i32*, i8*)

declare fastcc void @unshare_body(%struct.tree_node**, %struct.tree_node*)

declare %struct.cgraph_node* @cgraph_node(%struct.tree_node*)

declare fastcc void @unvisit_body(%struct.tree_node**, %struct.tree_node*)

declare void @unshare_all_trees(%struct.tree_node*)

declare %struct.tree_node* @unshare_expr(%struct.tree_node*)

declare %struct.tree_node* @build_and_jump(%struct.tree_node**)

declare %struct.tree_node* @build1_stat(i32, %struct.tree_node*, %struct.tree_node*)

declare i32 @compare_case_labels(i8*, i8*)

declare i32 @tree_int_cst_compare(%struct.tree_node*, %struct.tree_node*)

declare void @sort_case_labels(%struct.tree_node*)

declare void @tree_vec_elt_check_failed(i32, i32, i8*, i32, i8*)

declare void @qsort(i8*, i32, i32, i32 (i8*, i8*)*)

declare %struct.tree_node* @force_labels_r(%struct.tree_node**, i32*, i8*)

declare fastcc void @canonicalize_component_ref(%struct.tree_node**)

declare %struct.tree_node* @get_unwidened(%struct.tree_node*, %struct.tree_node*)

declare fastcc void @maybe_with_size_expr(%struct.tree_node**)

declare %struct.tree_node* @substitute_placeholder_in_expr(%struct.tree_node*, %struct.tree_node*)

declare %struct.tree_node* @build2_stat(i32, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*)

declare fastcc %struct.tree_node* @gimple_boolify(%struct.tree_node*)

declare %struct.tree_node* @convert(%struct.tree_node*, %struct.tree_node*)

declare %struct.tree_node* @gimplify_init_ctor_preeval_1(%struct.tree_node**, i32*, i8*)

declare i64 @get_alias_set(%struct.tree_node*)

declare i32 @alias_sets_conflict_p(i64, i64)

declare fastcc i8 @cpt_same_type(%struct.tree_node*, %struct.tree_node*) zeroext

declare %struct.tree_node* @check_pointer_types_r(%struct.tree_node**, i32*, i8*)

declare %struct.tree_node* @voidify_wrapper_expr(%struct.tree_node*, %struct.tree_node*)

declare i32 @integer_zerop(%struct.tree_node*)

declare fastcc void @append_to_statement_list_1(%struct.tree_node*, %struct.tree_node**)

declare %struct.tree_node* @alloc_stmt_list()

declare void @tsi_link_after(%struct.tree_stmt_iterator*, %struct.tree_node*, i32)

declare void @append_to_statement_list_force(%struct.tree_node*, %struct.tree_node**)

declare void @append_to_statement_list(%struct.tree_node*, %struct.tree_node**)

declare fastcc %struct.tree_node* @shortcut_cond_r(%struct.tree_node*, %struct.tree_node**, %struct.tree_node**)

declare %struct.tree_node* @build3_stat(i32, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*)

declare fastcc %struct.tree_node* @shortcut_cond_expr(%struct.tree_node*)

declare %struct.tree_node* @expr_last(%struct.tree_node*)

declare i8 @block_may_fallthru(%struct.tree_node*) zeroext 

declare fastcc void @gimple_pop_condition(%struct.tree_node**)

declare %struct.tree_node* @gimple_build_eh_filter(%struct.tree_node*, %struct.tree_node*, %struct.tree_node*)

declare void @annotate_all_with_locus(%struct.tree_node**, i32, i32)

declare fastcc %struct.tree_node* @internal_get_tmp_var(%struct.tree_node*, %struct.tree_node**, %struct.tree_node**, i8 zeroext )

define i32 @gimplify_expr(%struct.tree_node** %expr_p, %struct.tree_node** %pre_p, %struct.tree_node** %post_p, i8 (%struct.tree_node*) zeroext * %gimple_test_f, i32 %fallback) {
entry:
	%internal_post = alloca %struct.tree_node*, align 4		; <%struct.tree_node**> [#uses=2]
	%pre_p_addr.0 = select i1 false, %struct.tree_node** null, %struct.tree_node** %pre_p		; <%struct.tree_node**> [#uses=7]
	%post_p_addr.0 = select i1 false, %struct.tree_node** %internal_post, %struct.tree_node** %post_p		; <%struct.tree_node**> [#uses=7]
	br i1 false, label %bb277, label %bb191

bb191:		; preds = %entry
	ret i32 0

bb277:		; preds = %entry
	%tmp283 = call i32 null( %struct.tree_node** %expr_p, %struct.tree_node** %pre_p_addr.0, %struct.tree_node** %post_p_addr.0 )		; <i32> [#uses=1]
	switch i32 %tmp283, label %bb7478 [
		 i32 0, label %cond_next289
		 i32 -1, label %cond_next298
	]

cond_next289:		; preds = %bb277
	ret i32 0

cond_next298:		; preds = %bb277
	switch i32 0, label %bb7444 [
		 i32 24, label %bb7463
		 i32 25, label %bb7463
		 i32 26, label %bb7463
		 i32 27, label %bb7463
		 i32 28, label %bb7463
		 i32 33, label %bb4503
		 i32 39, label %bb397
		 i32 40, label %bb5650
		 i32 41, label %bb4339
		 i32 42, label %bb4350
		 i32 43, label %bb4350
		 i32 44, label %bb319
		 i32 45, label %bb397
		 i32 46, label %bb6124
		 i32 47, label %bb7463
		 i32 49, label %bb5524
		 i32 50, label %bb1283
		 i32 51, label %bb1289
		 i32 52, label %bb1289
		 i32 53, label %bb5969
		 i32 54, label %bb408
		 i32 56, label %bb5079
		 i32 57, label %bb428
		 i32 59, label %bb5965
		 i32 74, label %bb4275
		 i32 75, label %bb4275
		 i32 76, label %bb4275
		 i32 77, label %bb4275
		 i32 91, label %bb1296
		 i32 92, label %bb1296
		 i32 96, label %bb1322
		 i32 112, label %bb2548
		 i32 113, label %bb2548
		 i32 115, label %bb397
		 i32 116, label %bb5645
		 i32 117, label %bb1504
		 i32 121, label %bb397
		 i32 122, label %bb397
		 i32 123, label %bb313
		 i32 124, label %bb313
		 i32 125, label %bb313
		 i32 126, label %bb313
		 i32 127, label %bb2141
		 i32 128, label %cond_next5873
		 i32 129, label %cond_next5873
		 i32 130, label %bb4536
		 i32 131, label %bb5300
		 i32 132, label %bb5170
		 i32 133, label %bb5519
		 i32 134, label %bb5091
		 i32 135, label %bb5083
		 i32 136, label %bb5087
		 i32 137, label %bb5382
		 i32 139, label %bb7463
		 i32 140, label %bb7463
		 i32 142, label %bb5974
		 i32 143, label %bb6049
		 i32 147, label %bb6296
		 i32 151, label %cond_next6474
	]

bb313:		; preds = %cond_next298, %cond_next298, %cond_next298, %cond_next298
	ret i32 0

bb319:		; preds = %cond_next298
	ret i32 0

bb397:		; preds = %cond_next298, %cond_next298, %cond_next298, %cond_next298, %cond_next298
	ret i32 0

bb408:		; preds = %cond_next298
	%tmp413 = call fastcc i32 @gimplify_cond_expr( %struct.tree_node** %expr_p, %struct.tree_node** %pre_p_addr.0, %struct.tree_node** %post_p_addr.0, %struct.tree_node* null, i32 %fallback )		; <i32> [#uses=0]
	ret i32 0

bb428:		; preds = %cond_next298
	ret i32 0

bb1283:		; preds = %cond_next298
	ret i32 0

bb1289:		; preds = %cond_next298, %cond_next298
	ret i32 0

bb1296:		; preds = %cond_next298, %cond_next298
	ret i32 0

bb1322:		; preds = %cond_next298
	ret i32 0

bb1504:		; preds = %cond_next298
	ret i32 0

bb2141:		; preds = %cond_next298
	ret i32 0

bb2548:		; preds = %cond_next298, %cond_next298
	%tmp2554 = load %struct.tree_node** %expr_p		; <%struct.tree_node*> [#uses=2]
	%tmp2562 = and i32 0, 255		; <i32> [#uses=1]
	%tmp2569 = add i8 0, -4		; <i8> [#uses=1]
	icmp ugt i8 %tmp2569, 5		; <i1>:0 [#uses=2]
	%tmp2587 = load i8* null		; <i8> [#uses=1]
	icmp eq i8 %tmp2587, 0		; <i1>:1 [#uses=2]
	%tmp2607 = load %struct.tree_node** null		; <%struct.tree_node*> [#uses=2]
	br i1 false, label %bb2754, label %cond_next2617

cond_next2617:		; preds = %bb2548
	ret i32 0

bb2754:		; preds = %bb2548
	br i1 %0, label %cond_true2780, label %cond_next2783

cond_true2780:		; preds = %bb2754
	call void @tree_class_check_failed( %struct.tree_node* %tmp2554, i32 9, i8* getelementptr ([42 x i8]* @str, i32 0, i32 0), i32 1415, i8* getelementptr ([20 x i8]* @__FUNCTION__.22136, i32 0, i32 0) )
	unreachable

cond_next2783:		; preds = %bb2754
	%tmp2825 = and i32 0, 255		; <i32> [#uses=1]
	%tmp2829 = load i32* null		; <i32> [#uses=1]
	%tmp28292830 = trunc i32 %tmp2829 to i8		; <i8> [#uses=1]
	%tmp2832 = add i8 %tmp28292830, -4		; <i8> [#uses=1]
	icmp ugt i8 %tmp2832, 5		; <i1>:2 [#uses=1]
	icmp eq i8 0, 0		; <i1>:3 [#uses=1]
	%tmp28652866 = bitcast %struct.tree_node* %tmp2607 to %struct.tree_exp*		; <%struct.tree_exp*> [#uses=1]
	%tmp2868 = getelementptr %struct.tree_exp* %tmp28652866, i32 0, i32 4, i32 0		; <%struct.tree_node**> [#uses=1]
	%tmp2870 = load %struct.tree_node** %tmp2868		; <%struct.tree_node*> [#uses=1]
	br i1 %1, label %cond_true2915, label %cond_next2927

cond_true2915:		; preds = %cond_next2783
	unreachable

cond_next2927:		; preds = %cond_next2783
	%tmp2938 = load %struct.tree_node** null		; <%struct.tree_node*> [#uses=1]
	%tmp2944 = load i32* null		; <i32> [#uses=1]
	%tmp2946 = and i32 %tmp2944, 255		; <i32> [#uses=1]
	%tmp2949 = getelementptr [0 x i32]* @tree_code_type, i32 0, i32 %tmp2946		; <i32*> [#uses=1]
	%tmp2950 = load i32* %tmp2949		; <i32> [#uses=1]
	icmp eq i32 %tmp2950, 2		; <i1>:4 [#uses=1]
	br i1 %4, label %cond_next2954, label %cond_true2951

cond_true2951:		; preds = %cond_next2927
	call void @tree_class_check_failed( %struct.tree_node* %tmp2938, i32 2, i8* getelementptr ([42 x i8]* @str, i32 0, i32 0), i32 1415, i8* getelementptr ([20 x i8]* @__FUNCTION__.22136, i32 0, i32 0) )
	unreachable

cond_next2954:		; preds = %cond_next2927
	br i1 %0, label %cond_true2991, label %cond_next2994

cond_true2991:		; preds = %cond_next2954
	unreachable

cond_next2994:		; preds = %cond_next2954
	br i1 %1, label %cond_true3009, label %cond_next3021

cond_true3009:		; preds = %cond_next2994
	call void @tree_operand_check_failed( i32 0, i32 %tmp2562, i8* getelementptr ([42 x i8]* @str, i32 0, i32 0), i32 1415, i8* getelementptr ([20 x i8]* @__FUNCTION__.22136, i32 0, i32 0) )
	unreachable

cond_next3021:		; preds = %cond_next2994
	br i1 %2, label %cond_true3044, label %cond_next3047

cond_true3044:		; preds = %cond_next3021
	call void @tree_class_check_failed( %struct.tree_node* %tmp2607, i32 9, i8* getelementptr ([42 x i8]* @str, i32 0, i32 0), i32 1415, i8* getelementptr ([20 x i8]* @__FUNCTION__.22136, i32 0, i32 0) )
	unreachable

cond_next3047:		; preds = %cond_next3021
	br i1 %3, label %cond_true3062, label %cond_next3074

cond_true3062:		; preds = %cond_next3047
	call void @tree_operand_check_failed( i32 0, i32 %tmp2825, i8* getelementptr ([42 x i8]* @str, i32 0, i32 0), i32 1415, i8* getelementptr ([20 x i8]* @__FUNCTION__.22136, i32 0, i32 0) )
	unreachable

cond_next3074:		; preds = %cond_next3047
	%tmp3084 = getelementptr %struct.tree_node* %tmp2870, i32 0, i32 0, i32 0, i32 1		; <%struct.tree_node**> [#uses=1]
	%tmp3085 = load %struct.tree_node** %tmp3084		; <%struct.tree_node*> [#uses=1]
	%tmp31043105 = bitcast %struct.tree_node* %tmp3085 to %struct.tree_type*		; <%struct.tree_type*> [#uses=1]
	%tmp3106 = getelementptr %struct.tree_type* %tmp31043105, i32 0, i32 6		; <i16*> [#uses=1]
	%tmp31063107 = bitcast i16* %tmp3106 to i32*		; <i32*> [#uses=1]
	%tmp3108 = load i32* %tmp31063107		; <i32> [#uses=1]
	xor i32 %tmp3108, 0		; <i32>:5 [#uses=1]
	%tmp81008368 = and i32 %5, 65024		; <i32> [#uses=1]
	icmp eq i32 %tmp81008368, 0		; <i1>:6 [#uses=1]
	br i1 %6, label %cond_next3113, label %bb3351

cond_next3113:		; preds = %cond_next3074
	ret i32 0

bb3351:		; preds = %cond_next3074
	%tmp3354 = call i8 @tree_ssa_useless_type_conversion( %struct.tree_node* %tmp2554 ) zeroext 		; <i8> [#uses=1]
	icmp eq i8 %tmp3354, 0		; <i1>:7 [#uses=1]
	%tmp3424 = load i32* null		; <i32> [#uses=1]
	br i1 %7, label %cond_next3417, label %cond_true3356

cond_true3356:		; preds = %bb3351
	ret i32 0

cond_next3417:		; preds = %bb3351
	br i1 false, label %cond_true3429, label %cond_next4266

cond_true3429:		; preds = %cond_next3417
	%tmp3443 = and i32 %tmp3424, 255		; <i32> [#uses=0]
	ret i32 0

cond_next4266:		; preds = %cond_next3417
	%tmp4268 = load %struct.tree_node** %expr_p		; <%struct.tree_node*> [#uses=1]
	icmp eq %struct.tree_node* %tmp4268, null		; <i1>:8 [#uses=1]
	br i1 %8, label %bb4275, label %bb7463

bb4275:		; preds = %cond_next4266, %cond_next298, %cond_next298, %cond_next298, %cond_next298
	%tmp4289 = and i32 0, 255		; <i32> [#uses=2]
	%tmp4292 = getelementptr [0 x i32]* @tree_code_type, i32 0, i32 %tmp4289		; <i32*> [#uses=1]
	%tmp4293 = load i32* %tmp4292		; <i32> [#uses=1]
	%tmp42934294 = trunc i32 %tmp4293 to i8		; <i8> [#uses=1]
	%tmp4296 = add i8 %tmp42934294, -4		; <i8> [#uses=1]
	icmp ugt i8 %tmp4296, 5		; <i1>:9 [#uses=1]
	br i1 %9, label %cond_true4297, label %cond_next4300

cond_true4297:		; preds = %bb4275
	unreachable

cond_next4300:		; preds = %bb4275
	%tmp4314 = load i8* null		; <i8> [#uses=1]
	icmp eq i8 %tmp4314, 0		; <i1>:10 [#uses=1]
	br i1 %10, label %cond_true4315, label %cond_next4327

cond_true4315:		; preds = %cond_next4300
	call void @tree_operand_check_failed( i32 0, i32 %tmp4289, i8* getelementptr ([42 x i8]* @str, i32 0, i32 0), i32 3997, i8* getelementptr ([14 x i8]* @__FUNCTION__.26156, i32 0, i32 0) )
	unreachable

cond_next4327:		; preds = %cond_next4300
	%tmp4336 = call i32 @gimplify_expr( %struct.tree_node** null, %struct.tree_node** %pre_p_addr.0, %struct.tree_node** %post_p_addr.0, i8 (%struct.tree_node*) zeroext * @is_gimple_val, i32 1 )		; <i32> [#uses=0]
	ret i32 0

bb4339:		; preds = %cond_next298
	ret i32 0

bb4350:		; preds = %cond_next298, %cond_next298
	ret i32 0

bb4503:		; preds = %cond_next298
	ret i32 0

bb4536:		; preds = %cond_next298
	ret i32 0

bb5079:		; preds = %cond_next298
	ret i32 0

bb5083:		; preds = %cond_next298
	ret i32 0

bb5087:		; preds = %cond_next298
	ret i32 0

bb5091:		; preds = %cond_next298
	ret i32 0

bb5170:		; preds = %cond_next298
	ret i32 0

bb5300:		; preds = %cond_next298
	ret i32 0

bb5382:		; preds = %cond_next298
	ret i32 0

bb5519:		; preds = %cond_next298
	ret i32 0

bb5524:		; preds = %cond_next298
	ret i32 0

bb5645:		; preds = %cond_next298
	ret i32 0

bb5650:		; preds = %cond_next298
	ret i32 0

cond_next5873:		; preds = %cond_next298, %cond_next298
	ret i32 0

bb5965:		; preds = %cond_next298
	%tmp5968 = call fastcc i32 @gimplify_cleanup_point_expr( %struct.tree_node** %expr_p, %struct.tree_node** %pre_p_addr.0 )		; <i32> [#uses=0]
	ret i32 0

bb5969:		; preds = %cond_next298
	%tmp5973 = call fastcc i32 @gimplify_target_expr( %struct.tree_node** %expr_p, %struct.tree_node** %pre_p_addr.0, %struct.tree_node** %post_p_addr.0 )		; <i32> [#uses=0]
	ret i32 0

bb5974:		; preds = %cond_next298
	ret i32 0

bb6049:		; preds = %cond_next298
	ret i32 0

bb6124:		; preds = %cond_next298
	ret i32 0

bb6296:		; preds = %cond_next298
	ret i32 0

cond_next6474:		; preds = %cond_next298
	icmp eq %struct.tree_node** %internal_post, %post_p_addr.0		; <i1>:11 [#uses=1]
	%iftmp.381.0 = select i1 %11, %struct.tree_node** null, %struct.tree_node** %post_p_addr.0		; <%struct.tree_node**> [#uses=1]
	%tmp6490 = call i32 @gimplify_expr( %struct.tree_node** null, %struct.tree_node** %pre_p_addr.0, %struct.tree_node** %iftmp.381.0, i8 (%struct.tree_node*) zeroext * %gimple_test_f, i32 %fallback )		; <i32> [#uses=0]
	%tmp6551 = call i32 @gimplify_expr( %struct.tree_node** null, %struct.tree_node** %pre_p_addr.0, %struct.tree_node** %post_p_addr.0, i8 (%struct.tree_node*) zeroext * @is_gimple_val, i32 1 )		; <i32> [#uses=0]
	ret i32 0

bb7444:		; preds = %cond_next298
	ret i32 0

bb7463:		; preds = %cond_next4266, %cond_next298, %cond_next298, %cond_next298, %cond_next298, %cond_next298, %cond_next298, %cond_next298, %cond_next298
	ret i32 0

bb7478:		; preds = %bb277
	ret i32 0
}

declare i8 @is_gimple_formal_tmp_rhs(%struct.tree_node*) zeroext 

declare void @gimplify_and_add(%struct.tree_node*, %struct.tree_node**)

declare %struct.tree_node* @get_initialized_tmp_var(%struct.tree_node*, %struct.tree_node**, %struct.tree_node**)

declare %struct.tree_node* @get_formal_tmp_var(%struct.tree_node*, %struct.tree_node**)

declare fastcc void @gimplify_init_ctor_preeval(%struct.tree_node**, %struct.tree_node**, %struct.tree_node**, %struct.gimplify_init_ctor_preeval_data*)

declare i8 @type_contains_placeholder_p(%struct.tree_node*) zeroext 

declare i8 @is_gimple_mem_rhs(%struct.tree_node*) zeroext 

declare fastcc i32 @gimplify_modify_expr_rhs(%struct.tree_node**, %struct.tree_node**, %struct.tree_node**, %struct.tree_node**, %struct.tree_node**, i8 zeroext )

declare %struct.tree_node* @fold_indirect_ref(%struct.tree_node*)

declare fastcc i32 @gimplify_compound_expr(%struct.tree_node**, %struct.tree_node**, i8 zeroext )

declare i8 @is_gimple_lvalue(%struct.tree_node*) zeroext 

declare void @categorize_ctor_elements(%struct.tree_node*, i64*, i64*, i64*, i8*)

declare void @lhd_set_decl_assembler_name(%struct.tree_node*)

declare i64 @int_size_in_bytes(%struct.tree_node*)

declare i32 @can_move_by_pieces(i64, i32)

declare i64 @count_type_elements(%struct.tree_node*)

declare void @gimplify_stmt(%struct.tree_node**)

declare %struct.tree_node* @get_base_address(%struct.tree_node*)

declare fastcc void @gimplify_init_ctor_eval(%struct.tree_node*, %struct.tree_node*, %struct.tree_node**, i8 zeroext )

declare %struct.tree_node* @build_complex(%struct.tree_node*, %struct.tree_node*, %struct.tree_node*)

declare i8 (%struct.tree_node*) zeroext * @rhs_predicate_for(%struct.tree_node*)

declare %struct.tree_node* @build_vector(%struct.tree_node*, %struct.tree_node*)

declare i8 @is_gimple_val(%struct.tree_node*) zeroext 

declare i8 @is_gimple_reg_type(%struct.tree_node*) zeroext 

declare fastcc i32 @gimplify_cond_expr(%struct.tree_node**, %struct.tree_node**, %struct.tree_node**, %struct.tree_node*, i32)

declare fastcc i32 @gimplify_modify_expr(%struct.tree_node**, %struct.tree_node**, %struct.tree_node**, i8 zeroext )

declare %struct.tree_node* @tree_cons_stat(%struct.tree_node*, %struct.tree_node*, %struct.tree_node*)

declare %struct.tree_node* @build_fold_addr_expr(%struct.tree_node*)

declare %struct.tree_node* @build_function_call_expr(%struct.tree_node*, %struct.tree_node*)

declare i8 @is_gimple_addressable(%struct.tree_node*) zeroext 

declare i8 @is_gimple_reg(%struct.tree_node*) zeroext 

declare %struct.tree_node* @make_ssa_name(%struct.tree_node*, %struct.tree_node*)

declare i8 @tree_ssa_useless_type_conversion(%struct.tree_node*) zeroext 

declare fastcc i32 @gimplify_self_mod_expr(%struct.tree_node**, %struct.tree_node**, %struct.tree_node**, i8 zeroext )

declare fastcc i32 @gimplify_compound_lval(%struct.tree_node**, %struct.tree_node**, %struct.tree_node**, i32)

declare %struct.tree_node* @get_callee_fndecl(%struct.tree_node*)

declare %struct.tree_node* @fold_builtin(%struct.tree_node*, i8 zeroext )

declare void @error(i8*, ...)

declare %struct.tree_node* @build_empty_stmt()

declare i8 @fold_builtin_next_arg(%struct.tree_node*) zeroext 

declare fastcc i32 @gimplify_arg(%struct.tree_node**, %struct.tree_node**)

declare i8 @is_gimple_call_addr(%struct.tree_node*) zeroext 

declare i32 @call_expr_flags(%struct.tree_node*)

declare void @recalculate_side_effects(%struct.tree_node*)

declare %struct.tree_node* @fold_convert(%struct.tree_node*, %struct.tree_node*)

declare void @recompute_tree_invarant_for_addr_expr(%struct.tree_node*)

declare i32 @gimplify_va_arg_expr(%struct.tree_node**, %struct.tree_node**, %struct.tree_node**)

declare %struct.tree_node* @size_int_kind(i64, i32)

declare %struct.tree_node* @size_binop(i32, %struct.tree_node*, %struct.tree_node*)

declare %struct.tree_node* @build4_stat(i32, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*)

declare void @gimplify_type_sizes(%struct.tree_node*, %struct.tree_node**)

declare void @gimplify_one_sizepos(%struct.tree_node**, %struct.tree_node**)

declare %struct.tree_node* @build_pointer_type(%struct.tree_node*)

declare %struct.tree_node* @build_fold_indirect_ref(%struct.tree_node*)

declare fastcc i32 @gimplify_bind_expr(%struct.tree_node**, %struct.tree_node*, %struct.tree_node**)

declare fastcc void @gimplify_loop_expr(%struct.tree_node**, %struct.tree_node**)

declare fastcc i32 @gimplify_switch_expr(%struct.tree_node**, %struct.tree_node**)

declare %struct.tree_node* @decl_function_context(%struct.tree_node*)

declare %struct.varray_head_tag* @varray_grow(%struct.varray_head_tag*, i32)

declare fastcc void @gimplify_return_expr(%struct.tree_node*, %struct.tree_node**)

declare fastcc i32 @gimplify_save_expr(%struct.tree_node**, %struct.tree_node**, %struct.tree_node**)

declare fastcc i32 @gimplify_asm_expr(%struct.tree_node**, %struct.tree_node**, %struct.tree_node**)

declare void @gimplify_to_stmt_list(%struct.tree_node**)

declare fastcc i32 @gimplify_cleanup_point_expr(%struct.tree_node**, %struct.tree_node**)

declare fastcc i32 @gimplify_target_expr(%struct.tree_node**, %struct.tree_node**, %struct.tree_node**)

declare void @tsi_delink(%struct.tree_stmt_iterator*)

declare void @tsi_link_before(%struct.tree_stmt_iterator*, %struct.tree_node*, i32)

declare i8 @is_gimple_stmt(%struct.tree_node*) zeroext 

declare void @print_generic_expr(%struct.FILE*, %struct.tree_node*, i32)

declare void @debug_tree(%struct.tree_node*)

declare void @internal_error(i8*, ...)

declare %struct.tree_node* @force_gimple_operand(%struct.tree_node*, %struct.tree_node**, i8 zeroext , %struct.tree_node*)

declare i8 @is_gimple_reg_rhs(%struct.tree_node*) zeroext 

declare void @add_referenced_tmp_var(%struct.tree_node*)

declare i8 @contains_placeholder_p(%struct.tree_node*) zeroext 

declare %struct.varray_head_tag* @varray_init(i32, i32, i8*)

declare i32 @handled_component_p(%struct.tree_node*)

declare void @varray_check_failed(%struct.varray_head_tag*, i32, i8*, i32, i8*)

declare %struct.tree_node* @array_ref_low_bound(%struct.tree_node*)

declare i8 @is_gimple_min_invariant(%struct.tree_node*) zeroext 

declare i8 @is_gimple_formal_tmp_reg(%struct.tree_node*) zeroext 

declare %struct.tree_node* @array_ref_element_size(%struct.tree_node*)

declare %struct.tree_node* @component_ref_field_offset(%struct.tree_node*)

declare i8 @is_gimple_min_lval(%struct.tree_node*) zeroext 

declare void @varray_underflow(%struct.varray_head_tag*, i8*, i32, i8*)

declare i32 @list_length(%struct.tree_node*)

declare i8 @parse_output_constraint(i8**, i32, i32, i32, i8*, i8*, i8*) zeroext 

declare i8* @xstrdup(i8*)

declare %struct.tree_node* @build_string(i32, i8*)

declare i8* @strchr(i8*, i32)

declare %struct.tree_node* @build_tree_list_stat(%struct.tree_node*, %struct.tree_node*)

declare %struct.tree_node* @chainon(%struct.tree_node*, %struct.tree_node*)

declare i8 @parse_input_constraint(i8**, i32, i32, i32, i32, i8**, i8*, i8*) zeroext 

declare i8 @is_gimple_asm_val(%struct.tree_node*) zeroext 

declare void @gimplify_body(%struct.tree_node**, %struct.tree_node*, i8 zeroext )

declare void @timevar_push_1(i32)

declare %struct.tree_node* @gimplify_parameters()

declare %struct.tree_node* @expr_only(%struct.tree_node*)

declare void @timevar_pop_1(i32)

declare void @gimplify_function_tree(%struct.tree_node*)

declare void @allocate_struct_function(%struct.tree_node*)

declare %struct.tree_node* @make_tree_vec_stat(i32)

declare %struct.tree_node* @tsi_split_statement_list_after(%struct.tree_stmt_iterator*)

declare i8 @is_gimple_condexpr(%struct.tree_node*) zeroext 

declare %struct.tree_node* @invert_truthvalue(%struct.tree_node*)

declare i8 @initializer_zerop(%struct.tree_node*) zeroext 

declare i32 @simple_cst_equal(%struct.tree_node*, %struct.tree_node*)

declare i32 @aggregate_value_p(%struct.tree_node*, %struct.tree_node*)

declare i32 @fwrite(i8*, i32, i32, %struct.FILE*)
