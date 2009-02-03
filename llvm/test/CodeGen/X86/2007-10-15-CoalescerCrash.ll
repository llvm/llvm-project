; RUN: llvm-as < %s | llc -mtriple=x86_64-linux-gnu
; PR1729

	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.VEC_edge = type { i32, i32, [1 x %struct.edge_def*] }
	%struct.VEC_tree = type { i32, i32, [1 x %struct.tree_node*] }
	%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
	%struct._obstack_chunk = type { i8*, %struct._obstack_chunk*, [4 x i8] }
	%struct.addr_diff_vec_flags = type <{ i8, i8, i8, i8 }>
	%struct.alloc_pool_def = type { i8*, i64, i64, %struct.alloc_pool_list_def*, i64, i64, i64, %struct.alloc_pool_list_def*, i64, i64 }
	%struct.alloc_pool_list_def = type { %struct.alloc_pool_list_def* }
	%struct.basic_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.tree_node*, %struct.VEC_edge*, %struct.VEC_edge*, %struct.bitmap_head_def*, %struct.bitmap_head_def*, i8*, %struct.loop*, [2 x %struct.et_node*], %struct.basic_block_def*, %struct.basic_block_def*, %struct.reorder_block_def*, %struct.bb_ann_d*, i64, i32, i32, i32, i32 }
	%struct.bb_ann_d = type opaque
	%struct.bitmap_element_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, [2 x i64] }
	%struct.bitmap_head_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, %struct.bitmap_obstack* }
	%struct.bitmap_obstack = type { %struct.bitmap_element_def*, %struct.bitmap_head_def*, %struct.obstack }
	%struct.cselib_val_struct = type opaque
	%struct.dataflow_d = type opaque
	%struct.die_struct = type opaque
	%struct.edge_def = type { %struct.basic_block_def*, %struct.basic_block_def*, %struct.edge_def_insns, i8*, %struct.location_t*, i32, i32, i64, i32 }
	%struct.edge_def_insns = type { %struct.rtx_def* }
	%struct.edge_iterator = type { i32, %struct.VEC_edge** }
	%struct.eh_status = type opaque
	%struct.elt_list = type opaque
	%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, i32, %struct.location_t, i32, i8*, %struct.rtx_def** }
	%struct.et_node = type opaque
	%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i8, i32, i64, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, i32, %struct.var_refs_queue*, i32, i32, %struct.rtvec_def*, %struct.tree_node*, i32, i32, i32, %struct.machine_function*, i32, i32, i8, i8, %struct.language_function*, %struct.rtx_def*, i32, i32, i32, i32, %struct.location_t, %struct.varray_head_tag*, %struct.tree_node*, %struct.tree_node*, i8, i8, i8 }
	%struct.ht_identifier = type { i8*, i32, i32 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.lang_type = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { i8*, i32 }
	%struct.loop = type opaque
	%struct.machine_function = type { %struct.stack_local_entry*, i8*, %struct.rtx_def*, i32, i32, i32, i32, i32 }
	%struct.mem_attrs = type { i64, %struct.tree_node*, %struct.rtx_def*, %struct.rtx_def*, i32 }
	%struct.obstack = type { i64, %struct._obstack_chunk*, i8*, i8*, i8*, i64, i32, %struct._obstack_chunk* (i8*, i64)*, void (i8*, %struct._obstack_chunk*)*, i8*, i8 }
	%struct.phi_arg_d = type { %struct.tree_node*, i8 }
	%struct.ptr_info_def = type opaque
	%struct.real_value = type opaque
	%struct.reg_attrs = type { %struct.tree_node*, i64 }
	%struct.reg_info_def = type { i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.reorder_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.basic_block_def*, %struct.basic_block_def*, %struct.basic_block_def*, i32, i32, i32 }
	%struct.rtunion = type { i8* }
	%struct.rtvec_def = type { i32, [1 x %struct.rtx_def*] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.simple_bitmap_def = type { i32, i32, i32, [1 x i64] }
	%struct.stack_local_entry = type opaque
	%struct.temp_slot = type opaque
	%struct.tree_binfo = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.VEC_tree*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.VEC_tree }
	%struct.tree_block = type { %struct.tree_common, i32, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %union.tree_ann_d*, i8, i8, i8, i8, i8 }
	%struct.tree_complex = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, %struct.tree_node*, i8, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, i32, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u1_a = type <{ i32 }>
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
	%struct.tree_string = type { %struct.tree_common, i32, [1 x i8] }
	%struct.tree_type = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i32, i16, i8, i8, i32, %struct.tree_node*, %struct.tree_node*, %struct.rtunion, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_type* }
	%struct.tree_type_symtab = type { i8* }
	%struct.tree_value_handle = type { %struct.tree_common, %struct.value_set*, i32 }
	%struct.tree_vec = type { %struct.tree_common, i32, [1 x %struct.tree_node*] }
	%struct.tree_vector = type { %struct.tree_common, %struct.tree_node* }
	%struct.u = type { [1 x %struct.rtunion] }
	%struct.value_set = type opaque
	%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_data = type { [1 x i64] }
	%struct.varray_head_tag = type { i64, i64, i32, i8*, %struct.varray_data }
	%union.tree_ann_d = type opaque
@first_edge_aux_obj = external global i8*		; <i8**> [#uses=0]
@first_block_aux_obj = external global i8*		; <i8**> [#uses=0]
@n_edges = external global i32		; <i32*> [#uses=0]
@ENTRY_BLOCK_PTR = external global %struct.basic_block_def*		; <%struct.basic_block_def**> [#uses=0]
@EXIT_BLOCK_PTR = external global %struct.basic_block_def*		; <%struct.basic_block_def**> [#uses=0]
@n_basic_blocks = external global i32		; <i32*> [#uses=0]
@.str = external constant [9 x i8]		; <[9 x i8]*> [#uses=0]
@rbi_pool = external global %struct.alloc_pool_def*		; <%struct.alloc_pool_def**> [#uses=0]
@__FUNCTION__.19643 = external constant [18 x i8]		; <[18 x i8]*> [#uses=0]
@.str1 = external constant [20 x i8]		; <[20 x i8]*> [#uses=0]
@__FUNCTION__.19670 = external constant [15 x i8]		; <[15 x i8]*> [#uses=0]
@basic_block_info = external global %struct.varray_head_tag*		; <%struct.varray_head_tag**> [#uses=0]
@last_basic_block = external global i32		; <i32*> [#uses=0]
@__FUNCTION__.19696 = external constant [14 x i8]		; <[14 x i8]*> [#uses=0]
@__FUNCTION__.20191 = external constant [20 x i8]		; <[20 x i8]*> [#uses=0]
@block_aux_obstack = external global %struct.obstack		; <%struct.obstack*> [#uses=0]
@__FUNCTION__.20301 = external constant [20 x i8]		; <[20 x i8]*> [#uses=0]
@__FUNCTION__.20316 = external constant [19 x i8]		; <[19 x i8]*> [#uses=0]
@edge_aux_obstack = external global %struct.obstack		; <%struct.obstack*> [#uses=0]
@stderr = external global %struct._IO_FILE*		; <%struct._IO_FILE**> [#uses=0]
@__FUNCTION__.20463 = external constant [11 x i8]		; <[11 x i8]*> [#uses=0]
@.str2 = external constant [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str3 = external constant [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str4 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str5 = external constant [11 x i8]		; <[11 x i8]*> [#uses=0]
@.str6 = external constant [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str7 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@bitnames.20157 = external constant [13 x i8*]		; <[13 x i8*]*> [#uses=0]
@.str8 = external constant [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str9 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str10 = external constant [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str11 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str12 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str13 = external constant [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str14 = external constant [13 x i8]		; <[13 x i8]*> [#uses=0]
@.str15 = external constant [12 x i8]		; <[12 x i8]*> [#uses=0]
@.str16 = external constant [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str17 = external constant [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str18 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str19 = external constant [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str20 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str21 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str22 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@__FUNCTION__.19709 = external constant [20 x i8]		; <[20 x i8]*> [#uses=0]
@.str23 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str24 = external constant [10 x i8]		; <[10 x i8]*> [#uses=0]
@__FUNCTION__.19813 = external constant [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str25 = external constant [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str26 = external constant [6 x i8]		; <[6 x i8]*> [#uses=0]
@initialized.20241.b = external global i1		; <i1*> [#uses=0]
@__FUNCTION__.20244 = external constant [21 x i8]		; <[21 x i8]*> [#uses=0]
@__FUNCTION__.19601 = external constant [12 x i8]		; <[12 x i8]*> [#uses=0]
@__FUNCTION__.14571 = external constant [8 x i8]		; <[8 x i8]*> [#uses=0]
@__FUNCTION__.14535 = external constant [13 x i8]		; <[13 x i8]*> [#uses=0]
@.str27 = external constant [28 x i8]		; <[28 x i8]*> [#uses=0]
@__FUNCTION__.14589 = external constant [8 x i8]		; <[8 x i8]*> [#uses=0]
@__FUNCTION__.19792 = external constant [12 x i8]		; <[12 x i8]*> [#uses=0]
@__FUNCTION__.19851 = external constant [19 x i8]		; <[19 x i8]*> [#uses=0]
@profile_status = external global i32		; <i32*> [#uses=0]
@.str29 = external constant [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str30 = external constant [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str31 = external constant [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str32 = external constant [49 x i8]		; <[49 x i8]*> [#uses=1]
@__FUNCTION__.19948 = external constant [15 x i8]		; <[15 x i8]*> [#uses=0]
@reg_n_info = external global %struct.varray_head_tag*		; <%struct.varray_head_tag**> [#uses=0]
@reload_completed = external global i32		; <i32*> [#uses=0]
@.str33 = external constant [15 x i8]		; <[15 x i8]*> [#uses=0]
@.str34 = external constant [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str35 = external constant [13 x i8]		; <[13 x i8]*> [#uses=0]
@.str36 = external constant [1 x i8]		; <[1 x i8]*> [#uses=0]
@.str37 = external constant [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str38 = external constant [16 x i8]		; <[16 x i8]*> [#uses=0]
@cfun = external global %struct.function*		; <%struct.function**> [#uses=0]
@.str39 = external constant [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str40 = external constant [11 x i8]		; <[11 x i8]*> [#uses=0]
@.str41 = external constant [20 x i8]		; <[20 x i8]*> [#uses=0]
@.str42 = external constant [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str43 = external constant [19 x i8]		; <[19 x i8]*> [#uses=0]
@mode_size = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@target_flags = external global i32		; <i32*> [#uses=0]
@.str44 = external constant [11 x i8]		; <[11 x i8]*> [#uses=0]
@reg_class_names = external global [0 x i8*]		; <[0 x i8*]*> [#uses=0]
@.str45 = external constant [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str46 = external constant [13 x i8]		; <[13 x i8]*> [#uses=0]
@.str47 = external constant [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str48 = external constant [12 x i8]		; <[12 x i8]*> [#uses=0]
@.str49 = external constant [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str50 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str51 = external constant [29 x i8]		; <[29 x i8]*> [#uses=0]
@.str52 = external constant [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str53 = external constant [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str54 = external constant [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str55 = external constant [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str56 = external constant [12 x i8]		; <[12 x i8]*> [#uses=0]
@.str57 = external constant [26 x i8]		; <[26 x i8]*> [#uses=0]
@.str58 = external constant [15 x i8]		; <[15 x i8]*> [#uses=0]
@.str59 = external constant [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str60 = external constant [26 x i8]		; <[26 x i8]*> [#uses=0]
@.str61 = external constant [24 x i8]		; <[24 x i8]*> [#uses=0]
@initialized.20366.b = external global i1		; <i1*> [#uses=0]
@__FUNCTION__.20369 = external constant [20 x i8]		; <[20 x i8]*> [#uses=0]
@__FUNCTION__.20442 = external constant [19 x i8]		; <[19 x i8]*> [#uses=0]
@bb_bitnames.20476 = external constant [6 x i8*]		; <[6 x i8*]*> [#uses=0]
@.str62 = external constant [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str63 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str64 = external constant [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str65 = external constant [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str66 = external constant [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str67 = external constant [11 x i8]		; <[11 x i8]*> [#uses=0]
@.str68 = external constant [15 x i8]		; <[15 x i8]*> [#uses=0]
@.str69 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str70 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@__FUNCTION__.20520 = external constant [32 x i8]		; <[32 x i8]*> [#uses=0]
@dump_file = external global %struct._IO_FILE*		; <%struct._IO_FILE**> [#uses=0]
@.str71 = external constant [86 x i8]		; <[86 x i8]*> [#uses=0]
@.str72 = external constant [94 x i8]		; <[94 x i8]*> [#uses=0]
@reg_obstack = external global %struct.bitmap_obstack		; <%struct.bitmap_obstack*> [#uses=0]

declare void @init_flow()

declare i8* @ggc_alloc_cleared_stat(i64)

declare fastcc void @free_edge(%struct.edge_def*)

declare void @ggc_free(i8*)

declare %struct.basic_block_def* @alloc_block()

declare void @alloc_rbi_pool()

declare %struct.alloc_pool_def* @create_alloc_pool(i8*, i64, i64)

declare void @free_rbi_pool()

declare void @free_alloc_pool(%struct.alloc_pool_def*)

declare void @initialize_bb_rbi(%struct.basic_block_def*)

declare void @fancy_abort(i8*, i32, i8*)

declare i8* @pool_alloc(%struct.alloc_pool_def*)

declare void @llvm.memset.i64(i8*, i8, i64, i32)

declare void @link_block(%struct.basic_block_def*, %struct.basic_block_def*)

declare void @unlink_block(%struct.basic_block_def*)

declare void @compact_blocks()

declare void @varray_check_failed(%struct.varray_head_tag*, i64, i8*, i32, i8*)

declare void @expunge_block(%struct.basic_block_def*)

declare void @clear_bb_flags()

declare void @alloc_aux_for_block(%struct.basic_block_def*, i32)

declare void @_obstack_newchunk(%struct.obstack*, i32)

declare void @clear_aux_for_blocks()

declare void @free_aux_for_blocks()

declare void @obstack_free(%struct.obstack*, i8*)

declare void @alloc_aux_for_edge(%struct.edge_def*, i32)

declare void @debug_bb(%struct.basic_block_def*)

declare void @dump_bb(%struct.basic_block_def*, %struct._IO_FILE*, i32)

declare %struct.basic_block_def* @debug_bb_n(i32)

declare void @dump_edge_info(%struct._IO_FILE*, %struct.edge_def*, i32)

declare i32 @fputs_unlocked(i8* noalias , %struct._IO_FILE* noalias )

declare i32 @fprintf(%struct._IO_FILE* noalias , i8* noalias , ...)

declare i64 @fwrite(i8*, i64, i64, i8*)

declare i32 @__overflow(%struct._IO_FILE*, i32)

declare %struct.edge_def* @unchecked_make_edge(%struct.basic_block_def*, %struct.basic_block_def*, i32)

declare i8* @vec_gc_p_reserve(i8*, i32)

declare void @vec_assert_fail(i8*, i8*, i8*, i32, i8*)

declare void @execute_on_growing_pred(%struct.edge_def*)

declare %struct.edge_def* @make_edge(%struct.basic_block_def*, %struct.basic_block_def*, i32)

declare %struct.edge_def* @find_edge(%struct.basic_block_def*, %struct.basic_block_def*)

declare %struct.edge_def* @make_single_succ_edge(%struct.basic_block_def*, %struct.basic_block_def*, i32)

declare %struct.edge_def* @cached_make_edge(%struct.simple_bitmap_def**, %struct.basic_block_def*, %struct.basic_block_def*, i32)

declare void @redirect_edge_succ(%struct.edge_def*, %struct.basic_block_def*)

declare void @execute_on_shrinking_pred(%struct.edge_def*)

declare void @alloc_aux_for_blocks(i32)

declare i8* @xmalloc(i64)

declare i32 @_obstack_begin(%struct.obstack*, i32, i32, i8* (i64)*, void (i8*)*)

declare void @free(i8*)

declare void @clear_edges()

declare void @remove_edge(%struct.edge_def*)

declare %struct.edge_def* @redirect_edge_succ_nodup(%struct.edge_def*, %struct.basic_block_def*)

declare void @redirect_edge_pred(%struct.edge_def*, %struct.basic_block_def*)

define void @check_bb_profile(%struct.basic_block_def* %bb, %struct._IO_FILE* %file) {
entry:
	br i1 false, label %cond_false759.preheader, label %cond_false149.preheader

cond_false149.preheader:		; preds = %entry
	ret void

cond_false759.preheader:		; preds = %entry
	br i1 false, label %cond_next873, label %cond_true794

bb644:		; preds = %cond_next873
	ret void

cond_true794:		; preds = %cond_false759.preheader
	ret void

cond_next873:		; preds = %cond_false759.preheader
	br i1 false, label %bb882, label %bb644

bb882:		; preds = %cond_next873
	br i1 false, label %cond_true893, label %cond_next901

cond_true893:		; preds = %bb882
	br label %cond_false1036

cond_next901:		; preds = %bb882
	ret void

bb929:		; preds = %cond_next1150
	%tmp934 = add i64 0, %lsum.11225.0		; <i64> [#uses=1]
	br i1 false, label %cond_next979, label %cond_true974

cond_true974:		; preds = %bb929
	ret void

cond_next979:		; preds = %bb929
	br label %cond_false1036

cond_false1036:		; preds = %cond_next979, %cond_true893
	%lsum.11225.0 = phi i64 [ 0, %cond_true893 ], [ %tmp934, %cond_next979 ]		; <i64> [#uses=2]
	br i1 false, label %cond_next1056, label %cond_true1051

cond_true1051:		; preds = %cond_false1036
	ret void

cond_next1056:		; preds = %cond_false1036
	br i1 false, label %cond_next1150, label %cond_true1071

cond_true1071:		; preds = %cond_next1056
	ret void

cond_next1150:		; preds = %cond_next1056
	%tmp1156 = icmp eq %struct.edge_def* null, null		; <i1> [#uses=1]
	br i1 %tmp1156, label %bb1159, label %bb929

bb1159:		; preds = %cond_next1150
	br i1 false, label %cond_true1169, label %UnifiedReturnBlock

cond_true1169:		; preds = %bb1159
	%tmp11741175 = trunc i64 %lsum.11225.0 to i32		; <i32> [#uses=1]
	%tmp1178 = tail call i32 (%struct._IO_FILE* noalias , i8* noalias , ...)* @fprintf( %struct._IO_FILE* %file noalias , i8* getelementptr ([49 x i8]* @.str32, i32 0, i64 0) noalias , i32 %tmp11741175, i32 0 )		; <i32> [#uses=0]
	ret void

UnifiedReturnBlock:		; preds = %bb1159
	ret void
}

declare void @dump_flow_info(%struct._IO_FILE*)

declare i32 @max_reg_num()

declare void @rtl_check_failed_flag(i8*, %struct.rtx_def*, i8*, i32, i8*)

declare i32 @reg_preferred_class(i32)

declare i32 @reg_alternate_class(i32)

declare i8 @maybe_hot_bb_p(%struct.basic_block_def*) zeroext 

declare i8 @probably_never_executed_bb_p(%struct.basic_block_def*) zeroext 

declare void @dump_regset(%struct.bitmap_head_def*, %struct._IO_FILE*)

declare void @debug_flow_info()

declare void @alloc_aux_for_edges(i32)

declare void @clear_aux_for_edges()

declare void @free_aux_for_edges()

declare void @brief_dump_cfg(%struct._IO_FILE*)

declare i32 @fputc(i32, i8*)

declare void @update_bb_profile_for_threading(%struct.basic_block_def*, i32, i64, %struct.edge_def*)
