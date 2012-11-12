; RUN: llc < %s > %t
; PR6283

; Tricky coalescer bug:
; After coalescing %RAX with a virtual register, this instruction was rematted:
;
;   %EAX<def> = MOV32rr %reg1070<kill>
;
; This instruction silently defined %RAX, and when rematting removed the
; instruction, the live interval for %RAX was not properly updated. The valno
; referred to a deleted instruction and bad things happened.
;
; The fix is to implicitly define %RAX when coalescing:
;
;   %EAX<def> = MOV32rr %reg1070<kill>, %RAX<imp-def>
;

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

module asm "\09.ident\09\22GCC: (GNU) 4.5.0 20100212 (experimental) LLVM: 95975\22"

%0 = type { %"union gimple_statement_d"* }
%"BITMAP_WORD[]" = type [2 x i64]
%"uchar[]" = type [1 x i8]
%"char[]" = type [4 x i8]
%"enum dom_state[]" = type [2 x i32]
%"int[]" = type [4 x i32]
%"struct VEC_basic_block_base" = type { i32, i32, [1 x %"struct basic_block_def"*] }
%"struct VEC_basic_block_gc" = type { %"struct VEC_basic_block_base" }
%"struct VEC_edge_base" = type { i32, i32, [1 x %"struct edge_def"*] }
%"struct VEC_edge_gc" = type { %"struct VEC_edge_base" }
%"struct VEC_gimple_base" = type { i32, i32, [1 x %"union gimple_statement_d"*] }
%"struct VEC_gimple_gc" = type { %"struct VEC_gimple_base" }
%"struct VEC_iv_cand_p_base" = type { i32, i32, [1 x %"struct iv_cand"*] }
%"struct VEC_iv_cand_p_heap" = type { %"struct VEC_iv_cand_p_base" }
%"struct VEC_iv_use_p_base" = type { i32, i32, [1 x %"struct iv_use"*] }
%"struct VEC_iv_use_p_heap" = type { %"struct VEC_iv_use_p_base" }
%"struct VEC_loop_p_base" = type { i32, i32, [1 x %"struct loop"*] }
%"struct VEC_loop_p_gc" = type { %"struct VEC_loop_p_base" }
%"struct VEC_rtx_base" = type { i32, i32, [1 x %"struct rtx_def"*] }
%"struct VEC_rtx_gc" = type { %"struct VEC_rtx_base" }
%"struct VEC_tree_base" = type { i32, i32, [1 x %"union tree_node"*] }
%"struct VEC_tree_gc" = type { %"struct VEC_tree_base" }
%"struct _obstack_chunk" = type { i8*, %"struct _obstack_chunk"*, %"char[]" }
%"struct basic_block_def" = type { %"struct VEC_edge_gc"*, %"struct VEC_edge_gc"*, i8*, %"struct loop"*, [2 x %"struct et_node"*], %"struct basic_block_def"*, %"struct basic_block_def"*, %"union basic_block_il_dependent", i64, i32, i32, i32, i32, i32 }
%"struct bitmap_element" = type { %"struct bitmap_element"*, %"struct bitmap_element"*, i32, %"BITMAP_WORD[]" }
%"struct bitmap_head_def" = type { %"struct bitmap_element"*, %"struct bitmap_element"*, i32, %"struct bitmap_obstack"* }
%"struct bitmap_obstack" = type { %"struct bitmap_element"*, %"struct bitmap_head_def"*, %"struct obstack" }
%"struct block_symbol" = type { [3 x %"union rtunion"], %"struct object_block"*, i64 }
%"struct comp_cost" = type { i32, i32 }
%"struct control_flow_graph" = type { %"struct basic_block_def"*, %"struct basic_block_def"*, %"struct VEC_basic_block_gc"*, i32, i32, i32, %"struct VEC_basic_block_gc"*, i32, %"enum dom_state[]", %"enum dom_state[]", i32, i32 }
%"struct cost_pair" = type { %"struct iv_cand"*, %"struct comp_cost", %"struct bitmap_head_def"*, %"union tree_node"* }
%"struct def_optype_d" = type { %"struct def_optype_d"*, %"union tree_node"** }
%"struct double_int" = type { i64, i64 }
%"struct edge_def" = type { %"struct basic_block_def"*, %"struct basic_block_def"*, %"union edge_def_insns", i8*, %"union tree_node"*, i32, i32, i32, i32, i64 }
%"struct eh_status" = type opaque
%"struct et_node" = type opaque
%"struct function" = type { %"struct eh_status"*, %"struct control_flow_graph"*, %"struct gimple_seq_d"*, %"struct gimple_df"*, %"struct loops"*, %"struct htab"*, %"union tree_node"*, %"union tree_node"*, %"union tree_node"*, %"union tree_node"*, %"struct machine_function"*, %"struct language_function"*, %"struct htab"*, i32, i32, i32, i32, i32, i32, i8*, i8, i8, i8, i8 }
%"struct gimple_bb_info" = type { %"struct gimple_seq_d"*, %"struct gimple_seq_d"* }
%"struct gimple_df" = type { %"struct htab"*, %"struct VEC_gimple_gc"*, %"struct VEC_tree_gc"*, %"union tree_node"*, %"struct pt_solution", %"struct pt_solution", %"struct pointer_map_t"*, %"union tree_node"*, %"struct htab"*, %"struct bitmap_head_def"*, i8, %"struct ssa_operands" }
%"struct gimple_seq_d" = type { %"struct gimple_seq_node_d"*, %"struct gimple_seq_node_d"*, %"struct gimple_seq_d"* }
%"struct gimple_seq_node_d" = type { %"union gimple_statement_d"*, %"struct gimple_seq_node_d"*, %"struct gimple_seq_node_d"* }
%"struct gimple_statement_base" = type { i8, i8, i16, i32, i32, i32, %"struct basic_block_def"*, %"union tree_node"* }
%"struct phi_arg_d[]" = type [1 x %"struct phi_arg_d"]
%"struct gimple_statement_phi" = type { %"struct gimple_statement_base", i32, i32, %"union tree_node"*, %"struct phi_arg_d[]" }
%"struct htab" = type { i32 (i8*)*, i32 (i8*, i8*)*, void (i8*)*, i8**, i64, i64, i64, i32, i32, i8* (i64, i64)*, void (i8*)*, i8*, i8* (i8*, i64, i64)*, void (i8*, i8*)*, i32 }
%"struct iv" = type { %"union tree_node"*, %"union tree_node"*, %"union tree_node"*, %"union tree_node"*, i8, i8, i32 }
%"struct iv_cand" = type { i32, i8, i32, %"union gimple_statement_d"*, %"union tree_node"*, %"union tree_node"*, %"struct iv"*, i32, i32, %"struct iv_use"*, %"struct bitmap_head_def"* }
%"struct iv_use" = type { i32, i32, %"struct iv"*, %"union gimple_statement_d"*, %"union tree_node"**, %"struct bitmap_head_def"*, i32, %"struct cost_pair"*, %"struct iv_cand"* }
%"struct ivopts_data" = type { %"struct loop"*, %"struct pointer_map_t"*, i32, i32, %"struct version_info"*, %"struct bitmap_head_def"*, %"struct VEC_iv_use_p_heap"*, %"struct VEC_iv_cand_p_heap"*, %"struct bitmap_head_def"*, i32, i8, i8 }
%"struct lang_decl" = type opaque
%"struct language_function" = type opaque
%"struct loop" = type { i32, i32, %"struct basic_block_def"*, %"struct basic_block_def"*, %"struct comp_cost", i32, i32, %"struct VEC_loop_p_gc"*, %"struct loop"*, %"struct loop"*, i8*, %"union tree_node"*, %"struct double_int", %"struct double_int", i8, i8, i32, %"struct nb_iter_bound"*, %"struct loop_exit"*, i8, %"union tree_node"* }
%"struct loop_exit" = type { %"struct edge_def"*, %"struct loop_exit"*, %"struct loop_exit"*, %"struct loop_exit"* }
%"struct loops" = type { i32, %"struct VEC_loop_p_gc"*, %"struct htab"*, %"struct loop"* }
%"struct machine_cfa_state" = type { %"struct rtx_def"*, i64 }
%"struct machine_function" = type { %"struct stack_local_entry"*, i8*, i32, i32, %"int[]", i32, %"struct machine_cfa_state", i32, i8 }
%"struct nb_iter_bound" = type { %"union gimple_statement_d"*, %"struct double_int", i8, %"struct nb_iter_bound"* }
%"struct object_block" = type { %"union section"*, i32, i64, %"struct VEC_rtx_gc"*, %"struct VEC_rtx_gc"* }
%"struct obstack" = type { i64, %"struct _obstack_chunk"*, i8*, i8*, i8*, i64, i32, %"struct _obstack_chunk"* (i8*, i64)*, void (i8*, %"struct _obstack_chunk"*)*, i8*, i8 }
%"struct phi_arg_d" = type { %"struct ssa_use_operand_d", %"union tree_node"*, i32 }
%"struct pointer_map_t" = type opaque
%"struct pt_solution" = type { i8, %"struct bitmap_head_def"* }
%"struct rtx_def" = type { i16, i8, i8, %"union u" }
%"struct section_common" = type { i32 }
%"struct ssa_operand_memory_d" = type { %"struct ssa_operand_memory_d"*, %"uchar[]" }
%"struct ssa_operands" = type { %"struct ssa_operand_memory_d"*, i32, i32, i8, %"struct def_optype_d"*, %"struct use_optype_d"* }
%"struct ssa_use_operand_d" = type { %"struct ssa_use_operand_d"*, %"struct ssa_use_operand_d"*, %0, %"union tree_node"** }
%"struct stack_local_entry" = type opaque
%"struct tree_base" = type <{ i16, i8, i8, i8, [2 x i8], i8 }>
%"struct tree_common" = type { %"struct tree_base", %"union tree_node"*, %"union tree_node"* }
%"struct tree_decl_common" = type { %"struct tree_decl_minimal", %"union tree_node"*, i8, i8, i8, i8, i8, i32, %"union tree_node"*, %"union tree_node"*, %"union tree_node"*, %"union tree_node"*, %"struct lang_decl"* }
%"struct tree_decl_minimal" = type { %"struct tree_common", i32, i32, %"union tree_node"*, %"union tree_node"* }
%"struct tree_decl_non_common" = type { %"struct tree_decl_with_vis", %"union tree_node"*, %"union tree_node"*, %"union tree_node"*, %"union tree_node"* }
%"struct tree_decl_with_rtl" = type { %"struct tree_decl_common", %"struct rtx_def"* }
%"struct tree_decl_with_vis" = type { %"struct tree_decl_with_rtl", %"union tree_node"*, %"union tree_node"*, %"union tree_node"*, i8, i8, i8 }
%"struct tree_function_decl" = type { %"struct tree_decl_non_common", %"struct function"*, %"union tree_node"*, %"union tree_node"*, %"union tree_node"*, i16, i8, i8 }
%"struct unnamed_section" = type { %"struct section_common", void (i8*)*, i8*, %"union section"* }
%"struct use_optype_d" = type { %"struct use_optype_d"*, %"struct ssa_use_operand_d" }
%"struct version_info" = type { %"union tree_node"*, %"struct iv"*, i8, i32, i8 }
%"union basic_block_il_dependent" = type { %"struct gimple_bb_info"* }
%"union edge_def_insns" = type { %"struct gimple_seq_d"* }
%"union gimple_statement_d" = type { %"struct gimple_statement_phi" }
%"union rtunion" = type { i8* }
%"union section" = type { %"struct unnamed_section" }
%"union tree_node" = type { %"struct tree_function_decl" }
%"union u" = type { %"struct block_symbol" }

declare fastcc %"union tree_node"* @get_computation_at(%"struct loop"*, %"struct iv_use"* nocapture, %"struct iv_cand"* nocapture, %"union gimple_statement_d"*) nounwind

declare fastcc i32 @computation_cost(%"union tree_node"*, i8 zeroext) nounwind

define fastcc i64 @get_computation_cost_at(%"struct ivopts_data"* %data, %"struct iv_use"* nocapture %use, %"struct iv_cand"* nocapture %cand, i8 zeroext %address_p, %"struct bitmap_head_def"** %depends_on, %"union gimple_statement_d"* %at, i8* %can_autoinc) nounwind {
entry:
  br i1 undef, label %"100", label %"4"

"4":                                              ; preds = %entry
  br i1 undef, label %"6", label %"5"

"5":                                              ; preds = %"4"
  unreachable

"6":                                              ; preds = %"4"
  br i1 undef, label %"8", label %"7"

"7":                                              ; preds = %"6"
  unreachable

"8":                                              ; preds = %"6"
  br i1 undef, label %"100", label %"10"

"10":                                             ; preds = %"8"
  br i1 undef, label %"17", label %"16"

"16":                                             ; preds = %"10"
  unreachable

"17":                                             ; preds = %"10"
  br i1 undef, label %"19", label %"18"

"18":                                             ; preds = %"17"
  unreachable

"19":                                             ; preds = %"17"
  br i1 undef, label %"93", label %"20"

"20":                                             ; preds = %"19"
  br i1 undef, label %"23", label %"21"

"21":                                             ; preds = %"20"
  unreachable

"23":                                             ; preds = %"20"
  br i1 undef, label %"100", label %"25"

"25":                                             ; preds = %"23"
  br i1 undef, label %"100", label %"26"

"26":                                             ; preds = %"25"
  br i1 undef, label %"30", label %"28"

"28":                                             ; preds = %"26"
  unreachable

"30":                                             ; preds = %"26"
  br i1 undef, label %"59", label %"51"

"51":                                             ; preds = %"30"
  br i1 undef, label %"55", label %"52"

"52":                                             ; preds = %"51"
  unreachable

"55":                                             ; preds = %"51"
  %0 = icmp ugt i32 0, undef                      ; <i1> [#uses=1]
  br i1 %0, label %"50.i", label %"9.i"

"9.i":                                            ; preds = %"55"
  unreachable

"50.i":                                           ; preds = %"55"
  br i1 undef, label %"55.i", label %"54.i"

"54.i":                                           ; preds = %"50.i"
  br i1 undef, label %"57.i", label %"55.i"

"55.i":                                           ; preds = %"54.i", %"50.i"
  unreachable

"57.i":                                           ; preds = %"54.i"
  br label %"63.i"

"61.i":                                           ; preds = %"63.i"
  br i1 undef, label %"64.i", label %"62.i"

"62.i":                                           ; preds = %"61.i"
  br label %"63.i"

"63.i":                                           ; preds = %"62.i", %"57.i"
  br i1 undef, label %"61.i", label %"64.i"

"64.i":                                           ; preds = %"63.i", %"61.i"
  unreachable

"59":                                             ; preds = %"30"
  br i1 undef, label %"60", label %"82"

"60":                                             ; preds = %"59"
  br i1 undef, label %"61", label %"82"

"61":                                             ; preds = %"60"
  br i1 undef, label %"62", label %"82"

"62":                                             ; preds = %"61"
  br i1 undef, label %"100", label %"63"

"63":                                             ; preds = %"62"
  br i1 undef, label %"65", label %"64"

"64":                                             ; preds = %"63"
  unreachable

"65":                                             ; preds = %"63"
  br i1 undef, label %"66", label %"67"

"66":                                             ; preds = %"65"
  unreachable

"67":                                             ; preds = %"65"
  %1 = load i32* undef, align 4                   ; <i32> [#uses=0]
  br label %"100"

"82":                                             ; preds = %"61", %"60", %"59"
  unreachable

"93":                                             ; preds = %"19"
  %2 = call fastcc %"union tree_node"* @get_computation_at(%"struct loop"* undef, %"struct iv_use"* %use, %"struct iv_cand"* %cand, %"union gimple_statement_d"* %at) nounwind ; <%"union tree_node"*> [#uses=1]
  br i1 undef, label %"100", label %"97"

"97":                                             ; preds = %"93"
  br i1 undef, label %"99", label %"98"

"98":                                             ; preds = %"97"
  br label %"99"

"99":                                             ; preds = %"98", %"97"
  %3 = phi %"union tree_node"* [ undef, %"98" ], [ %2, %"97" ] ; <%"union tree_node"*> [#uses=1]
  %4 = call fastcc i32 @computation_cost(%"union tree_node"* %3, i8 zeroext undef) nounwind ; <i32> [#uses=1]
  br label %"100"

"100":                                            ; preds = %"99", %"93", %"67", %"62", %"25", %"23", %"8", %entry
  %memtmp1.1.0 = phi i32 [ 0, %"99" ], [ 10000000, %entry ], [ 10000000, %"8" ], [ 10000000, %"23" ], [ 10000000, %"25" ], [ undef, %"62" ], [ undef, %"67" ], [ 10000000, %"93" ] ; <i32> [#uses=1]
  %memtmp1.0.0 = phi i32 [ %4, %"99" ], [ 10000000, %entry ], [ 10000000, %"8" ], [ 10000000, %"23" ], [ 10000000, %"25" ], [ undef, %"62" ], [ undef, %"67" ], [ 10000000, %"93" ] ; <i32> [#uses=1]
  %5 = zext i32 %memtmp1.0.0 to i64               ; <i64> [#uses=1]
  %6 = zext i32 %memtmp1.1.0 to i64               ; <i64> [#uses=1]
  %7 = shl i64 %6, 32                             ; <i64> [#uses=1]
  %8 = or i64 %7, %5                              ; <i64> [#uses=1]
  ret i64 %8
}
