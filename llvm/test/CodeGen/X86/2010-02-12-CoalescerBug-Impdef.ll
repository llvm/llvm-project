; RUN: llc < %s > %t
; PR6283

; Tricky coalescer bug:
; After coalescing %RAX with a virtual register, this instruction was rematted:
;
;   %EAX = MOV32rr killed %reg1070
;
; This instruction silently defined %RAX, and when rematting removed the
; instruction, the live interval for %RAX was not properly updated. The valno
; referred to a deleted instruction and bad things happened.
;
; The fix is to implicitly define %RAX when coalescing:
;
;   %EAX = MOV32rr killed %reg1070, implicit-def %RAX
;

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

module asm "\09.ident\09\22GCC: (GNU) 4.5.0 20100212 (experimental) LLVM: 95975\22"

%0 = type { ptr }
%"BITMAP_WORD[]" = type [2 x i64]
%"uchar[]" = type [1 x i8]
%"char[]" = type [4 x i8]
%"enum dom_state[]" = type [2 x i32]
%"int[]" = type [4 x i32]
%"struct VEC_basic_block_base" = type { i32, i32, [1 x ptr] }
%"struct VEC_basic_block_gc" = type { %"struct VEC_basic_block_base" }
%"struct VEC_edge_base" = type { i32, i32, [1 x ptr] }
%"struct VEC_edge_gc" = type { %"struct VEC_edge_base" }
%"struct VEC_gimple_base" = type { i32, i32, [1 x ptr] }
%"struct VEC_gimple_gc" = type { %"struct VEC_gimple_base" }
%"struct VEC_iv_cand_p_base" = type { i32, i32, [1 x ptr] }
%"struct VEC_iv_cand_p_heap" = type { %"struct VEC_iv_cand_p_base" }
%"struct VEC_iv_use_p_base" = type { i32, i32, [1 x ptr] }
%"struct VEC_iv_use_p_heap" = type { %"struct VEC_iv_use_p_base" }
%"struct VEC_loop_p_base" = type { i32, i32, [1 x ptr] }
%"struct VEC_loop_p_gc" = type { %"struct VEC_loop_p_base" }
%"struct VEC_rtx_base" = type { i32, i32, [1 x ptr] }
%"struct VEC_rtx_gc" = type { %"struct VEC_rtx_base" }
%"struct VEC_tree_base" = type { i32, i32, [1 x ptr] }
%"struct VEC_tree_gc" = type { %"struct VEC_tree_base" }
%"struct _obstack_chunk" = type { ptr, ptr, %"char[]" }
%"struct basic_block_def" = type { ptr, ptr, ptr, ptr, [2 x ptr], ptr, ptr, %"union basic_block_il_dependent", i64, i32, i32, i32, i32, i32 }
%"struct bitmap_element" = type { ptr, ptr, i32, %"BITMAP_WORD[]" }
%"struct bitmap_head_def" = type { ptr, ptr, i32, ptr }
%"struct bitmap_obstack" = type { ptr, ptr, %"struct obstack" }
%"struct block_symbol" = type { [3 x %"union rtunion"], ptr, i64 }
%"struct comp_cost" = type { i32, i32 }
%"struct control_flow_graph" = type { ptr, ptr, ptr, i32, i32, i32, ptr, i32, %"enum dom_state[]", %"enum dom_state[]", i32, i32 }
%"struct cost_pair" = type { ptr, %"struct comp_cost", ptr, ptr }
%"struct def_optype_d" = type { ptr, ptr }
%"struct double_int" = type { i64, i64 }
%"struct edge_def" = type { ptr, ptr, %"union edge_def_insns", ptr, ptr, i32, i32, i32, i32, i64 }
%"struct eh_status" = type opaque
%"struct et_node" = type opaque
%"struct function" = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, ptr, i8, i8, i8, i8 }
%"struct gimple_bb_info" = type { ptr, ptr }
%"struct gimple_df" = type { ptr, ptr, ptr, ptr, %"struct pt_solution", %"struct pt_solution", ptr, ptr, ptr, ptr, i8, %"struct ssa_operands" }
%"struct gimple_seq_d" = type { ptr, ptr, ptr }
%"struct gimple_seq_node_d" = type { ptr, ptr, ptr }
%"struct gimple_statement_base" = type { i8, i8, i16, i32, i32, i32, ptr, ptr }
%"struct phi_arg_d[]" = type [1 x %"struct phi_arg_d"]
%"struct gimple_statement_phi" = type { %"struct gimple_statement_base", i32, i32, ptr, %"struct phi_arg_d[]" }
%"struct htab" = type { ptr, ptr, ptr, ptr, i64, i64, i64, i32, i32, ptr, ptr, ptr, ptr, ptr, i32 }
%"struct iv" = type { ptr, ptr, ptr, ptr, i8, i8, i32 }
%"struct iv_cand" = type { i32, i8, i32, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr }
%"struct iv_use" = type { i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr }
%"struct ivopts_data" = type { ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, i32, i8, i8 }
%"struct lang_decl" = type opaque
%"struct language_function" = type opaque
%"struct loop" = type { i32, i32, ptr, ptr, %"struct comp_cost", i32, i32, ptr, ptr, ptr, ptr, ptr, %"struct double_int", %"struct double_int", i8, i8, i32, ptr, ptr, i8, ptr }
%"struct loop_exit" = type { ptr, ptr, ptr, ptr }
%"struct loops" = type { i32, ptr, ptr, ptr }
%"struct machine_cfa_state" = type { ptr, i64 }
%"struct machine_function" = type { ptr, ptr, i32, i32, %"int[]", i32, %"struct machine_cfa_state", i32, i8 }
%"struct nb_iter_bound" = type { ptr, %"struct double_int", i8, ptr }
%"struct object_block" = type { ptr, i32, i64, ptr, ptr }
%"struct obstack" = type { i64, ptr, ptr, ptr, ptr, i64, i32, ptr, ptr, ptr, i8 }
%"struct phi_arg_d" = type { %"struct ssa_use_operand_d", ptr, i32 }
%"struct pointer_map_t" = type opaque
%"struct pt_solution" = type { i8, ptr }
%"struct rtx_def" = type { i16, i8, i8, %"union u" }
%"struct section_common" = type { i32 }
%"struct ssa_operand_memory_d" = type { ptr, %"uchar[]" }
%"struct ssa_operands" = type { ptr, i32, i32, i8, ptr, ptr }
%"struct ssa_use_operand_d" = type { ptr, ptr, %0, ptr }
%"struct stack_local_entry" = type opaque
%"struct tree_base" = type <{ i16, i8, i8, i8, [2 x i8], i8 }>
%"struct tree_common" = type { %"struct tree_base", ptr, ptr }
%"struct tree_decl_common" = type { %"struct tree_decl_minimal", ptr, i8, i8, i8, i8, i8, i32, ptr, ptr, ptr, ptr, ptr }
%"struct tree_decl_minimal" = type { %"struct tree_common", i32, i32, ptr, ptr }
%"struct tree_decl_non_common" = type { %"struct tree_decl_with_vis", ptr, ptr, ptr, ptr }
%"struct tree_decl_with_rtl" = type { %"struct tree_decl_common", ptr }
%"struct tree_decl_with_vis" = type { %"struct tree_decl_with_rtl", ptr, ptr, ptr, i8, i8, i8 }
%"struct tree_function_decl" = type { %"struct tree_decl_non_common", ptr, ptr, ptr, ptr, i16, i8, i8 }
%"struct unnamed_section" = type { %"struct section_common", ptr, ptr, ptr }
%"struct use_optype_d" = type { ptr, %"struct ssa_use_operand_d" }
%"struct version_info" = type { ptr, ptr, i8, i32, i8 }
%"union basic_block_il_dependent" = type { ptr }
%"union edge_def_insns" = type { ptr }
%"union gimple_statement_d" = type { %"struct gimple_statement_phi" }
%"union rtunion" = type { ptr }
%"union section" = type { %"struct unnamed_section" }
%"union tree_node" = type { %"struct tree_function_decl" }
%"union u" = type { %"struct block_symbol" }

declare fastcc ptr @get_computation_at(ptr, ptr nocapture, ptr nocapture, ptr) nounwind

declare fastcc i32 @computation_cost(ptr, i8 zeroext) nounwind

define fastcc i64 @get_computation_cost_at(ptr %data, ptr nocapture %use, ptr nocapture %cand, i8 zeroext %address_p, ptr %depends_on, ptr %at, ptr %can_autoinc) nounwind {
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
  %1 = load i32, ptr undef, align 4                   ; <i32> [#uses=0]
  br label %"100"

"82":                                             ; preds = %"61", %"60", %"59"
  unreachable

"93":                                             ; preds = %"19"
  %2 = call fastcc ptr @get_computation_at(ptr undef, ptr %use, ptr %cand, ptr %at) nounwind ; <ptr> [#uses=1]
  br i1 undef, label %"100", label %"97"

"97":                                             ; preds = %"93"
  br i1 undef, label %"99", label %"98"

"98":                                             ; preds = %"97"
  br label %"99"

"99":                                             ; preds = %"98", %"97"
  %3 = phi ptr [ undef, %"98" ], [ %2, %"97" ] ; <ptr> [#uses=1]
  %4 = call fastcc i32 @computation_cost(ptr %3, i8 zeroext undef) nounwind ; <i32> [#uses=1]
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
