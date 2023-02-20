(*===-- llvm_scalar_opts.ml - LLVM OCaml Interface ------------*- OCaml -*-===*
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*)

external add_aggressive_dce
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_aggressive_dce"
external add_dce
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_dce"
external add_alignment_from_assumptions
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_alignment_from_assumptions"
external add_cfg_simplification
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_cfg_simplification"
external add_dead_store_elimination
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_dead_store_elimination"
external add_scalarizer
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_scalarizer"
external add_merged_load_store_motion
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_merged_load_store_motion"
external add_gvn
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_gvn"
external add_ind_var_simplification
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_ind_var_simplify"
external add_instruction_combination
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_instruction_combining"
external add_jump_threading
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_jump_threading"
external add_licm
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_licm"
external add_loop_rotation
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_loop_rotate"
external add_loop_unroll
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_loop_unroll"
external add_memcpy_opt
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_memcpy_opt"
external add_partially_inline_lib_calls
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_partially_inline_lib_calls"
external add_lower_atomic
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_lower_atomic"
external add_lower_switch
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_lower_switch"
external add_memory_to_register_promotion
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_promote_memory_to_register"
external add_reassociation
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_reassociation"
external add_sccp
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_sccp"
external add_scalar_repl_aggregation
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_scalar_repl_aggregates"
external add_scalar_repl_aggregation_ssa
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_scalar_repl_aggregates_ssa"
external add_scalar_repl_aggregation_with_threshold
  : int -> [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_scalar_repl_aggregates_with_threshold"
external add_lib_call_simplification
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_simplify_lib_calls"
external add_tail_call_elimination
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_tail_call_elimination"
external add_memory_to_register_demotion
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_demote_memory_to_register"
external add_verifier
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_verifier"
external add_correlated_value_propagation
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_correlated_value_propagation"
external add_early_cse
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_early_cse"
external add_lower_expect_intrinsic
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_lower_expect_intrinsic"
external add_lower_constant_intrinsics
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_lower_constant_intrinsics"
external add_type_based_alias_analysis
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_type_based_alias_analysis"
external add_scoped_no_alias_alias_analysis
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_scoped_no_alias_aa"
external add_basic_alias_analysis
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_basic_alias_analysis"
external add_unify_function_exit_nodes
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_unify_function_exit_nodes"
