(*===-- llvm_scalar_opts.mli - LLVM OCaml Interface -----------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Scalar Transforms.

    This interface provides an OCaml API for LLVM scalar transforms, the
    classes in the [LLVMScalarOpts] library. *)

(** See the [llvm::createAggressiveDCEPass] function. *)
external add_aggressive_dce
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_aggressive_dce"

(** See the [llvm::createAlignmentFromAssumptionsPass] function. *)
external add_alignment_from_assumptions
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_alignment_from_assumptions"

(** See the [llvm::createCFGSimplificationPass] function. *)
external add_cfg_simplification
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_cfg_simplification"

(** See [llvm::createDeadStoreEliminationPass] function. *)
external add_dead_store_elimination
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_dead_store_elimination"

(** See [llvm::createScalarizerPass] function. *)
external add_scalarizer
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_scalarizer"

(** See [llvm::createMergedLoadStoreMotionPass] function. *)
external add_merged_load_store_motion
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_merged_load_store_motion"

(** See the [llvm::createGVNPass] function. *)
external add_gvn
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_gvn"

(** See the [llvm::createIndVarSimplifyPass] function. *)
external add_ind_var_simplification
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_ind_var_simplify"

(** See the [llvm::createInstructionCombiningPass] function. *)
external add_instruction_combination
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_instruction_combining"

(** See the [llvm::createJumpThreadingPass] function. *)
external add_jump_threading
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_jump_threading"

(** See the [llvm::createLICMPass] function. *)
external add_licm
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_licm"

(** See the [llvm::createLoopDeletionPass] function. *)
external add_loop_deletion
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_loop_deletion"

(** See the [llvm::createLoopIdiomPass] function. *)
external add_loop_idiom
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_loop_idiom"

(** See the [llvm::createLoopRotatePass] function. *)
external add_loop_rotation
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_loop_rotate"

(** See the [llvm::createLoopRerollPass] function. *)
external add_loop_reroll
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_loop_reroll"

(** See the [llvm::createLoopUnrollPass] function. *)
external add_loop_unroll
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_loop_unroll"

(** See the [llvm::createLoopUnswitchPass] function. *)
external add_loop_unswitch
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_loop_unswitch"

(** See the [llvm::createMemCpyOptPass] function. *)
external add_memcpy_opt
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_memcpy_opt"

(** See the [llvm::createPartiallyInlineLibCallsPass] function. *)
external add_partially_inline_lib_calls
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_partially_inline_lib_calls"

(** See the [llvm::createLowerAtomicPass] function. *)
external add_lower_atomic
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_lower_atomic"

(** See the [llvm::createLowerSwitchPass] function. *)
external add_lower_switch
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_lower_switch"

(** See the [llvm::createPromoteMemoryToRegisterPass] function. *)
external add_memory_to_register_promotion
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_promote_memory_to_register"

(** See the [llvm::createReassociatePass] function. *)
external add_reassociation
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_reassociation"

(** See the [llvm::createSCCPPass] function. *)
external add_sccp
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_sccp"

(** See the [llvm::createSROAPass] function. *)
external add_scalar_repl_aggregation
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_scalar_repl_aggregates"

(** See the [llvm::createSROAPass] function. *)
external add_scalar_repl_aggregation_ssa
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_scalar_repl_aggregates_ssa"

(** See the [llvm::createSROAPass] function. *)
external add_scalar_repl_aggregation_with_threshold
  : int -> [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_scalar_repl_aggregates_with_threshold"

(** See the [llvm::createSimplifyLibCallsPass] function. *)
external add_lib_call_simplification
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_simplify_lib_calls"

(** See the [llvm::createTailCallEliminationPass] function. *)
external add_tail_call_elimination
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_tail_call_elimination"

(** See the [llvm::createConstantPropagationPass] function. *)
external add_constant_propagation
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_constant_propagation"

(** See the [llvm::createDemoteMemoryToRegisterPass] function. *)
external add_memory_to_register_demotion
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_demote_memory_to_register"

(** See the [llvm::createVerifierPass] function. *)
external add_verifier
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_verifier"

(** See the [llvm::createCorrelatedValuePropagationPass] function. *)
external add_correlated_value_propagation
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_correlated_value_propagation"

(** See the [llvm::createEarlyCSE] function. *)
external add_early_cse
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_early_cse"

(** See the [llvm::createLowerExpectIntrinsicPass] function. *)
external add_lower_expect_intrinsic
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_lower_expect_intrinsic"

(** See the [llvm::createTypeBasedAliasAnalysisPass] function. *)
external add_type_based_alias_analysis
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_type_based_alias_analysis"

(** See the [llvm::createScopedNoAliasAAPass] function. *)
external add_scoped_no_alias_alias_analysis
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_scoped_no_alias_aa"

(** See the [llvm::createBasicAliasAnalysisPass] function. *)
external add_basic_alias_analysis
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_basic_alias_analysis"

(** See the [llvm::createUnifyFunctionExitNodesPass] function. *)
external add_unify_function_exit_nodes
  : [< Llvm.PassManager.any ] Llvm.PassManager.t -> unit
  = "llvm_add_unify_function_exit_nodes"
