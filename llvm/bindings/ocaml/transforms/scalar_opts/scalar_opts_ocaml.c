/*===-- scalar_opts_ocaml.c - LLVM OCaml Glue -------------------*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file glues LLVM's OCaml interface to its C interface. These functions *|
|* are by and large transparent wrappers to the corresponding C functions.    *|
|*                                                                            *|
|* Note that these functions intentionally take liberties with the CAMLparamX *|
|* macros, since most of the parameters are not GC heap objects.              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c/Transforms/Scalar.h"
#include "llvm-c/Transforms/Utils.h"
#include "caml/mlvalues.h"
#include "caml/misc.h"

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_aggressive_dce(LLVMPassManagerRef PM) {
  LLVMAddAggressiveDCEPass(PM);
  return Val_unit;
}

value llvm_add_dce(LLVMPassManagerRef PM) {
  LLVMAddDCEPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_alignment_from_assumptions(LLVMPassManagerRef PM) {
  LLVMAddAlignmentFromAssumptionsPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_cfg_simplification(LLVMPassManagerRef PM) {
  LLVMAddCFGSimplificationPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_dead_store_elimination(LLVMPassManagerRef PM) {
  LLVMAddDeadStoreEliminationPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_scalarizer(LLVMPassManagerRef PM) {
  LLVMAddScalarizerPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_merged_load_store_motion(LLVMPassManagerRef PM) {
  LLVMAddMergedLoadStoreMotionPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_gvn(LLVMPassManagerRef PM) {
  LLVMAddGVNPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_ind_var_simplify(LLVMPassManagerRef PM) {
  LLVMAddIndVarSimplifyPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_instruction_combining(LLVMPassManagerRef PM) {
  LLVMAddInstructionCombiningPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_jump_threading(LLVMPassManagerRef PM) {
  LLVMAddJumpThreadingPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_licm(LLVMPassManagerRef PM) {
  LLVMAddLICMPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_loop_rotate(LLVMPassManagerRef PM) {
  LLVMAddLoopRotatePass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_loop_unroll(LLVMPassManagerRef PM) {
  LLVMAddLoopUnrollPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_memcpy_opt(LLVMPassManagerRef PM) {
  LLVMAddMemCpyOptPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_partially_inline_lib_calls(LLVMPassManagerRef PM) {
  LLVMAddPartiallyInlineLibCallsPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_lower_atomic(LLVMPassManagerRef PM) {
  LLVMAddLowerAtomicPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_lower_switch(LLVMPassManagerRef PM) {
  LLVMAddLowerSwitchPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_promote_memory_to_register(LLVMPassManagerRef PM) {
  LLVMAddPromoteMemoryToRegisterPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_reassociation(LLVMPassManagerRef PM) {
  LLVMAddReassociatePass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_sccp(LLVMPassManagerRef PM) {
  LLVMAddSCCPPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_scalar_repl_aggregates(LLVMPassManagerRef PM) {
  LLVMAddScalarReplAggregatesPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_scalar_repl_aggregates_ssa(LLVMPassManagerRef PM) {
  LLVMAddScalarReplAggregatesPassSSA(PM);
  return Val_unit;
}

/* int -> [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_scalar_repl_aggregates_with_threshold(value threshold,
                                                     LLVMPassManagerRef PM) {
  LLVMAddScalarReplAggregatesPassWithThreshold(PM, Int_val(threshold));
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_simplify_lib_calls(LLVMPassManagerRef PM) {
  LLVMAddSimplifyLibCallsPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_tail_call_elimination(LLVMPassManagerRef PM) {
  LLVMAddTailCallEliminationPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_demote_memory_to_register(LLVMPassManagerRef PM) {
  LLVMAddDemoteMemoryToRegisterPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_verifier(LLVMPassManagerRef PM) {
  LLVMAddVerifierPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_correlated_value_propagation(LLVMPassManagerRef PM) {
  LLVMAddCorrelatedValuePropagationPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_early_cse(LLVMPassManagerRef PM) {
  LLVMAddEarlyCSEPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_lower_expect_intrinsic(LLVMPassManagerRef PM) {
  LLVMAddLowerExpectIntrinsicPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_lower_constant_intrinsics(LLVMPassManagerRef PM) {
  LLVMAddLowerConstantIntrinsicsPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_type_based_alias_analysis(LLVMPassManagerRef PM) {
  LLVMAddTypeBasedAliasAnalysisPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_scoped_no_alias_aa(LLVMPassManagerRef PM) {
  LLVMAddScopedNoAliasAAPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_basic_alias_analysis(LLVMPassManagerRef PM) {
  LLVMAddBasicAliasAnalysisPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
value llvm_add_unify_function_exit_nodes(LLVMPassManagerRef PM) {
  LLVMAddUnifyFunctionExitNodesPass(PM);
  return Val_unit;
}
