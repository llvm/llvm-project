/*===-- ipo_ocaml.c - LLVM Ocaml Glue -------------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file glues LLVM's ocaml interface to its C interface. These functions *|
|* are by and large transparent wrappers to the corresponding C functions.    *|
|*                                                                            *|
|* Note that these functions intentionally take liberties with the CAMLparamX *|
|* macros, since most of the parameters are not GC heap objects.              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c/Transforms/IPO.h"
#include "caml/mlvalues.h"
#include "caml/misc.h"

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_argument_promotion(LLVMPassManagerRef PM) {
  LLVMAddArgumentPromotionPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_constant_merge(LLVMPassManagerRef PM) {
  LLVMAddConstantMergePass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_dead_arg_elimination(LLVMPassManagerRef PM) {
  LLVMAddDeadArgEliminationPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_function_attrs(LLVMPassManagerRef PM) {
  LLVMAddFunctionAttrsPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_function_inlining(LLVMPassManagerRef PM) {
  LLVMAddFunctionInliningPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_always_inliner_pass(LLVMPassManagerRef PM) {
  LLVMAddAlwaysInlinerPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_global_dce(LLVMPassManagerRef PM) {
  LLVMAddGlobalDCEPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_global_optimizer(LLVMPassManagerRef PM) {
  LLVMAddGlobalOptimizerPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_ipc_propagation(LLVMPassManagerRef PM) {
  LLVMAddIPConstantPropagationPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_prune_eh(LLVMPassManagerRef PM) {
  LLVMAddPruneEHPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_ipsccp(LLVMPassManagerRef PM) {
  LLVMAddIPSCCPPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> bool -> unit */
CAMLprim value llvm_add_internalize(LLVMPassManagerRef PM, value AllButMain) {
  LLVMAddInternalizePass(PM, Bool_val(AllButMain));
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_strip_dead_prototypes(LLVMPassManagerRef PM) {
  LLVMAddStripDeadPrototypesPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_strip_symbols(LLVMPassManagerRef PM) {
  LLVMAddStripSymbolsPass(PM);
  return Val_unit;
}
