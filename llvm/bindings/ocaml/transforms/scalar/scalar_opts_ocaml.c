/*===-- scalar_opts_ocaml.c - LLVM Ocaml Glue -------------------*- C++ -*-===*\
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

#include "llvm-c/Transforms/Scalar.h"
#include "caml/mlvalues.h"
#include "caml/misc.h"

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_constant_propagation(LLVMPassManagerRef PM) {
  LLVMAddConstantPropagationPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_instruction_combining(LLVMPassManagerRef PM) {
  LLVMAddInstructionCombiningPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_memory_to_register_promotion(LLVMPassManagerRef PM) {
  LLVMAddPromoteMemoryToRegisterPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_memory_to_register_demotion(LLVMPassManagerRef PM) {
  LLVMAddDemoteMemoryToRegisterPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_reassociation(LLVMPassManagerRef PM) {
  LLVMAddReassociatePass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_gvn(LLVMPassManagerRef PM) {
  LLVMAddGVNPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_cfg_simplification(LLVMPassManagerRef PM) {
  LLVMAddCFGSimplificationPass(PM);
  return Val_unit;
}
