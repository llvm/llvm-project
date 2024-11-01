/*===-- ipo_ocaml.c - LLVM OCaml Glue ---------------------------*- C++ -*-===*\
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

#include "llvm-c/Transforms/IPO.h"
#include "caml/mlvalues.h"
#include "caml/misc.h"

/* [`Module] Llvm.PassManager.t -> unit */
value llvm_add_constant_merge(LLVMPassManagerRef PM) {
  LLVMAddConstantMergePass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
value llvm_add_dead_arg_elimination(LLVMPassManagerRef PM) {
  LLVMAddDeadArgEliminationPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
value llvm_add_function_attrs(LLVMPassManagerRef PM) {
  LLVMAddFunctionAttrsPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
value llvm_add_always_inliner(LLVMPassManagerRef PM) {
  LLVMAddAlwaysInlinerPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
value llvm_add_global_dce(LLVMPassManagerRef PM) {
  LLVMAddGlobalDCEPass(PM);
  return Val_unit;
}
