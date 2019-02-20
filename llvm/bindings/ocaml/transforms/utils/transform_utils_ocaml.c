/*===-- transform_utils_ocaml.c - LLVM OCaml Glue ---------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
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

#include "llvm-c/Core.h"
#include "caml/mlvalues.h"
#include "caml/misc.h"

/*
 * Do not move directly into external. This function is here to pull in
 * -lLLVMTransformUtils, which would otherwise be not linked on static builds,
 * as ld can't see the reference from OCaml code.
 */

/* llmodule -> llmodule */
CAMLprim LLVMModuleRef llvm_clone_module(LLVMModuleRef M) {
  return LLVMCloneModule(M);
}
