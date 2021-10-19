/*===-- transform_utils_ocaml.c - LLVM OCaml Glue ---------------*- C++ -*-===*\
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

#include "llvm-c/Core.h"
#include "caml/mlvalues.h"
#include "caml/misc.h"

/*
 * Do not move directly into external. This function is here to pull in
 * -lLLVMTransformUtils, which would otherwise be not linked on static builds,
 * as ld can't see the reference from OCaml code.
 */

/* llmodule -> llmodule */
LLVMModuleRef llvm_clone_module(LLVMModuleRef M) { return LLVMCloneModule(M); }
