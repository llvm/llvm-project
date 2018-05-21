/*===-- tapir_opts_ocaml.c - LLVM OCaml Glue --------------------*- C++ -*-===*\
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
\*===----------------------------------------------------------------------===*/

#include "caml/custom.h"
#include "llvm-c/Transforms/Tapir.h"

/* [`Module] Llvm.PassManager.t -> unit
 */
CAMLprim value llvm_add_lower_tapir_to_cilk(LLVMPassManagerRef PM,
                                            TapirTargetRef tt)
{
    LLVMAddLowerTapirToCilk(PM, TapirTarget_val(tt));
    return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit
 */
CAMLprim value llvm_add_loop_spawning(LLVMPassManagerRef PM,
                                      TapirTargetRef tt)
{
    LLVMAddLoopSpawning(PM, TapirTarget_val(tt));
    return Val_unit;
}
