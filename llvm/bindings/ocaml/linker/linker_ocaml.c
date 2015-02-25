/*===-- linker_ocaml.c - LLVM OCaml Glue ------------------------*- C++ -*-===*\
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

#include "llvm-c/Linker.h"
#include "caml/alloc.h"
#include "caml/memory.h"
#include "caml/fail.h"
#include "caml/callback.h"

void llvm_raise(value Prototype, char *Message);

/* llmodule -> llmodule -> Mode.t -> unit */
CAMLprim value llvm_link_modules(LLVMModuleRef Dst, LLVMModuleRef Src, value Mode) {
  char* Message;

  if (LLVMLinkModules(Dst, Src, Int_val(Mode), &Message))
    llvm_raise(*caml_named_value("Llvm_linker.Error"), Message);

  return Val_unit;
}
