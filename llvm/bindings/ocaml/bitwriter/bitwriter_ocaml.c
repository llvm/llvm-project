/*===-- bitwriter_ocaml.c - LLVM Ocaml Glue ---------------------*- C++ -*-===*\
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

#include "llvm-c/BitWriter.h"
#include "llvm-c/Core.h"
#include "caml/alloc.h"
#include "caml/mlvalues.h"
#include "caml/memory.h"

/*===-- Modules -----------------------------------------------------------===*/

/* Llvm.llmodule -> string -> bool */
CAMLprim value llvm_write_bitcode_file(value M, value Path) {
  int res = LLVMWriteBitcodeToFile((LLVMModuleRef) M, String_val(Path));
  return Val_bool(res == 0);
}

/* ?unbuffered:bool -> Llvm.llmodule -> Unix.file_descr -> bool */
CAMLprim value llvm_write_bitcode_to_fd(value U, value M, value FD) {
  int Unbuffered;
  int res;

  if (U == Val_int(0)) {
    Unbuffered = 0;
  } else {
    Unbuffered = Bool_val(Field(U,0));
  }

  res = LLVMWriteBitcodeToFD((LLVMModuleRef) M, Int_val(FD), 0, Unbuffered);
  return Val_bool(res == 0);
}
