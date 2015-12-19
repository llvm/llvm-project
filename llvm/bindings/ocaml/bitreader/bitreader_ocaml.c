/*===-- bitwriter_ocaml.c - LLVM OCaml Glue ---------------------*- C++ -*-===*\
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

#include "llvm-c/BitReader.h"
#include "caml/alloc.h"
#include "caml/fail.h"
#include "caml/memory.h"
#include "caml/callback.h"

void llvm_raise(value Prototype, char *Message);

/* Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule */
CAMLprim LLVMModuleRef llvm_get_module(LLVMContextRef C, LLVMMemoryBufferRef MemBuf) {
  LLVMModuleRef M;

  if (LLVMGetBitcodeModuleInContext2(C, MemBuf, &M))
    llvm_raise(*caml_named_value("Llvm_bitreader.Error"), "");

  return M;
}

/* Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule */
CAMLprim LLVMModuleRef llvm_parse_bitcode(LLVMContextRef C, LLVMMemoryBufferRef MemBuf) {
  LLVMModuleRef M;

  if (LLVMParseBitcodeInContext2(C, MemBuf, &M))
    llvm_raise(*caml_named_value("Llvm_bitreader.Error"), "");

  return M;
}
