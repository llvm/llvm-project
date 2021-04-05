/*===-- bitwriter_ocaml.c - LLVM OCaml Glue ---------------------*- C++ -*-===*\
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
\*===----------------------------------------------------------------------===*/

#include "llvm-c/BitReader.h"
#include "llvm-c/Core.h"
#include "caml/alloc.h"
#include "caml/fail.h"
#include "caml/memory.h"
#include "caml/callback.h"

void llvm_raise(value Prototype, char *Message);

/* Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule */
LLVMModuleRef llvm_get_module(LLVMContextRef C, LLVMMemoryBufferRef MemBuf) {
  LLVMModuleRef M;

  if (LLVMGetBitcodeModuleInContext2(C, MemBuf, &M))
    llvm_raise(*caml_named_value("Llvm_bitreader.Error"), LLVMCreateMessage(""));

  return M;
}

/* Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule */
LLVMModuleRef llvm_parse_bitcode(LLVMContextRef C, LLVMMemoryBufferRef MemBuf) {
  LLVMModuleRef M;

  if (LLVMParseBitcodeInContext2(C, MemBuf, &M))
    llvm_raise(*caml_named_value("Llvm_bitreader.Error"), LLVMCreateMessage(""));

  return M;
}
