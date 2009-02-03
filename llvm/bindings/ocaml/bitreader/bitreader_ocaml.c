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
\*===----------------------------------------------------------------------===*/

#include "llvm-c/BitReader.h"
#include "caml/alloc.h"
#include "caml/fail.h"
#include "caml/memory.h"


/* Can't use the recommended caml_named_value mechanism for backwards
   compatibility reasons. This is largely equivalent. */
static value llvm_bitreader_error_exn;

CAMLprim value llvm_register_bitreader_exns(value Error) {
  llvm_bitreader_error_exn = Field(Error, 0);
  register_global_root(&llvm_bitreader_error_exn);
  return Val_unit;
}

static void llvm_raise(value Prototype, char *Message) {
  CAMLparam1(Prototype);
  CAMLlocal1(CamlMessage);
  
  CamlMessage = copy_string(Message);
  LLVMDisposeMessage(Message);
  
  raise_with_arg(Prototype, CamlMessage);
  abort(); /* NOTREACHED */
#ifdef CAMLnoreturn
  CAMLnoreturn; /* Silences warnings, but is missing in some versions. */
#endif
}


/*===-- Modules -----------------------------------------------------------===*/

/* Llvm.llmemorybuffer -> Llvm.module */
CAMLprim value llvm_get_module_provider(LLVMMemoryBufferRef MemBuf) {
  CAMLparam0();
  CAMLlocal2(Variant, MessageVal);
  char *Message;
  
  LLVMModuleProviderRef MP;
  if (LLVMGetBitcodeModuleProvider(MemBuf, &MP, &Message))
    llvm_raise(llvm_bitreader_error_exn, Message);
  
  CAMLreturn((value) MemBuf);
}

/* Llvm.llmemorybuffer -> Llvm.llmodule */
CAMLprim value llvm_parse_bitcode(LLVMMemoryBufferRef MemBuf) {
  CAMLparam0();
  CAMLlocal2(Variant, MessageVal);
  LLVMModuleRef M;
  char *Message;
  
  if (LLVMParseBitcode(MemBuf, &M, &Message))
    llvm_raise(llvm_bitreader_error_exn, Message);
  
  CAMLreturn((value) M);
}
