/*===-- analysis_ocaml.c - LLVM Ocaml Glue ----------------------*- C++ -*-===*\
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

#include "llvm-c/Analysis.h"
#include "caml/alloc.h"
#include "caml/mlvalues.h"
#include "caml/memory.h"


/* Llvm.llmodule -> string option */
CAMLprim value llvm_verify_module(LLVMModuleRef M) {
  CAMLparam0();
  CAMLlocal2(String, Option);
  
  char *Message;
  int Result = LLVMVerifyModule(M, LLVMReturnStatusAction, &Message);
  
  if (0 == Result) {
    Option = Val_int(0);
  } else {
    Option = alloc(1, 0);
    String = copy_string(Message);
    Store_field(Option, 0, String);
  }
  
  LLVMDisposeMessage(Message);
  
  CAMLreturn(Option);
}

/* Llvm.llvalue -> bool */
CAMLprim value llvm_verify_function(LLVMValueRef Fn) {
  return Val_bool(LLVMVerifyFunction(Fn, LLVMReturnStatusAction) == 0);
}

/* Llvm.llmodule -> unit */
CAMLprim value llvm_assert_valid_module(LLVMModuleRef M) {
  LLVMVerifyModule(M, LLVMAbortProcessAction, 0);
  return Val_unit;
}

/* Llvm.llvalue -> unit */
CAMLprim value llvm_assert_valid_function(LLVMValueRef Fn) {
  LLVMVerifyFunction(Fn, LLVMAbortProcessAction);
  return Val_unit;
}

/* Llvm.llvalue -> unit */
CAMLprim value llvm_view_function_cfg(LLVMValueRef Fn) {
  LLVMViewFunctionCFG(Fn);
  return Val_unit;
}

/* Llvm.llvalue -> unit */
CAMLprim value llvm_view_function_cfg_only(LLVMValueRef Fn) {
  LLVMViewFunctionCFGOnly(Fn);
  return Val_unit;
}
