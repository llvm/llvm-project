/*===-- executionengine_ocaml.c - LLVM OCaml Glue ---------------*- C++ -*-===*\
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

#include <string.h>
#include <assert.h>
#include "llvm-c/Core.h"
#include "llvm-c/ExecutionEngine.h"
#include "llvm-c/Target.h"
#include "caml/alloc.h"
#include "caml/custom.h"
#include "caml/fail.h"
#include "caml/memory.h"
#include "caml/callback.h"

void llvm_raise(value Prototype, char *Message);

/* unit -> bool */
CAMLprim value llvm_ee_initialize(value Unit) {
  LLVMLinkInMCJIT();

  return Val_bool(!LLVMInitializeNativeTarget() &&
                  !LLVMInitializeNativeAsmParser() &&
                  !LLVMInitializeNativeAsmPrinter());
}

/* llmodule -> llcompileroption -> ExecutionEngine.t */
CAMLprim LLVMExecutionEngineRef llvm_ee_create(value OptRecordOpt, LLVMModuleRef M) {
  value OptRecord;
  LLVMExecutionEngineRef MCJIT;
  char *Error;
  struct LLVMMCJITCompilerOptions Options;

  LLVMInitializeMCJITCompilerOptions(&Options, sizeof(Options));
  if (OptRecordOpt != Val_int(0)) {
    OptRecord = Field(OptRecordOpt, 0);
    Options.OptLevel = Int_val(Field(OptRecord, 0));
    Options.CodeModel = Int_val(Field(OptRecord, 1));
    Options.NoFramePointerElim = Int_val(Field(OptRecord, 2));
    Options.EnableFastISel = Int_val(Field(OptRecord, 3));
    Options.MCJMM = NULL;
  }

  if (LLVMCreateMCJITCompilerForModule(&MCJIT, M, &Options,
                                      sizeof(Options), &Error))
    llvm_raise(*caml_named_value("Llvm_executionengine.Error"), Error);
  return MCJIT;
}

/* ExecutionEngine.t -> unit */
CAMLprim value llvm_ee_dispose(LLVMExecutionEngineRef EE) {
  LLVMDisposeExecutionEngine(EE);
  return Val_unit;
}

/* llmodule -> ExecutionEngine.t -> unit */
CAMLprim value llvm_ee_add_module(LLVMModuleRef M, LLVMExecutionEngineRef EE) {
  LLVMAddModule(EE, M);
  return Val_unit;
}

/* llmodule -> ExecutionEngine.t -> llmodule */
CAMLprim value llvm_ee_remove_module(LLVMModuleRef M, LLVMExecutionEngineRef EE) {
  LLVMModuleRef RemovedModule;
  char *Error;
  if (LLVMRemoveModule(EE, M, &RemovedModule, &Error))
    llvm_raise(*caml_named_value("Llvm_executionengine.Error"), Error);
  return Val_unit;
}

/* ExecutionEngine.t -> unit */
CAMLprim value llvm_ee_run_static_ctors(LLVMExecutionEngineRef EE) {
  LLVMRunStaticConstructors(EE);
  return Val_unit;
}

/* ExecutionEngine.t -> unit */
CAMLprim value llvm_ee_run_static_dtors(LLVMExecutionEngineRef EE) {
  LLVMRunStaticDestructors(EE);
  return Val_unit;
}

extern value llvm_alloc_data_layout(LLVMTargetDataRef TargetData);

/* ExecutionEngine.t -> Llvm_target.DataLayout.t */
CAMLprim value llvm_ee_get_data_layout(LLVMExecutionEngineRef EE) {
  value DataLayout;
  LLVMTargetDataRef OrigDataLayout;
  char* TargetDataCStr;

  OrigDataLayout = LLVMGetExecutionEngineTargetData(EE);
  TargetDataCStr = LLVMCopyStringRepOfTargetData(OrigDataLayout);
  DataLayout = llvm_alloc_data_layout(LLVMCreateTargetData(TargetDataCStr));
  LLVMDisposeMessage(TargetDataCStr);

  return DataLayout;
}

/* Llvm.llvalue -> int64 -> llexecutionengine -> unit */
CAMLprim value llvm_ee_add_global_mapping(LLVMValueRef Global, value Ptr,
                                          LLVMExecutionEngineRef EE) {
  LLVMAddGlobalMapping(EE, Global, (void*) (Int64_val(Ptr)));
  return Val_unit;
}

CAMLprim value llvm_ee_get_global_value_address(value Name,
						LLVMExecutionEngineRef EE) {
  return caml_copy_int64((int64_t) LLVMGetGlobalValueAddress(EE, String_val(Name)));
}

CAMLprim value llvm_ee_get_function_address(value Name,
					    LLVMExecutionEngineRef EE) {
  return caml_copy_int64((int64_t) LLVMGetFunctionAddress(EE, String_val(Name)));
}
