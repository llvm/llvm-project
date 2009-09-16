/*===-- executionengine_ocaml.c - LLVM Ocaml Glue ---------------*- C++ -*-===*\
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

#include "llvm-c/ExecutionEngine.h"
#include "llvm-c/Target.h"
#include "caml/alloc.h"
#include "caml/custom.h"
#include "caml/fail.h"
#include "caml/memory.h"
#include <string.h>
#include <assert.h>

/* Force the LLVM interpreter and JIT to be linked in. */
void llvm_initialize(void) {
  LLVMLinkInInterpreter();
  LLVMLinkInJIT();
}

/* unit -> bool */
CAMLprim value llvm_initialize_native_target(value Unit) {
  return Val_bool(LLVMInitializeNativeTarget());
}

/* Can't use the recommended caml_named_value mechanism for backwards
   compatibility reasons. This is largely equivalent. */
static value llvm_ee_error_exn;

CAMLprim value llvm_register_ee_exns(value Error) {
  llvm_ee_error_exn = Field(Error, 0);
  register_global_root(&llvm_ee_error_exn);
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


/*--... Operations on generic values .......................................--*/

#define Genericvalue_val(v)  (*(LLVMGenericValueRef *)(Data_custom_val(v)))

static void llvm_finalize_generic_value(value GenVal) {
  LLVMDisposeGenericValue(Genericvalue_val(GenVal));
}

static struct custom_operations generic_value_ops = {
  (char *) "LLVMGenericValue",
  llvm_finalize_generic_value,
  custom_compare_default,
  custom_hash_default,
  custom_serialize_default,
  custom_deserialize_default
};

static value alloc_generic_value(LLVMGenericValueRef Ref) {
  value Val = alloc_custom(&generic_value_ops, sizeof(LLVMGenericValueRef), 0, 1);
  Genericvalue_val(Val) = Ref;
  return Val;
}

/* Llvm.lltype -> float -> t */
CAMLprim value llvm_genericvalue_of_float(LLVMTypeRef Ty, value N) {
  CAMLparam1(N);
  CAMLreturn(alloc_generic_value(
    LLVMCreateGenericValueOfFloat(Ty, Double_val(N))));
}

/* 'a -> t */
CAMLprim value llvm_genericvalue_of_value(value V) {
  CAMLparam1(V);
  CAMLreturn(alloc_generic_value(LLVMCreateGenericValueOfPointer(Op_val(V))));
}

/* Llvm.lltype -> int -> t */
CAMLprim value llvm_genericvalue_of_int(LLVMTypeRef Ty, value Int) {
  return alloc_generic_value(LLVMCreateGenericValueOfInt(Ty, Int_val(Int), 1));
}

/* Llvm.lltype -> int32 -> t */
CAMLprim value llvm_genericvalue_of_int32(LLVMTypeRef Ty, value Int32) {
  CAMLparam1(Int32);
  CAMLreturn(alloc_generic_value(
    LLVMCreateGenericValueOfInt(Ty, Int32_val(Int32), 1)));
}

/* Llvm.lltype -> nativeint -> t */
CAMLprim value llvm_genericvalue_of_nativeint(LLVMTypeRef Ty, value NatInt) {
  CAMLparam1(NatInt);
  CAMLreturn(alloc_generic_value(
    LLVMCreateGenericValueOfInt(Ty, Nativeint_val(NatInt), 1)));
}

/* Llvm.lltype -> int64 -> t */
CAMLprim value llvm_genericvalue_of_int64(LLVMTypeRef Ty, value Int64) {
  CAMLparam1(Int64);
  CAMLreturn(alloc_generic_value(
    LLVMCreateGenericValueOfInt(Ty, Int64_val(Int64), 1)));
}

/* Llvm.lltype -> t -> float */
CAMLprim value llvm_genericvalue_as_float(LLVMTypeRef Ty, value GenVal) {
  CAMLparam1(GenVal);
  CAMLreturn(copy_double(
    LLVMGenericValueToFloat(Ty, Genericvalue_val(GenVal))));
}

/* t -> 'a */
CAMLprim value llvm_genericvalue_as_value(value GenVal) {
  return Val_op(LLVMGenericValueToPointer(Genericvalue_val(GenVal)));
}

/* t -> int */
CAMLprim value llvm_genericvalue_as_int(value GenVal) {
  assert(LLVMGenericValueIntWidth(Genericvalue_val(GenVal)) <= 8 * sizeof(value)
         && "Generic value too wide to treat as an int!");
  return Val_int(LLVMGenericValueToInt(Genericvalue_val(GenVal), 1));
}

/* t -> int32 */
CAMLprim value llvm_genericvalue_as_int32(value GenVal) {
  CAMLparam1(GenVal);
  assert(LLVMGenericValueIntWidth(Genericvalue_val(GenVal)) <= 32
         && "Generic value too wide to treat as an int32!");
  CAMLreturn(copy_int32(LLVMGenericValueToInt(Genericvalue_val(GenVal), 1)));
}

/* t -> int64 */
CAMLprim value llvm_genericvalue_as_int64(value GenVal) {
  CAMLparam1(GenVal);
  assert(LLVMGenericValueIntWidth(Genericvalue_val(GenVal)) <= 64
         && "Generic value too wide to treat as an int64!");
  CAMLreturn(copy_int64(LLVMGenericValueToInt(Genericvalue_val(GenVal), 1)));
}

/* t -> nativeint */
CAMLprim value llvm_genericvalue_as_nativeint(value GenVal) {
  CAMLparam1(GenVal);
  assert(LLVMGenericValueIntWidth(Genericvalue_val(GenVal)) <= 8 * sizeof(value)
         && "Generic value too wide to treat as a nativeint!");
  CAMLreturn(copy_nativeint(LLVMGenericValueToInt(Genericvalue_val(GenVal),1)));
}


/*--... Operations on execution engines ....................................--*/

/* llmoduleprovider -> ExecutionEngine.t */
CAMLprim LLVMExecutionEngineRef llvm_ee_create(LLVMModuleProviderRef MP) {
  LLVMExecutionEngineRef Interp;
  char *Error;
  if (LLVMCreateExecutionEngine(&Interp, MP, &Error))
    llvm_raise(llvm_ee_error_exn, Error);
  return Interp;
}

/* llmoduleprovider -> ExecutionEngine.t */
CAMLprim LLVMExecutionEngineRef
llvm_ee_create_interpreter(LLVMModuleProviderRef MP) {
  LLVMExecutionEngineRef Interp;
  char *Error;
  if (LLVMCreateInterpreter(&Interp, MP, &Error))
    llvm_raise(llvm_ee_error_exn, Error);
  return Interp;
}

/* llmoduleprovider -> ExecutionEngine.t */
CAMLprim LLVMExecutionEngineRef
llvm_ee_create_jit(LLVMModuleProviderRef MP) {
  LLVMExecutionEngineRef JIT;
  char *Error;
  if (LLVMCreateJITCompiler(&JIT, MP, 3, &Error))
    llvm_raise(llvm_ee_error_exn, Error);
  return JIT;
}

/* llmoduleprovider -> ExecutionEngine.t */
CAMLprim LLVMExecutionEngineRef
llvm_ee_create_fast_jit(LLVMModuleProviderRef MP) {
  LLVMExecutionEngineRef JIT;
  char *Error;
  if (LLVMCreateJITCompiler(&JIT, MP, 0, &Error))
    llvm_raise(llvm_ee_error_exn, Error);
  return JIT;
}

/* ExecutionEngine.t -> unit */
CAMLprim value llvm_ee_dispose(LLVMExecutionEngineRef EE) {
  LLVMDisposeExecutionEngine(EE);
  return Val_unit;
}

/* llmoduleprovider -> ExecutionEngine.t -> unit */
CAMLprim value llvm_ee_add_mp(LLVMModuleProviderRef MP,
                              LLVMExecutionEngineRef EE) {
  LLVMAddModuleProvider(EE, MP);
  return Val_unit;
}

/* llmoduleprovider -> ExecutionEngine.t -> llmodule */
CAMLprim LLVMModuleRef llvm_ee_remove_mp(LLVMModuleProviderRef MP,
                                         LLVMExecutionEngineRef EE) {
  LLVMModuleRef RemovedModule;
  char *Error;
  if (LLVMRemoveModuleProvider(EE, MP, &RemovedModule, &Error))
    llvm_raise(llvm_ee_error_exn, Error);
  return RemovedModule;
}

/* string -> ExecutionEngine.t -> llvalue option */
CAMLprim value llvm_ee_find_function(value Name, LLVMExecutionEngineRef EE) {
  CAMLparam1(Name);
  CAMLlocal1(Option);
  LLVMValueRef Found;
  if (LLVMFindFunction(EE, String_val(Name), &Found))
    CAMLreturn(Val_unit);
  Option = alloc(1, 1);
  Field(Option, 0) = Val_op(Found);
  CAMLreturn(Option);
}

/* llvalue -> GenericValue.t array -> ExecutionEngine.t -> GenericValue.t */
CAMLprim value llvm_ee_run_function(LLVMValueRef F, value Args,
                                    LLVMExecutionEngineRef EE) {
  unsigned NumArgs;
  LLVMGenericValueRef Result, *GVArgs;
  unsigned I;
  
  NumArgs = Wosize_val(Args);
  GVArgs = (LLVMGenericValueRef*) malloc(NumArgs * sizeof(LLVMGenericValueRef));
  for (I = 0; I != NumArgs; ++I)
    GVArgs[I] = Genericvalue_val(Field(Args, I));
  
  Result = LLVMRunFunction(EE, F, NumArgs, GVArgs);
  
  free(GVArgs);
  return alloc_generic_value(Result);
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

/* llvalue -> string array -> (string * string) array -> ExecutionEngine.t ->
   int */
CAMLprim value llvm_ee_run_function_as_main(LLVMValueRef F,
                                            value Args, value Env,
                                            LLVMExecutionEngineRef EE) {
  CAMLparam2(Args, Env);
  int I, NumArgs, NumEnv, EnvSize, Result;
  const char **CArgs, **CEnv;
  char *CEnvBuf, *Pos;
  
  NumArgs = Wosize_val(Args);
  NumEnv = Wosize_val(Env);
  
  /* Build the environment. */
  CArgs = (const char **) malloc(NumArgs * sizeof(char*));
  for (I = 0; I != NumArgs; ++I)
    CArgs[I] = String_val(Field(Args, I));
  
  /* Compute the size of the environment string buffer. */
  for (I = 0, EnvSize = 0; I != NumEnv; ++I) {
    EnvSize += strlen(String_val(Field(Field(Env, I), 0))) + 1;
    EnvSize += strlen(String_val(Field(Field(Env, I), 1))) + 1;
  }
  
  /* Build the environment. */
  CEnv = (const char **) malloc((NumEnv + 1) * sizeof(char*));
  CEnvBuf = (char*) malloc(EnvSize);
  Pos = CEnvBuf;
  for (I = 0; I != NumEnv; ++I) {
    char *Name  = String_val(Field(Field(Env, I), 0)),
         *Value = String_val(Field(Field(Env, I), 1));
    int NameLen  = strlen(Name),
        ValueLen = strlen(Value);
    
    CEnv[I] = Pos;
    memcpy(Pos, Name, NameLen);
    Pos += NameLen;
    *Pos++ = '=';
    memcpy(Pos, Value, ValueLen);
    Pos += ValueLen;
    *Pos++ = '\0';
  }
  CEnv[NumEnv] = NULL;
  
  Result = LLVMRunFunctionAsMain(EE, F, NumArgs, CArgs, CEnv);
  
  free(CArgs);
  free(CEnv);
  free(CEnvBuf);
  
  CAMLreturn(Val_int(Result));
}

/* llvalue -> ExecutionEngine.t -> unit */
CAMLprim value llvm_ee_free_machine_code(LLVMValueRef F,
                                         LLVMExecutionEngineRef EE) {
  LLVMFreeMachineCodeForFunction(EE, F);
  return Val_unit;
}

