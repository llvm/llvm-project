//===-- ExecutionEngineBindings.cpp - C bindings for EEs ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the C bindings for the ExecutionEngine library.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "llvm-c/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstring>

using namespace llvm;

/*===-- Operations on generic values --------------------------------------===*/

LLVMGenericValueRef LLVMCreateGenericValueOfInt(LLVMTypeRef Ty,
                                                unsigned long long N,
                                                LLVMBool IsSigned) {
  GenericValue *GenVal = new GenericValue();
  GenVal->IntVal = APInt(unwrap<IntegerType>(Ty)->getBitWidth(), N, IsSigned);
  return wrap(GenVal);
}

LLVMGenericValueRef LLVMCreateGenericValueOfPointer(void *P) {
  GenericValue *GenVal = new GenericValue();
  GenVal->PointerVal = P;
  return wrap(GenVal);
}

LLVMGenericValueRef LLVMCreateGenericValueOfFloat(LLVMTypeRef TyRef, double N) {
  GenericValue *GenVal = new GenericValue();
  switch (unwrap(TyRef)->getTypeID()) {
  case Type::FloatTyID:
    GenVal->FloatVal = N;
    break;
  case Type::DoubleTyID:
    GenVal->DoubleVal = N;
    break;
  default:
    llvm_unreachable("LLVMGenericValueToFloat supports only float and double.");
  }
  return wrap(GenVal);
}

unsigned LLVMGenericValueIntWidth(LLVMGenericValueRef GenValRef) {
  return unwrap(GenValRef)->IntVal.getBitWidth();
}

unsigned long long LLVMGenericValueToInt(LLVMGenericValueRef GenValRef,
                                         LLVMBool IsSigned) {
  GenericValue *GenVal = unwrap(GenValRef);
  if (IsSigned)
    return GenVal->IntVal.getSExtValue();
  else
    return GenVal->IntVal.getZExtValue();
}

void *LLVMGenericValueToPointer(LLVMGenericValueRef GenVal) {
  return unwrap(GenVal)->PointerVal;
}

double LLVMGenericValueToFloat(LLVMTypeRef TyRef, LLVMGenericValueRef GenVal) {
  switch (unwrap(TyRef)->getTypeID()) {
  case Type::FloatTyID:
    return unwrap(GenVal)->FloatVal;
  case Type::DoubleTyID:
    return unwrap(GenVal)->DoubleVal;
  default:
    llvm_unreachable("LLVMGenericValueToFloat supports only float and double.");
  }
}

void LLVMDisposeGenericValue(LLVMGenericValueRef GenVal) {
  delete unwrap(GenVal);
}

/*===-- Operations on execution engines -----------------------------------===*/

LLVMBool LLVMCreateExecutionEngineForModule(LLVMExecutionEngineRef *OutEE,
                                            LLVMModuleRef M,
                                            char **OutError) {
  std::string Error;
  EngineBuilder builder(unwrap(M));
  builder.setEngineKind(EngineKind::Either)
         .setErrorStr(&Error);
  if (ExecutionEngine *EE = builder.create()){
    *OutEE = wrap(EE);
    return 0;
  }
  *OutError = strdup(Error.c_str());
  return 1;
}

LLVMBool LLVMCreateInterpreterForModule(LLVMExecutionEngineRef *OutInterp,
                                        LLVMModuleRef M,
                                        char **OutError) {
  std::string Error;
  EngineBuilder builder(unwrap(M));
  builder.setEngineKind(EngineKind::Interpreter)
         .setErrorStr(&Error);
  if (ExecutionEngine *Interp = builder.create()) {
    *OutInterp = wrap(Interp);
    return 0;
  }
  *OutError = strdup(Error.c_str());
  return 1;
}

LLVMBool LLVMCreateJITCompilerForModule(LLVMExecutionEngineRef *OutJIT,
                                        LLVMModuleRef M,
                                        unsigned OptLevel,
                                        char **OutError) {
  std::string Error;
  EngineBuilder builder(unwrap(M));
  builder.setEngineKind(EngineKind::JIT)
         .setErrorStr(&Error)
         .setOptLevel((CodeGenOpt::Level)OptLevel);
  if (ExecutionEngine *JIT = builder.create()) {
    *OutJIT = wrap(JIT);
    return 0;
  }
  *OutError = strdup(Error.c_str());
  return 1;
}

LLVMBool LLVMCreateExecutionEngine(LLVMExecutionEngineRef *OutEE,
                                   LLVMModuleProviderRef MP,
                                   char **OutError) {
  /* The module provider is now actually a module. */
  return LLVMCreateExecutionEngineForModule(OutEE,
                                            reinterpret_cast<LLVMModuleRef>(MP),
                                            OutError);
}

LLVMBool LLVMCreateInterpreter(LLVMExecutionEngineRef *OutInterp,
                               LLVMModuleProviderRef MP,
                               char **OutError) {
  /* The module provider is now actually a module. */
  return LLVMCreateInterpreterForModule(OutInterp,
                                        reinterpret_cast<LLVMModuleRef>(MP),
                                        OutError);
}

LLVMBool LLVMCreateJITCompiler(LLVMExecutionEngineRef *OutJIT,
                               LLVMModuleProviderRef MP,
                               unsigned OptLevel,
                               char **OutError) {
  /* The module provider is now actually a module. */
  return LLVMCreateJITCompilerForModule(OutJIT,
                                        reinterpret_cast<LLVMModuleRef>(MP),
                                        OptLevel, OutError);
}


void LLVMDisposeExecutionEngine(LLVMExecutionEngineRef EE) {
  delete unwrap(EE);
}

void LLVMRunStaticConstructors(LLVMExecutionEngineRef EE) {
  unwrap(EE)->runStaticConstructorsDestructors(false);
}

void LLVMRunStaticDestructors(LLVMExecutionEngineRef EE) {
  unwrap(EE)->runStaticConstructorsDestructors(true);
}

int LLVMRunFunctionAsMain(LLVMExecutionEngineRef EE, LLVMValueRef F,
                          unsigned ArgC, const char * const *ArgV,
                          const char * const *EnvP) {
  std::vector<std::string> ArgVec;
  for (unsigned I = 0; I != ArgC; ++I)
    ArgVec.push_back(ArgV[I]);
  
  return unwrap(EE)->runFunctionAsMain(unwrap<Function>(F), ArgVec, EnvP);
}

LLVMGenericValueRef LLVMRunFunction(LLVMExecutionEngineRef EE, LLVMValueRef F,
                                    unsigned NumArgs,
                                    LLVMGenericValueRef *Args) {
  std::vector<GenericValue> ArgVec;
  ArgVec.reserve(NumArgs);
  for (unsigned I = 0; I != NumArgs; ++I)
    ArgVec.push_back(*unwrap(Args[I]));
  
  GenericValue *Result = new GenericValue();
  *Result = unwrap(EE)->runFunction(unwrap<Function>(F), ArgVec);
  return wrap(Result);
}

void LLVMFreeMachineCodeForFunction(LLVMExecutionEngineRef EE, LLVMValueRef F) {
  unwrap(EE)->freeMachineCodeForFunction(unwrap<Function>(F));
}

void LLVMAddModule(LLVMExecutionEngineRef EE, LLVMModuleRef M){
  unwrap(EE)->addModule(unwrap(M));
}

void LLVMAddModuleProvider(LLVMExecutionEngineRef EE, LLVMModuleProviderRef MP){
  /* The module provider is now actually a module. */
  LLVMAddModule(EE, reinterpret_cast<LLVMModuleRef>(MP));
}

LLVMBool LLVMRemoveModule(LLVMExecutionEngineRef EE, LLVMModuleRef M,
                          LLVMModuleRef *OutMod, char **OutError) {
  Module *Mod = unwrap(M);
  unwrap(EE)->removeModule(Mod);
  *OutMod = wrap(Mod);
  return 0;
}

LLVMBool LLVMRemoveModuleProvider(LLVMExecutionEngineRef EE,
                                  LLVMModuleProviderRef MP,
                                  LLVMModuleRef *OutMod, char **OutError) {
  /* The module provider is now actually a module. */
  return LLVMRemoveModule(EE, reinterpret_cast<LLVMModuleRef>(MP), OutMod,
                          OutError);
}

LLVMBool LLVMFindFunction(LLVMExecutionEngineRef EE, const char *Name,
                          LLVMValueRef *OutFn) {
  if (Function *F = unwrap(EE)->FindFunctionNamed(Name)) {
    *OutFn = wrap(F);
    return 0;
  }
  return 1;
}

void *LLVMRecompileAndRelinkFunction(LLVMExecutionEngineRef EE, LLVMValueRef Fn) {
  return unwrap(EE)->recompileAndRelinkFunction(unwrap<Function>(Fn));
}

LLVMTargetDataRef LLVMGetExecutionEngineTargetData(LLVMExecutionEngineRef EE) {
  return wrap(unwrap(EE)->getDataLayout());
}

void LLVMAddGlobalMapping(LLVMExecutionEngineRef EE, LLVMValueRef Global,
                          void* Addr) {
  unwrap(EE)->addGlobalMapping(unwrap<GlobalValue>(Global), Addr);
}

void *LLVMGetPointerToGlobal(LLVMExecutionEngineRef EE, LLVMValueRef Global) {
  return unwrap(EE)->getPointerToGlobal(unwrap<GlobalValue>(Global));
}
