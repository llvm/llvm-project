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
#include <cstring>

using namespace llvm;

/*===-- Operations on generic values --------------------------------------===*/

LLVMGenericValueRef LLVMCreateGenericValueOfInt(LLVMTypeRef Ty,
                                                unsigned long long N,
                                                int IsSigned) {
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
    assert(0 && "LLVMGenericValueToFloat supports only float and double.");
    break;
  }
  return wrap(GenVal);
}

unsigned LLVMGenericValueIntWidth(LLVMGenericValueRef GenValRef) {
  return unwrap(GenValRef)->IntVal.getBitWidth();
}

unsigned long long LLVMGenericValueToInt(LLVMGenericValueRef GenValRef,
                                         int IsSigned) {
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
    assert(0 && "LLVMGenericValueToFloat supports only float and double.");
    break;
  }
  return 0; // Not reached
}

void LLVMDisposeGenericValue(LLVMGenericValueRef GenVal) {
  delete unwrap(GenVal);
}

/*===-- Operations on execution engines -----------------------------------===*/

int LLVMCreateExecutionEngine(LLVMExecutionEngineRef *OutEE,
                              LLVMModuleProviderRef MP,
                              char **OutError) {
  std::string Error;
  if (ExecutionEngine *EE = ExecutionEngine::create(unwrap(MP), false, &Error)){
    *OutEE = wrap(EE);
    return 0;
  }
  *OutError = strdup(Error.c_str());
  return 1;
}

int LLVMCreateInterpreter(LLVMExecutionEngineRef *OutInterp,
                          LLVMModuleProviderRef MP,
                          char **OutError) {
  std::string Error;
  if (ExecutionEngine *Interp =
      ExecutionEngine::create(unwrap(MP), true, &Error)) {
    *OutInterp = wrap(Interp);
    return 0;
  }
  *OutError = strdup(Error.c_str());
  return 1;
}

int LLVMCreateJITCompiler(LLVMExecutionEngineRef *OutJIT,
                          LLVMModuleProviderRef MP,
                          int Fast,
                          char **OutError) {
  std::string Error;
  if (ExecutionEngine *JIT = ExecutionEngine::createJIT(unwrap(MP), &Error, 0,
                                                        Fast != 0)) {
    *OutJIT = wrap(JIT);
    return 0;
  }
  *OutError = strdup(Error.c_str());
  return 1;
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

void LLVMAddModuleProvider(LLVMExecutionEngineRef EE, LLVMModuleProviderRef MP){
  unwrap(EE)->addModuleProvider(unwrap(MP));
}

int LLVMRemoveModuleProvider(LLVMExecutionEngineRef EE,
                             LLVMModuleProviderRef MP,
                             LLVMModuleRef *OutMod, char **OutError) {
  std::string Error;
  if (Module *Gone = unwrap(EE)->removeModuleProvider(unwrap(MP), &Error)) {
    *OutMod = wrap(Gone);
    return 0;
  }
  if (OutError)
    *OutError = strdup(Error.c_str());
  return 1;
}

int LLVMFindFunction(LLVMExecutionEngineRef EE, const char *Name,
                     LLVMValueRef *OutFn) {
  if (Function *F = unwrap(EE)->FindFunctionNamed(Name)) {
    *OutFn = wrap(F);
    return 0;
  }
  return 1;
}

LLVMTargetDataRef LLVMGetExecutionEngineTargetData(LLVMExecutionEngineRef EE) {
  return wrap(unwrap(EE)->getTargetData());
}

void LLVMAddGlobalMapping(LLVMExecutionEngineRef EE, LLVMValueRef Global,
                          void* Addr) {
  unwrap(EE)->addGlobalMapping(unwrap<GlobalValue>(Global), Addr);
}

void *LLVMGetPointerToGlobal(LLVMExecutionEngineRef EE, LLVMValueRef Global) {
  return unwrap(EE)->getPointerToGlobal(unwrap<GlobalValue>(Global));
}
