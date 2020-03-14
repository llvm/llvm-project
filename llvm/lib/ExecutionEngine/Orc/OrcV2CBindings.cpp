//===--------------- OrcV2CBindings.cpp - C bindings OrcV2 APIs -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Orc.h"
#include "llvm-c/TargetMachine.h"

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

using namespace llvm;
using namespace llvm::orc;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ThreadSafeContext,
                                   LLVMOrcThreadSafeContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ThreadSafeModule, LLVMOrcThreadSafeModuleRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LLJIT, LLVMOrcLLJITRef)

LLVMOrcThreadSafeContextRef LLVMOrcCreateNewThreadSafeContext() {
  return wrap(new ThreadSafeContext(std::make_unique<LLVMContext>()));
}

LLVMContextRef
LLVMOrcThreadSafeContextGetContext(LLVMOrcThreadSafeContextRef TSCtx) {
  return wrap(unwrap(TSCtx)->getContext());
}

void LLVMOrcDisposeThreadSafeContext(LLVMOrcThreadSafeContextRef TSCtx) {
  delete unwrap(TSCtx);
}

LLVMOrcThreadSafeModuleRef
LLVMOrcCreateNewThreadSafeModule(LLVMModuleRef M,
                                 LLVMOrcThreadSafeContextRef TSCtx) {
  return wrap(
      new ThreadSafeModule(std::unique_ptr<Module>(unwrap(M)), *unwrap(TSCtx)));
}

void LLVMOrcDisposeThreadSafeModule(LLVMOrcThreadSafeModuleRef TSM) {
  delete unwrap(TSM);
}

LLVMErrorRef LLVMOrcCreateDefaultLLJIT(LLVMOrcLLJITRef *Result) {
  auto J = LLJITBuilder().create();

  if (!J) {
    Result = 0;
    return wrap(J.takeError());
  }

  *Result = wrap(J->release());
  return LLVMErrorSuccess;
}

LLVMErrorRef LLVMOrcDisposeLLJIT(LLVMOrcLLJITRef J) {
  delete unwrap(J);
  return LLVMErrorSuccess;
}

LLVMErrorRef LLVMOrcLLJITAddLLVMIRModule(LLVMOrcLLJITRef J,
                                         LLVMOrcThreadSafeModuleRef TSM) {
  return wrap(unwrap(J)->addIRModule(std::move(*unwrap(TSM))));
}

LLVMErrorRef LLVMOrcLLJITLookup(LLVMOrcLLJITRef J,
                                LLVMOrcJITTargetAddress *Result,
                                const char *Name) {
  auto Sym = unwrap(J)->lookup(Name);
  if (!Sym) {
    *Result = 0;
    return wrap(Sym.takeError());
  }

  *Result = Sym->getAddress();
  return LLVMErrorSuccess;
}
