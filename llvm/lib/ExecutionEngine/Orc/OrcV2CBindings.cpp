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

namespace llvm {
namespace orc {

class OrcV2CAPIHelper {
public:
  using PoolEntry = SymbolStringPtr::PoolEntry;
  using PoolEntryPtr = SymbolStringPtr::PoolEntryPtr;

  static PoolEntryPtr releaseSymbolStringPtr(SymbolStringPtr S) {
    PoolEntryPtr Result = nullptr;
    std::swap(Result, S.S);
    return Result;
  }

  static PoolEntryPtr getRawPoolEntryPtr(const SymbolStringPtr &S) {
    return S.S;
  }

  static void releasePoolEntry(PoolEntryPtr P) {
    SymbolStringPtr S;
    S.S = P;
  }
};

} // end namespace orc
} // end namespace llvm

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ExecutionSession, LLVMOrcExecutionSessionRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(OrcV2CAPIHelper::PoolEntry,
                                   LLVMOrcSymbolStringPoolEntryRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(JITDylib, LLVMOrcJITDylibRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(JITDylib::DefinitionGenerator,
                                   LLVMOrcJITDylibDefinitionGeneratorRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ThreadSafeContext,
                                   LLVMOrcThreadSafeContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ThreadSafeModule, LLVMOrcThreadSafeModuleRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(JITTargetMachineBuilder,
                                   LLVMOrcJITTargetMachineBuilderRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LLJITBuilder, LLVMOrcLLJITBuilderRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LLJIT, LLVMOrcLLJITRef)

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef)

LLVMOrcSymbolStringPoolEntryRef
LLVMOrcExecutionSessionIntern(LLVMOrcExecutionSessionRef ES, const char *Name) {
  return wrap(
      OrcV2CAPIHelper::releaseSymbolStringPtr(unwrap(ES)->intern(Name)));
}

void LLVMOrcReleaseSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S) {
  OrcV2CAPIHelper::releasePoolEntry(unwrap(S));
}

void LLVMOrcDisposeJITDylibDefinitionGenerator(
    LLVMOrcJITDylibDefinitionGeneratorRef DG) {
  delete unwrap(DG);
}

void LLVMOrcJITDylibAddGenerator(LLVMOrcJITDylibRef JD,
                                 LLVMOrcJITDylibDefinitionGeneratorRef DG) {
  unwrap(JD)->addGenerator(
      std::unique_ptr<JITDylib::DefinitionGenerator>(unwrap(DG)));
}

LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
    LLVMOrcJITDylibDefinitionGeneratorRef *Result, char GlobalPrefix,
    LLVMOrcSymbolPredicate Filter, void *FilterCtx) {
  assert(Result && "Result can not be null");
  assert((Filter || !FilterCtx) &&
         "if Filter is null then FilterCtx must also be null");

  DynamicLibrarySearchGenerator::SymbolPredicate Pred;
  if (Filter)
    Pred = [=](const SymbolStringPtr &Name) -> bool {
      return Filter(wrap(OrcV2CAPIHelper::getRawPoolEntryPtr(Name)), FilterCtx);
    };

  auto ProcessSymsGenerator =
      DynamicLibrarySearchGenerator::GetForCurrentProcess(GlobalPrefix, Pred);

  if (!ProcessSymsGenerator) {
    *Result = 0;
    return wrap(ProcessSymsGenerator.takeError());
  }

  *Result = wrap(ProcessSymsGenerator->release());
  return LLVMErrorSuccess;
}

LLVMOrcThreadSafeContextRef LLVMOrcCreateNewThreadSafeContext(void) {
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

LLVMErrorRef LLVMOrcJITTargetMachineBuilderDetectHost(
    LLVMOrcJITTargetMachineBuilderRef *Result) {
  assert(Result && "Result can not be null");

  auto JTMB = JITTargetMachineBuilder::detectHost();
  if (!JTMB) {
    Result = 0;
    return wrap(JTMB.takeError());
  }

  *Result = wrap(new JITTargetMachineBuilder(std::move(*JTMB)));
  return LLVMErrorSuccess;
}

LLVMOrcJITTargetMachineBuilderRef
LLVMOrcJITTargetMachineBuilderFromTargetMachine(LLVMTargetMachineRef TM) {
  auto *TemplateTM = unwrap(TM);

  auto JTMB =
      std::make_unique<JITTargetMachineBuilder>(TemplateTM->getTargetTriple());

  (*JTMB)
      .setCPU(TemplateTM->getTargetCPU().str())
      .setRelocationModel(TemplateTM->getRelocationModel())
      .setCodeModel(TemplateTM->getCodeModel())
      .setCodeGenOptLevel(TemplateTM->getOptLevel())
      .setFeatures(TemplateTM->getTargetFeatureString())
      .setOptions(TemplateTM->Options);

  LLVMDisposeTargetMachine(TM);

  return wrap(JTMB.release());
}

void LLVMOrcDisposeJITTargetMachineBuilder(
    LLVMOrcJITTargetMachineBuilderRef JTMB) {
  delete unwrap(JTMB);
}

LLVMOrcLLJITBuilderRef LLVMOrcCreateLLJITBuilder(void) {
  return wrap(new LLJITBuilder());
}

void LLVMOrcDisposeLLJITBuilder(LLVMOrcLLJITBuilderRef Builder) {
  delete unwrap(Builder);
}

void LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(
    LLVMOrcLLJITBuilderRef Builder, LLVMOrcJITTargetMachineBuilderRef JTMB) {
  unwrap(Builder)->setJITTargetMachineBuilder(*unwrap(JTMB));
}

LLVMErrorRef LLVMOrcCreateLLJIT(LLVMOrcLLJITRef *Result,
                                LLVMOrcLLJITBuilderRef Builder) {
  assert(Result && "Result can not be null");

  if (!Builder)
    Builder = LLVMOrcCreateLLJITBuilder();

  auto J = unwrap(Builder)->create();
  LLVMOrcDisposeLLJITBuilder(Builder);

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

LLVMOrcExecutionSessionRef LLVMOrcLLJITGetExecutionSession(LLVMOrcLLJITRef J) {
  return wrap(&unwrap(J)->getExecutionSession());
}

LLVMOrcJITDylibRef LLVMOrcLLJITGetMainJITDylib(LLVMOrcLLJITRef J) {
  return wrap(&unwrap(J)->getMainJITDylib());
}

const char *LLVMOrcLLJITGetTripleString(LLVMOrcLLJITRef J) {
  return unwrap(J)->getTargetTriple().str().c_str();
}

char LLVMOrcLLJITGetGlobalPrefix(LLVMOrcLLJITRef J) {
  return unwrap(J)->getDataLayout().getGlobalPrefix();
}

LLVMOrcSymbolStringPoolEntryRef
LLVMOrcLLJITMangleAndIntern(LLVMOrcLLJITRef J, const char *UnmangledName) {
  return wrap(OrcV2CAPIHelper::releaseSymbolStringPtr(
      unwrap(J)->mangleAndIntern(UnmangledName)));
}

LLVMErrorRef LLVMOrcLLJITAddObjectFile(LLVMOrcLLJITRef J, LLVMOrcJITDylibRef JD,
                                       LLVMMemoryBufferRef ObjBuffer) {
  return wrap(unwrap(J)->addObjectFile(
      *unwrap(JD), std::unique_ptr<MemoryBuffer>(unwrap(ObjBuffer))));
}

LLVMErrorRef LLVMOrcLLJITAddLLVMIRModule(LLVMOrcLLJITRef J,
                                         LLVMOrcJITDylibRef JD,
                                         LLVMOrcThreadSafeModuleRef TSM) {
  return wrap(unwrap(J)->addIRModule(*unwrap(JD), std::move(*unwrap(TSM))));
}

LLVMErrorRef LLVMOrcLLJITLookup(LLVMOrcLLJITRef J,
                                LLVMOrcJITTargetAddress *Result,
                                const char *Name) {
  assert(Result && "Result can not be null");

  auto Sym = unwrap(J)->lookup(Name);
  if (!Sym) {
    *Result = 0;
    return wrap(Sym.takeError());
  }

  *Result = Sym->getAddress();
  return LLVMErrorSuccess;
}
