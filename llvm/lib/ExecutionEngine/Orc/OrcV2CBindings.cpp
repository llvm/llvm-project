//===--------------- OrcV2CBindings.cpp - C bindings OrcV2 APIs -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/LLJIT.h"
#include "llvm-c/Orc.h"
#include "llvm-c/OrcEE.h"
#include "llvm-c/TargetMachine.h"

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

using namespace llvm;
using namespace llvm::orc;

namespace llvm {
namespace orc {

class InProgressLookupState;

class OrcV2CAPIHelper {
public:
  using PoolEntry = SymbolStringPtr::PoolEntry;
  using PoolEntryPtr = SymbolStringPtr::PoolEntryPtr;

  static PoolEntryPtr releaseSymbolStringPtr(SymbolStringPtr S) {
    PoolEntryPtr Result = nullptr;
    std::swap(Result, S.S);
    return Result;
  }

  static SymbolStringPtr retainSymbolStringPtr(PoolEntryPtr P) {
    return SymbolStringPtr(P);
  }

  static PoolEntryPtr getRawPoolEntryPtr(const SymbolStringPtr &S) {
    return S.S;
  }

  static void retainPoolEntry(PoolEntryPtr P) {
    SymbolStringPtr S(P);
    S.S = nullptr;
  }

  static void releasePoolEntry(PoolEntryPtr P) {
    SymbolStringPtr S;
    S.S = P;
  }

  static InProgressLookupState *extractLookupState(LookupState &LS) {
    return LS.IPLS.release();
  }

  static void resetLookupState(LookupState &LS, InProgressLookupState *IPLS) {
    return LS.reset(IPLS);
  }
};

} // namespace orc
} // namespace llvm

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ExecutionSession, LLVMOrcExecutionSessionRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(SymbolStringPool, LLVMOrcSymbolStringPoolRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(OrcV2CAPIHelper::PoolEntry,
                                   LLVMOrcSymbolStringPoolEntryRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(MaterializationUnit,
                                   LLVMOrcMaterializationUnitRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(JITDylib, LLVMOrcJITDylibRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ResourceTracker, LLVMOrcResourceTrackerRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DefinitionGenerator,
                                   LLVMOrcDefinitionGeneratorRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(InProgressLookupState, LLVMOrcLookupStateRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ThreadSafeContext,
                                   LLVMOrcThreadSafeContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ThreadSafeModule, LLVMOrcThreadSafeModuleRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(JITTargetMachineBuilder,
                                   LLVMOrcJITTargetMachineBuilderRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ObjectLayer, LLVMOrcObjectLayerRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LLJITBuilder, LLVMOrcLLJITBuilderRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LLJIT, LLVMOrcLLJITRef)

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef)

namespace llvm {
namespace orc {

class CAPIDefinitionGenerator final : public DefinitionGenerator {
public:
  CAPIDefinitionGenerator(
      void *Ctx,
      LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction TryToGenerate)
      : Ctx(Ctx), TryToGenerate(TryToGenerate) {}

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &LookupSet) override {

    // Take the lookup state.
    LLVMOrcLookupStateRef LSR = ::wrap(OrcV2CAPIHelper::extractLookupState(LS));

    // Translate the lookup kind.
    LLVMOrcLookupKind CLookupKind;
    switch (K) {
    case LookupKind::Static:
      CLookupKind = LLVMOrcLookupKindStatic;
      break;
    case LookupKind::DLSym:
      CLookupKind = LLVMOrcLookupKindDLSym;
      break;
    }

    // Translate the JITDylibSearchFlags.
    LLVMOrcJITDylibLookupFlags CJDLookupFlags;
    switch (JDLookupFlags) {
    case JITDylibLookupFlags::MatchExportedSymbolsOnly:
      CJDLookupFlags = LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly;
      break;
    case JITDylibLookupFlags::MatchAllSymbols:
      CJDLookupFlags = LLVMOrcJITDylibLookupFlagsMatchAllSymbols;
      break;
    }

    // Translate the lookup set.
    std::vector<LLVMOrcCLookupSetElement> CLookupSet;
    CLookupSet.reserve(LookupSet.size());
    for (auto &KV : LookupSet) {
      LLVMOrcSymbolLookupFlags SLF;
      LLVMOrcSymbolStringPoolEntryRef Name =
        ::wrap(OrcV2CAPIHelper::getRawPoolEntryPtr(KV.first));
      switch (KV.second) {
      case SymbolLookupFlags::RequiredSymbol:
        SLF = LLVMOrcSymbolLookupFlagsRequiredSymbol;
        break;
      case SymbolLookupFlags::WeaklyReferencedSymbol:
        SLF = LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol;
        break;
      }
      CLookupSet.push_back({Name, SLF});
    }

    // Run the C TryToGenerate function.
    auto Err = unwrap(TryToGenerate(::wrap(this), Ctx, &LSR, CLookupKind,
                                    ::wrap(&JD), CJDLookupFlags,
                                    CLookupSet.data(), CLookupSet.size()));

    // Restore the lookup state.
    OrcV2CAPIHelper::resetLookupState(LS, ::unwrap(LSR));

    return Err;
  }

private:
  void *Ctx;
  LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction TryToGenerate;
};

} // end namespace orc
} // end namespace llvm

void LLVMOrcExecutionSessionSetErrorReporter(
    LLVMOrcExecutionSessionRef ES, LLVMOrcErrorReporterFunction ReportError,
    void *Ctx) {
  unwrap(ES)->setErrorReporter(
      [=](Error Err) { ReportError(Ctx, wrap(std::move(Err))); });
}

LLVMOrcSymbolStringPoolRef
LLVMOrcExecutionSessionGetSymbolStringPool(LLVMOrcExecutionSessionRef ES) {
  return wrap(unwrap(ES)->getSymbolStringPool().get());
}

void LLVMOrcSymbolStringPoolClearDeadEntries(LLVMOrcSymbolStringPoolRef SSP) {
  unwrap(SSP)->clearDeadEntries();
}

LLVMOrcSymbolStringPoolEntryRef
LLVMOrcExecutionSessionIntern(LLVMOrcExecutionSessionRef ES, const char *Name) {
  return wrap(
      OrcV2CAPIHelper::releaseSymbolStringPtr(unwrap(ES)->intern(Name)));
}

void LLVMOrcRetainSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S) {
  OrcV2CAPIHelper::retainPoolEntry(unwrap(S));
}

void LLVMOrcReleaseSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S) {
  OrcV2CAPIHelper::releasePoolEntry(unwrap(S));
}

const char *LLVMOrcSymbolStringPoolEntryStr(LLVMOrcSymbolStringPoolEntryRef S) {
  return unwrap(S)->getKey().data();
}

LLVMOrcResourceTrackerRef
LLVMOrcJITDylibCreateResourceTracker(LLVMOrcJITDylibRef JD) {
  auto RT = unwrap(JD)->createResourceTracker();
  // Retain the pointer for the C API client.
  RT->Retain();
  return wrap(RT.get());
}

LLVMOrcResourceTrackerRef
LLVMOrcJITDylibGetDefaultResourceTracker(LLVMOrcJITDylibRef JD) {
  auto RT = unwrap(JD)->getDefaultResourceTracker();
  // Retain the pointer for the C API client.
  return wrap(RT.get());
}

void LLVMOrcReleaseResourceTracker(LLVMOrcResourceTrackerRef RT) {
  ResourceTrackerSP TmpRT(unwrap(RT));
  TmpRT->Release();
}

void LLVMOrcResourceTrackerTransferTo(LLVMOrcResourceTrackerRef SrcRT,
                                      LLVMOrcResourceTrackerRef DstRT) {
  ResourceTrackerSP TmpRT(unwrap(SrcRT));
  TmpRT->transferTo(*unwrap(DstRT));
}

LLVMErrorRef LLVMOrcResourceTrackerRemove(LLVMOrcResourceTrackerRef RT) {
  ResourceTrackerSP TmpRT(unwrap(RT));
  return wrap(TmpRT->remove());
}

void LLVMOrcDisposeDefinitionGenerator(LLVMOrcDefinitionGeneratorRef DG) {
  std::unique_ptr<DefinitionGenerator> TmpDG(unwrap(DG));
}

void LLVMOrcDisposeMaterializationUnit(LLVMOrcMaterializationUnitRef MU) {
  std::unique_ptr<MaterializationUnit> TmpMU(unwrap(MU));
}

LLVMOrcMaterializationUnitRef
LLVMOrcAbsoluteSymbols(LLVMOrcCSymbolMapPairs Syms, size_t NumPairs) {
  SymbolMap SM;
  for (size_t I = 0; I != NumPairs; ++I) {
    JITSymbolFlags Flags;

    if (Syms[I].Sym.Flags.GenericFlags & LLVMJITSymbolGenericFlagsExported)
      Flags |= JITSymbolFlags::Exported;
    if (Syms[I].Sym.Flags.GenericFlags & LLVMJITSymbolGenericFlagsWeak)
      Flags |= JITSymbolFlags::Weak;

    Flags.getTargetFlags() = Syms[I].Sym.Flags.TargetFlags;

    SM[OrcV2CAPIHelper::retainSymbolStringPtr(unwrap(Syms[I].Name))] =
        JITEvaluatedSymbol(Syms[I].Sym.Address, Flags);
  }

  return wrap(absoluteSymbols(std::move(SM)).release());
}

LLVMOrcJITDylibRef
LLVMOrcExecutionSessionCreateBareJITDylib(LLVMOrcExecutionSessionRef ES,
                                          const char *Name) {
  return wrap(&unwrap(ES)->createBareJITDylib(Name));
}

LLVMErrorRef
LLVMOrcExecutionSessionCreateJITDylib(LLVMOrcExecutionSessionRef ES,
                                      LLVMOrcJITDylibRef *Result,
                                      const char *Name) {
  auto JD = unwrap(ES)->createJITDylib(Name);
  if (!JD)
    return wrap(JD.takeError());
  *Result = wrap(&*JD);
  return LLVMErrorSuccess;
}

LLVMOrcJITDylibRef
LLVMOrcExecutionSessionGetJITDylibByName(LLVMOrcExecutionSessionRef ES,
                                         const char *Name) {
  return wrap(unwrap(ES)->getJITDylibByName(Name));
}

LLVMErrorRef LLVMOrcJITDylibDefine(LLVMOrcJITDylibRef JD,
                                   LLVMOrcMaterializationUnitRef MU) {
  std::unique_ptr<MaterializationUnit> TmpMU(unwrap(MU));

  if (auto Err = unwrap(JD)->define(TmpMU)) {
    TmpMU.release();
    return wrap(std::move(Err));
  }
  return LLVMErrorSuccess;
}

LLVMErrorRef LLVMOrcJITDylibClear(LLVMOrcJITDylibRef JD) {
  return wrap(unwrap(JD)->clear());
}

void LLVMOrcJITDylibAddGenerator(LLVMOrcJITDylibRef JD,
                                 LLVMOrcDefinitionGeneratorRef DG) {
  unwrap(JD)->addGenerator(std::unique_ptr<DefinitionGenerator>(unwrap(DG)));
}

LLVMOrcDefinitionGeneratorRef LLVMOrcCreateCustomCAPIDefinitionGenerator(
    LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction F, void *Ctx) {
  auto DG = std::make_unique<CAPIDefinitionGenerator>(Ctx, F);
  return wrap(DG.release());
}

LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
    LLVMOrcDefinitionGeneratorRef *Result, char GlobalPrefix,
    LLVMOrcSymbolPredicate Filter, void *FilterCtx) {
  assert(Result && "Result can not be null");
  assert((Filter || !FilterCtx) &&
         "if Filter is null then FilterCtx must also be null");

  DynamicLibrarySearchGenerator::SymbolPredicate Pred;
  if (Filter)
    Pred = [=](const SymbolStringPtr &Name) -> bool {
      return Filter(FilterCtx, wrap(OrcV2CAPIHelper::getRawPoolEntryPtr(Name)));
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
LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(LLVMTargetMachineRef TM) {
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

void lLVMOrcDisposeObjectLayer(LLVMOrcObjectLayerRef ObjLayer) {
  delete unwrap(ObjLayer);
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

void LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(
    LLVMOrcLLJITBuilderRef Builder,
    LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction F, void *Ctx) {
  unwrap(Builder)->setObjectLinkingLayerCreator(
      [=](ExecutionSession &ES, const Triple &TT) {
        auto TTStr = TT.str();
        return std::unique_ptr<ObjectLayer>(
            unwrap(F(Ctx, wrap(&ES), TTStr.c_str())));
      });
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

LLVMErrorRef LLVMOrcLLJITAddObjectFileWithRT(LLVMOrcLLJITRef J,
                                             LLVMOrcResourceTrackerRef RT,
                                             LLVMMemoryBufferRef ObjBuffer) {
  return wrap(unwrap(J)->addObjectFile(
      ResourceTrackerSP(unwrap(RT)),
      std::unique_ptr<MemoryBuffer>(unwrap(ObjBuffer))));
}

LLVMErrorRef LLVMOrcLLJITAddLLVMIRModule(LLVMOrcLLJITRef J,
                                         LLVMOrcJITDylibRef JD,
                                         LLVMOrcThreadSafeModuleRef TSM) {
  std::unique_ptr<ThreadSafeModule> TmpTSM(unwrap(TSM));
  return wrap(unwrap(J)->addIRModule(*unwrap(JD), std::move(*TmpTSM)));
}

LLVMErrorRef LLVMOrcLLJITAddLLVMIRModuleWithRT(LLVMOrcLLJITRef J,
                                               LLVMOrcResourceTrackerRef RT,
                                               LLVMOrcThreadSafeModuleRef TSM) {
  std::unique_ptr<ThreadSafeModule> TmpTSM(unwrap(TSM));
  return wrap(unwrap(J)->addIRModule(ResourceTrackerSP(unwrap(RT)),
                                     std::move(*TmpTSM)));
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

LLVMOrcObjectLayerRef
LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager(
    LLVMOrcExecutionSessionRef ES) {
  assert(ES && "ES must not be null");
  return wrap(new RTDyldObjectLinkingLayer(
      *unwrap(ES), [] { return std::make_unique<SectionMemoryManager>(); }));
}

void LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(
    LLVMOrcObjectLayerRef RTDyldObjLinkingLayer,
    LLVMJITEventListenerRef Listener) {
  assert(RTDyldObjLinkingLayer && "RTDyldObjLinkingLayer must not be null");
  assert(Listener && "Listener must not be null");
  reinterpret_cast<RTDyldObjectLinkingLayer *>(unwrap(RTDyldObjLinkingLayer))
      ->registerJITEventListener(*unwrap(Listener));
}
