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
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
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

  // Move from SymbolStringPtr to PoolEntryPtr (no change in ref count).
  static PoolEntryPtr moveFromSymbolStringPtr(SymbolStringPtr S) {
    PoolEntryPtr Result = nullptr;
    std::swap(Result, S.S);
    return Result;
  }

  // Move from a PoolEntryPtr to a SymbolStringPtr (no change in ref count).
  static SymbolStringPtr moveToSymbolStringPtr(PoolEntryPtr P) {
    SymbolStringPtr S;
    S.S = P;
    return S;
  }

  // Copy a pool entry to a SymbolStringPtr (increments ref count).
  static SymbolStringPtr copyToSymbolStringPtr(PoolEntryPtr P) {
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
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(MaterializationResponsibility,
                                   LLVMOrcMaterializationResponsibilityRef)
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
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(IRTransformLayer, LLVMOrcIRTransformLayerRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ObjectTransformLayer,
                                   LLVMOrcObjectTransformLayerRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DumpObjects, LLVMOrcDumpObjectsRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(IndirectStubsManager,
                                   LLVMOrcIndirectStubsManagerRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LazyCallThroughManager,
                                   LLVMOrcLazyCallThroughManagerRef)
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

namespace {

class OrcCAPIMaterializationUnit : public llvm::orc::MaterializationUnit {
public:
  OrcCAPIMaterializationUnit(
      std::string Name, SymbolFlagsMap InitialSymbolFlags,
      SymbolStringPtr InitSymbol, void *Ctx,
      LLVMOrcMaterializationUnitMaterializeFunction Materialize,
      LLVMOrcMaterializationUnitDiscardFunction Discard,
      LLVMOrcMaterializationUnitDestroyFunction Destroy)
      : llvm::orc::MaterializationUnit(std::move(InitialSymbolFlags),
                                       std::move(InitSymbol)),
        Name(std::move(Name)), Ctx(Ctx), Materialize(Materialize),
        Discard(Discard), Destroy(Destroy) {}

  ~OrcCAPIMaterializationUnit() {
    if (Ctx)
      Destroy(Ctx);
  }

  StringRef getName() const override { return Name; }

  void materialize(std::unique_ptr<MaterializationResponsibility> R) override {
    void *Tmp = Ctx;
    Ctx = nullptr;
    Materialize(Tmp, wrap(R.release()));
  }

private:
  void discard(const JITDylib &JD, const SymbolStringPtr &Name) override {
    Discard(Ctx, wrap(&JD), wrap(OrcV2CAPIHelper::getRawPoolEntryPtr(Name)));
  }

  std::string Name;
  void *Ctx = nullptr;
  LLVMOrcMaterializationUnitMaterializeFunction Materialize = nullptr;
  LLVMOrcMaterializationUnitDiscardFunction Discard = nullptr;
  LLVMOrcMaterializationUnitDestroyFunction Destroy = nullptr;
};

static JITSymbolFlags toJITSymbolFlags(LLVMJITSymbolFlags F) {

  JITSymbolFlags JSF;

  if (F.GenericFlags & LLVMJITSymbolGenericFlagsExported)
    JSF |= JITSymbolFlags::Exported;
  if (F.GenericFlags & LLVMJITSymbolGenericFlagsWeak)
    JSF |= JITSymbolFlags::Weak;
  if (F.GenericFlags & LLVMJITSymbolGenericFlagsCallable)
    JSF |= JITSymbolFlags::Callable;
  if (F.GenericFlags & LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly)
    JSF |= JITSymbolFlags::MaterializationSideEffectsOnly;

  JSF.getTargetFlags() = F.TargetFlags;

  return JSF;
}

static LLVMJITSymbolFlags fromJITSymbolFlags(JITSymbolFlags JSF) {
  LLVMJITSymbolFlags F = {0, 0};
  if (JSF & JITSymbolFlags::Exported)
    F.GenericFlags |= LLVMJITSymbolGenericFlagsExported;
  if (JSF & JITSymbolFlags::Weak)
    F.GenericFlags |= LLVMJITSymbolGenericFlagsWeak;
  if (JSF & JITSymbolFlags::Callable)
    F.GenericFlags |= LLVMJITSymbolGenericFlagsCallable;
  if (JSF & JITSymbolFlags::MaterializationSideEffectsOnly)
    F.GenericFlags |= LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly;

  F.TargetFlags = JSF.getTargetFlags();

  return F;
}

static SymbolMap toSymbolMap(LLVMOrcCSymbolMapPairs Syms, size_t NumPairs) {
  SymbolMap SM;
  for (size_t I = 0; I != NumPairs; ++I) {
    JITSymbolFlags Flags = toJITSymbolFlags(Syms[I].Sym.Flags);
    SM[OrcV2CAPIHelper::moveToSymbolStringPtr(unwrap(Syms[I].Name))] =
        JITEvaluatedSymbol(Syms[I].Sym.Address, Flags);
  }
  return SM;
}

static SymbolDependenceMap
toSymbolDependenceMap(LLVMOrcCDependenceMapPairs Pairs, size_t NumPairs) {
  SymbolDependenceMap SDM;
  for (size_t I = 0; I != NumPairs; ++I) {
    JITDylib *JD = unwrap(Pairs[I].JD);
    SymbolNameSet Names;

    for (size_t J = 0; J != Pairs[I].Names.Length; ++J) {
      auto Sym = Pairs[I].Names.Symbols[J];
      Names.insert(OrcV2CAPIHelper::moveToSymbolStringPtr(unwrap(Sym)));
    }
    SDM[JD] = Names;
  }
  return SDM;
}

} // end anonymous namespace

void LLVMOrcExecutionSessionSetErrorReporter(
    LLVMOrcExecutionSessionRef ES, LLVMOrcErrorReporterFunction ReportError,
    void *Ctx) {
  unwrap(ES)->setErrorReporter(
      [=](Error Err) { ReportError(Ctx, wrap(std::move(Err))); });
}

LLVMOrcSymbolStringPoolRef
LLVMOrcExecutionSessionGetSymbolStringPool(LLVMOrcExecutionSessionRef ES) {
  return wrap(
      unwrap(ES)->getExecutorProcessControl().getSymbolStringPool().get());
}

void LLVMOrcSymbolStringPoolClearDeadEntries(LLVMOrcSymbolStringPoolRef SSP) {
  unwrap(SSP)->clearDeadEntries();
}

LLVMOrcSymbolStringPoolEntryRef
LLVMOrcExecutionSessionIntern(LLVMOrcExecutionSessionRef ES, const char *Name) {
  return wrap(
      OrcV2CAPIHelper::moveFromSymbolStringPtr(unwrap(ES)->intern(Name)));
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

LLVMOrcMaterializationUnitRef LLVMOrcCreateCustomMaterializationUnit(
    const char *Name, void *Ctx, LLVMOrcCSymbolFlagsMapPairs Syms,
    size_t NumSyms, LLVMOrcSymbolStringPoolEntryRef InitSym,
    LLVMOrcMaterializationUnitMaterializeFunction Materialize,
    LLVMOrcMaterializationUnitDiscardFunction Discard,
    LLVMOrcMaterializationUnitDestroyFunction Destroy) {
  SymbolFlagsMap SFM;
  for (size_t I = 0; I != NumSyms; ++I)
    SFM[OrcV2CAPIHelper::moveToSymbolStringPtr(unwrap(Syms[I].Name))] =
        toJITSymbolFlags(Syms[I].Flags);

  auto IS = OrcV2CAPIHelper::moveToSymbolStringPtr(unwrap(InitSym));

  return wrap(new OrcCAPIMaterializationUnit(
      Name, std::move(SFM), std::move(IS), Ctx, Materialize, Discard, Destroy));
}

LLVMOrcMaterializationUnitRef
LLVMOrcAbsoluteSymbols(LLVMOrcCSymbolMapPairs Syms, size_t NumPairs) {
  SymbolMap SM = toSymbolMap(Syms, NumPairs);
  return wrap(absoluteSymbols(std::move(SM)).release());
}

LLVMOrcMaterializationUnitRef LLVMOrcLazyReexports(
    LLVMOrcLazyCallThroughManagerRef LCTM, LLVMOrcIndirectStubsManagerRef ISM,
    LLVMOrcJITDylibRef SourceJD, LLVMOrcCSymbolAliasMapPairs CallableAliases,
    size_t NumPairs) {

  SymbolAliasMap SAM;
  for (size_t I = 0; I != NumPairs; ++I) {
    auto pair = CallableAliases[I];
    JITSymbolFlags Flags = toJITSymbolFlags(pair.Entry.Flags);
    SymbolStringPtr Name =
        OrcV2CAPIHelper::moveToSymbolStringPtr(unwrap(pair.Entry.Name));
    SAM[OrcV2CAPIHelper::moveToSymbolStringPtr(unwrap(pair.Name))] =
        SymbolAliasMapEntry(Name, Flags);
  }

  return wrap(lazyReexports(*unwrap(LCTM), *unwrap(ISM), *unwrap(SourceJD),
                            std::move(SAM))
                  .release());
}

void LLVMOrcDisposeMaterializationResponsibility(
    LLVMOrcMaterializationResponsibilityRef MR) {
  std::unique_ptr<MaterializationResponsibility> TmpMR(unwrap(MR));
}

LLVMOrcJITDylibRef LLVMOrcMaterializationResponsibilityGetTargetDylib(
    LLVMOrcMaterializationResponsibilityRef MR) {
  return wrap(&unwrap(MR)->getTargetJITDylib());
}

LLVMOrcExecutionSessionRef
LLVMOrcMaterializationResponsibilityGetExecutionSession(
    LLVMOrcMaterializationResponsibilityRef MR) {
  return wrap(&unwrap(MR)->getExecutionSession());
}

LLVMOrcCSymbolFlagsMapPairs LLVMOrcMaterializationResponsibilityGetSymbols(
    LLVMOrcMaterializationResponsibilityRef MR, size_t *NumPairs) {

  auto Symbols = unwrap(MR)->getSymbols();
  LLVMOrcCSymbolFlagsMapPairs Result = static_cast<LLVMOrcCSymbolFlagsMapPairs>(
      safe_malloc(Symbols.size() * sizeof(LLVMOrcCSymbolFlagsMapPair)));
  size_t I = 0;
  for (auto const &pair : Symbols) {
    auto Name = wrap(OrcV2CAPIHelper::getRawPoolEntryPtr(pair.first));
    auto Flags = pair.second;
    Result[I] = {Name, fromJITSymbolFlags(Flags)};
    I++;
  }
  *NumPairs = Symbols.size();
  return Result;
}

void LLVMOrcDisposeCSymbolFlagsMap(LLVMOrcCSymbolFlagsMapPairs Pairs) {
  free(Pairs);
}

LLVMOrcSymbolStringPoolEntryRef
LLVMOrcMaterializationResponsibilityGetInitializerSymbol(
    LLVMOrcMaterializationResponsibilityRef MR) {
  auto Sym = unwrap(MR)->getInitializerSymbol();
  return wrap(OrcV2CAPIHelper::getRawPoolEntryPtr(Sym));
}

LLVMOrcSymbolStringPoolEntryRef *
LLVMOrcMaterializationResponsibilityGetRequestedSymbols(
    LLVMOrcMaterializationResponsibilityRef MR, size_t *NumSymbols) {

  auto Symbols = unwrap(MR)->getRequestedSymbols();
  LLVMOrcSymbolStringPoolEntryRef *Result =
      static_cast<LLVMOrcSymbolStringPoolEntryRef *>(safe_malloc(
          Symbols.size() * sizeof(LLVMOrcSymbolStringPoolEntryRef)));
  size_t I = 0;
  for (auto &Name : Symbols) {
    Result[I] = wrap(OrcV2CAPIHelper::getRawPoolEntryPtr(Name));
    I++;
  }
  *NumSymbols = Symbols.size();
  return Result;
}

void LLVMOrcDisposeSymbols(LLVMOrcSymbolStringPoolEntryRef *Symbols) {
  free(Symbols);
}

LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyResolved(
    LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolMapPairs Symbols,
    size_t NumPairs) {
  SymbolMap SM = toSymbolMap(Symbols, NumPairs);
  return wrap(unwrap(MR)->notifyResolved(std::move(SM)));
}

LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyEmitted(
    LLVMOrcMaterializationResponsibilityRef MR) {
  return wrap(unwrap(MR)->notifyEmitted());
}

LLVMErrorRef LLVMOrcMaterializationResponsibilityDefineMaterializing(
    LLVMOrcMaterializationResponsibilityRef MR,
    LLVMOrcCSymbolFlagsMapPairs Syms, size_t NumSyms) {
  SymbolFlagsMap SFM;
  for (size_t I = 0; I != NumSyms; ++I)
    SFM[OrcV2CAPIHelper::moveToSymbolStringPtr(unwrap(Syms[I].Name))] =
        toJITSymbolFlags(Syms[I].Flags);

  return wrap(unwrap(MR)->defineMaterializing(std::move(SFM)));
}

LLVMErrorRef LLVMOrcMaterializationResponsibilityReplace(
    LLVMOrcMaterializationResponsibilityRef MR,
    LLVMOrcMaterializationUnitRef MU) {
  std::unique_ptr<MaterializationUnit> TmpMU(unwrap(MU));
  return wrap(unwrap(MR)->replace(std::move(TmpMU)));
}

LLVMErrorRef LLVMOrcMaterializationResponsibilityDelegate(
    LLVMOrcMaterializationResponsibilityRef MR,
    LLVMOrcSymbolStringPoolEntryRef *Symbols, size_t NumSymbols,
    LLVMOrcMaterializationResponsibilityRef *Result) {
  SymbolNameSet Syms;
  for (size_t I = 0; I != NumSymbols; I++) {
    Syms.insert(OrcV2CAPIHelper::moveToSymbolStringPtr(unwrap(Symbols[I])));
  }
  auto OtherMR = unwrap(MR)->delegate(Syms);

  if (!OtherMR) {
    return wrap(OtherMR.takeError());
  }
  *Result = wrap(OtherMR->release());
  return LLVMErrorSuccess;
}

void LLVMOrcMaterializationResponsibilityAddDependencies(
    LLVMOrcMaterializationResponsibilityRef MR,
    LLVMOrcSymbolStringPoolEntryRef Name,
    LLVMOrcCDependenceMapPairs Dependencies, size_t NumPairs) {

  SymbolDependenceMap SDM = toSymbolDependenceMap(Dependencies, NumPairs);
  auto Sym = OrcV2CAPIHelper::moveToSymbolStringPtr(unwrap(Name));
  unwrap(MR)->addDependencies(Sym, SDM);
}

void LLVMOrcMaterializationResponsibilityAddDependenciesForAll(
    LLVMOrcMaterializationResponsibilityRef MR,
    LLVMOrcCDependenceMapPairs Dependencies, size_t NumPairs) {

  SymbolDependenceMap SDM = toSymbolDependenceMap(Dependencies, NumPairs);
  unwrap(MR)->addDependenciesForAll(SDM);
}

void LLVMOrcMaterializationResponsibilityFailMaterialization(
    LLVMOrcMaterializationResponsibilityRef MR) {
  unwrap(MR)->failMaterialization();
}

void LLVMOrcIRTransformLayerEmit(LLVMOrcIRTransformLayerRef IRLayer,
                                 LLVMOrcMaterializationResponsibilityRef MR,
                                 LLVMOrcThreadSafeModuleRef TSM) {
  std::unique_ptr<ThreadSafeModule> TmpTSM(unwrap(TSM));
  unwrap(IRLayer)->emit(
      std::unique_ptr<MaterializationResponsibility>(unwrap(MR)),
      std::move(*TmpTSM));
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

LLVMErrorRef
LLVMOrcThreadSafeModuleWithModuleDo(LLVMOrcThreadSafeModuleRef TSM,
                                    LLVMOrcGenericIRModuleOperationFunction F,
                                    void *Ctx) {
  return wrap(unwrap(TSM)->withModuleDo(
      [&](Module &M) { return unwrap(F(Ctx, wrap(&M))); }));
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

char *LLVMOrcJITTargetMachineBuilderGetTargetTriple(
    LLVMOrcJITTargetMachineBuilderRef JTMB) {
  auto Tmp = unwrap(JTMB)->getTargetTriple().str();
  char *TargetTriple = (char *)malloc(Tmp.size() + 1);
  strcpy(TargetTriple, Tmp.c_str());
  return TargetTriple;
}

void LLVMOrcJITTargetMachineBuilderSetTargetTriple(
    LLVMOrcJITTargetMachineBuilderRef JTMB, const char *TargetTriple) {
  unwrap(JTMB)->getTargetTriple() = Triple(TargetTriple);
}

LLVMErrorRef LLVMOrcObjectLayerAddObjectFile(LLVMOrcObjectLayerRef ObjLayer,
                                             LLVMOrcJITDylibRef JD,
                                             LLVMMemoryBufferRef ObjBuffer) {
  return wrap(unwrap(ObjLayer)->add(
      *unwrap(JD), std::unique_ptr<MemoryBuffer>(unwrap(ObjBuffer))));
}

LLVMErrorRef LLVMOrcLLJITAddObjectFileWithRT(LLVMOrcObjectLayerRef ObjLayer,
                                             LLVMOrcResourceTrackerRef RT,
                                             LLVMMemoryBufferRef ObjBuffer) {
  return wrap(
      unwrap(ObjLayer)->add(ResourceTrackerSP(unwrap(RT)),
                            std::unique_ptr<MemoryBuffer>(unwrap(ObjBuffer))));
}

void LLVMOrcObjectLayerEmit(LLVMOrcObjectLayerRef ObjLayer,
                            LLVMOrcMaterializationResponsibilityRef R,
                            LLVMMemoryBufferRef ObjBuffer) {
  unwrap(ObjLayer)->emit(
      std::unique_ptr<MaterializationResponsibility>(unwrap(R)),
      std::unique_ptr<MemoryBuffer>(unwrap(ObjBuffer)));
}

void LLVMOrcDisposeObjectLayer(LLVMOrcObjectLayerRef ObjLayer) {
  delete unwrap(ObjLayer);
}

void LLVMOrcIRTransformLayerSetTransform(
    LLVMOrcIRTransformLayerRef IRTransformLayer,
    LLVMOrcIRTransformLayerTransformFunction TransformFunction, void *Ctx) {
  unwrap(IRTransformLayer)
      ->setTransform(
          [=](ThreadSafeModule TSM,
              MaterializationResponsibility &R) -> Expected<ThreadSafeModule> {
            LLVMOrcThreadSafeModuleRef TSMRef =
                wrap(new ThreadSafeModule(std::move(TSM)));
            if (LLVMErrorRef Err = TransformFunction(Ctx, &TSMRef, wrap(&R))) {
              assert(!TSMRef && "TSMRef was not reset to null on error");
              return unwrap(Err);
            }
            return std::move(*unwrap(TSMRef));
          });
}

void LLVMOrcObjectTransformLayerSetTransform(
    LLVMOrcObjectTransformLayerRef ObjTransformLayer,
    LLVMOrcObjectTransformLayerTransformFunction TransformFunction, void *Ctx) {
  unwrap(ObjTransformLayer)
      ->setTransform([TransformFunction, Ctx](std::unique_ptr<MemoryBuffer> Obj)
                         -> Expected<std::unique_ptr<MemoryBuffer>> {
        LLVMMemoryBufferRef ObjBuffer = wrap(Obj.release());
        if (LLVMErrorRef Err = TransformFunction(Ctx, &ObjBuffer)) {
          assert(!ObjBuffer && "ObjBuffer was not reset to null on error");
          return unwrap(Err);
        }
        return std::unique_ptr<MemoryBuffer>(unwrap(ObjBuffer));
      });
}

LLVMOrcDumpObjectsRef LLVMOrcCreateDumpObjects(const char *DumpDir,
                                               const char *IdentifierOverride) {
  assert(DumpDir && "DumpDir should not be null");
  assert(IdentifierOverride && "IdentifierOverride should not be null");
  return wrap(new DumpObjects(DumpDir, IdentifierOverride));
}

void LLVMOrcDisposeDumpObjects(LLVMOrcDumpObjectsRef DumpObjects) {
  delete unwrap(DumpObjects);
}

LLVMErrorRef LLVMOrcDumpObjects_CallOperator(LLVMOrcDumpObjectsRef DumpObjects,
                                             LLVMMemoryBufferRef *ObjBuffer) {
  std::unique_ptr<MemoryBuffer> OB(unwrap(*ObjBuffer));
  if (auto Result = (*unwrap(DumpObjects))(std::move(OB))) {
    *ObjBuffer = wrap(Result->release());
    return LLVMErrorSuccess;
  } else {
    *ObjBuffer = nullptr;
    return wrap(Result.takeError());
  }
}

LLVMOrcLLJITBuilderRef LLVMOrcCreateLLJITBuilder(void) {
  return wrap(new LLJITBuilder());
}

void LLVMOrcDisposeLLJITBuilder(LLVMOrcLLJITBuilderRef Builder) {
  delete unwrap(Builder);
}

void LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(
    LLVMOrcLLJITBuilderRef Builder, LLVMOrcJITTargetMachineBuilderRef JTMB) {
  unwrap(Builder)->setJITTargetMachineBuilder(std::move(*unwrap(JTMB)));
  LLVMOrcDisposeJITTargetMachineBuilder(JTMB);
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
  return wrap(OrcV2CAPIHelper::moveFromSymbolStringPtr(
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

LLVMOrcObjectLayerRef LLVMOrcLLJITGetObjLinkingLayer(LLVMOrcLLJITRef J) {
  return wrap(&unwrap(J)->getObjLinkingLayer());
}

LLVMOrcObjectTransformLayerRef
LLVMOrcLLJITGetObjTransformLayer(LLVMOrcLLJITRef J) {
  return wrap(&unwrap(J)->getObjTransformLayer());
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

LLVMOrcIRTransformLayerRef LLVMOrcLLJITGetIRTransformLayer(LLVMOrcLLJITRef J) {
  return wrap(&unwrap(J)->getIRTransformLayer());
}

const char *LLVMOrcLLJITGetDataLayoutStr(LLVMOrcLLJITRef J) {
  return unwrap(J)->getDataLayout().getStringRepresentation().c_str();
}

LLVMOrcIndirectStubsManagerRef
LLVMOrcCreateLocalIndirectStubsManager(const char *TargetTriple) {
  auto builder = createLocalIndirectStubsManagerBuilder(Triple(TargetTriple));
  return wrap(builder().release());
}

void LLVMOrcDisposeIndirectStubsManager(LLVMOrcIndirectStubsManagerRef ISM) {
  std::unique_ptr<IndirectStubsManager> TmpISM(unwrap(ISM));
}

LLVMErrorRef LLVMOrcCreateLocalLazyCallThroughManager(
    const char *TargetTriple, LLVMOrcExecutionSessionRef ES,
    LLVMOrcJITTargetAddress ErrorHandlerAddr,
    LLVMOrcLazyCallThroughManagerRef *Result) {
  auto LCTM = createLocalLazyCallThroughManager(Triple(TargetTriple),
                                                *unwrap(ES), ErrorHandlerAddr);

  if (!LCTM)
    return wrap(LCTM.takeError());
  *Result = wrap(LCTM->release());
  return LLVMErrorSuccess;
}

void LLVMOrcDisposeLazyCallThroughManager(
    LLVMOrcLazyCallThroughManagerRef LCM) {
  std::unique_ptr<LazyCallThroughManager> TmpLCM(unwrap(LCM));
}
