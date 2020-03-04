//===---------- LazyReexports.cpp - Utilities for lazy reexports ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LazyReexports.h"

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/OrcABISupport.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

LazyCallThroughManager::LazyCallThroughManager(
    ExecutionSession &ES, JITTargetAddress ErrorHandlerAddr,
    std::unique_ptr<TrampolinePool> TP)
    : ES(ES), ErrorHandlerAddr(ErrorHandlerAddr), TP(std::move(TP)) {}

Expected<JITTargetAddress> LazyCallThroughManager::getCallThroughTrampoline(
    JITDylib &SourceJD, SymbolStringPtr SymbolName,
    NotifyResolvedFunction NotifyResolved) {
  std::lock_guard<std::mutex> Lock(LCTMMutex);
  auto Trampoline = TP->getTrampoline();

  if (!Trampoline)
    return Trampoline.takeError();

  Reexports[*Trampoline] = ReexportsEntry{&SourceJD, std::move(SymbolName)};
  Notifiers[*Trampoline] = std::move(NotifyResolved);
  return *Trampoline;
}

Expected<LazyCallThroughManager::ReexportsEntry>
LazyCallThroughManager::findReexport(JITTargetAddress TrampolineAddr) {
  std::lock_guard<std::mutex> Lock(LCTMMutex);
  auto I = Reexports.find(TrampolineAddr);
  if (I == Reexports.end())
    return createStringError(inconvertibleErrorCode(),
                             "Missing reexport for trampoline address %p",
                             TrampolineAddr);
  return I->second;
}

Expected<JITTargetAddress>
LazyCallThroughManager::resolveSymbol(const ReexportsEntry &RE) {
  auto LookupResult =
      ES.lookup(makeJITDylibSearchOrder(RE.SourceJD,
                                        JITDylibLookupFlags::MatchAllSymbols),
                RE.SymbolName, SymbolState::Ready);

  if (!LookupResult)
    return LookupResult.takeError();

  return LookupResult->getAddress();
}

Error LazyCallThroughManager::notifyResolved(JITTargetAddress TrampolineAddr,
                                             JITTargetAddress ResolvedAddr) {
  NotifyResolvedFunction NotifyResolved;
  {
    std::lock_guard<std::mutex> Lock(LCTMMutex);
    auto I = Notifiers.find(TrampolineAddr);
    if (I != Notifiers.end()) {
      NotifyResolved = std::move(I->second);
      Notifiers.erase(I);
    }
  }

  return NotifyResolved ? NotifyResolved(ResolvedAddr) : Error::success();
}

Expected<std::unique_ptr<LazyCallThroughManager>>
createLocalLazyCallThroughManager(const Triple &T, ExecutionSession &ES,
                                  JITTargetAddress ErrorHandlerAddr) {
  switch (T.getArch()) {
  default:
    return make_error<StringError>(
        std::string("No callback manager available for ") + T.str(),
        inconvertibleErrorCode());

  case Triple::aarch64:
  case Triple::aarch64_32:
    return LocalLazyCallThroughManager::Create<OrcAArch64>(ES,
                                                           ErrorHandlerAddr);

  case Triple::x86:
    return LocalLazyCallThroughManager::Create<OrcI386>(ES, ErrorHandlerAddr);

  case Triple::mips:
    return LocalLazyCallThroughManager::Create<OrcMips32Be>(ES,
                                                            ErrorHandlerAddr);

  case Triple::mipsel:
    return LocalLazyCallThroughManager::Create<OrcMips32Le>(ES,
                                                            ErrorHandlerAddr);

  case Triple::mips64:
  case Triple::mips64el:
    return LocalLazyCallThroughManager::Create<OrcMips64>(ES, ErrorHandlerAddr);

  case Triple::x86_64:
    if (T.getOS() == Triple::OSType::Win32)
      return LocalLazyCallThroughManager::Create<OrcX86_64_Win32>(
          ES, ErrorHandlerAddr);
    else
      return LocalLazyCallThroughManager::Create<OrcX86_64_SysV>(
          ES, ErrorHandlerAddr);
  }
}

LazyReexportsMaterializationUnit::LazyReexportsMaterializationUnit(
    LazyCallThroughManager &LCTManager, IndirectStubsManager &ISManager,
    JITDylib &SourceJD, SymbolAliasMap CallableAliases, ImplSymbolMap *SrcJDLoc,
    VModuleKey K)
    : MaterializationUnit(extractFlags(CallableAliases), nullptr, std::move(K)),
      LCTManager(LCTManager), ISManager(ISManager), SourceJD(SourceJD),
      CallableAliases(std::move(CallableAliases)), AliaseeTable(SrcJDLoc) {}

StringRef LazyReexportsMaterializationUnit::getName() const {
  return "<Lazy Reexports>";
}

void LazyReexportsMaterializationUnit::materialize(
    MaterializationResponsibility R) {
  auto RequestedSymbols = R.getRequestedSymbols();

  SymbolAliasMap RequestedAliases;
  for (auto &RequestedSymbol : RequestedSymbols) {
    auto I = CallableAliases.find(RequestedSymbol);
    assert(I != CallableAliases.end() && "Symbol not found in alias map?");
    RequestedAliases[I->first] = std::move(I->second);
    CallableAliases.erase(I);
  }

  if (!CallableAliases.empty())
    R.replace(lazyReexports(LCTManager, ISManager, SourceJD,
                            std::move(CallableAliases), AliaseeTable));

  IndirectStubsManager::StubInitsMap StubInits;
  for (auto &Alias : RequestedAliases) {

    auto CallThroughTrampoline = LCTManager.getCallThroughTrampoline(
        SourceJD, Alias.second.Aliasee,
        [&ISManager = this->ISManager,
         StubSym = Alias.first](JITTargetAddress ResolvedAddr) -> Error {
          return ISManager.updatePointer(*StubSym, ResolvedAddr);
        });

    if (!CallThroughTrampoline) {
      SourceJD.getExecutionSession().reportError(
          CallThroughTrampoline.takeError());
      R.failMaterialization();
      return;
    }

    StubInits[*Alias.first] =
        std::make_pair(*CallThroughTrampoline, Alias.second.AliasFlags);
  }

  if (AliaseeTable != nullptr && !RequestedAliases.empty())
    AliaseeTable->trackImpls(RequestedAliases, &SourceJD);

  if (auto Err = ISManager.createStubs(StubInits)) {
    SourceJD.getExecutionSession().reportError(std::move(Err));
    R.failMaterialization();
    return;
  }

  SymbolMap Stubs;
  for (auto &Alias : RequestedAliases)
    Stubs[Alias.first] = ISManager.findStub(*Alias.first, false);

  // No registered dependencies, so these calls cannot fail.
  cantFail(R.notifyResolved(Stubs));
  cantFail(R.notifyEmitted());
}

void LazyReexportsMaterializationUnit::discard(const JITDylib &JD,
                                               const SymbolStringPtr &Name) {
  assert(CallableAliases.count(Name) &&
         "Symbol not covered by this MaterializationUnit");
  CallableAliases.erase(Name);
}

SymbolFlagsMap
LazyReexportsMaterializationUnit::extractFlags(const SymbolAliasMap &Aliases) {
  SymbolFlagsMap SymbolFlags;
  for (auto &KV : Aliases) {
    assert(KV.second.AliasFlags.isCallable() &&
           "Lazy re-exports must be callable symbols");
    SymbolFlags[KV.first] = KV.second.AliasFlags;
  }
  return SymbolFlags;
}

} // End namespace orc.
} // End namespace llvm.
