//===----- Core.cpp - Core ORC APIs (MaterializationUnit, VSO, etc.) ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"
#include "llvm/Support/Format.h"

#if LLVM_ENABLE_THREADS
#include <future>
#endif

namespace llvm {
namespace orc {

char FailedToMaterialize::ID = 0;
char FailedToResolve::ID = 0;
char FailedToFinalize::ID = 0;

void MaterializationUnit::anchor() {}
void SymbolResolver::anchor() {}

raw_ostream &operator<<(raw_ostream &OS, const JITSymbolFlags &Flags) {
  if (Flags.isWeak())
    OS << 'W';
  else if (Flags.isCommon())
    OS << 'C';
  else
    OS << 'S';

  if (Flags.isExported())
    OS << 'E';
  else
    OS << 'H';

  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const JITEvaluatedSymbol &Sym) {
  OS << format("0x%016x", Sym.getAddress()) << " " << Sym.getFlags();
  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolMap::value_type &KV) {
  OS << "\"" << *KV.first << "\": " << KV.second;
  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolNameSet &Symbols) {
  OS << "{";
  if (!Symbols.empty()) {
    OS << " \"" << **Symbols.begin() << "\"";
    for (auto &Sym : make_range(std::next(Symbols.begin()), Symbols.end()))
      OS << ", \"" << *Sym << "\"";
  }
  OS << " }";
  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolMap &Symbols) {
  OS << "{";
  if (!Symbols.empty()) {
    OS << " {" << *Symbols.begin() << "}";
    for (auto &Sym : make_range(std::next(Symbols.begin()), Symbols.end()))
      OS << ", {" << Sym << "}";
  }
  OS << " }";
  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolFlagsMap &SymbolFlags) {
  OS << "{";
  if (SymbolFlags.empty()) {
    OS << " {\"" << *SymbolFlags.begin()->first
       << "\": " << SymbolFlags.begin()->second << "}";
    for (auto &KV :
         make_range(std::next(SymbolFlags.begin()), SymbolFlags.end()))
      OS << ", {\"" << *KV.first << "\": " << KV.second << "}";
  }
  OS << " }";
  return OS;
}

FailedToResolve::FailedToResolve(SymbolNameSet Symbols)
    : Symbols(std::move(Symbols)) {
  assert(!this->Symbols.empty() && "Can not fail to resolve an empty set");
}

std::error_code FailedToResolve::convertToErrorCode() const {
  return orcError(OrcErrorCode::UnknownORCError);
}

void FailedToResolve::log(raw_ostream &OS) const {
  OS << "Failed to resolve symbols: " << Symbols;
}

FailedToFinalize::FailedToFinalize(SymbolNameSet Symbols)
    : Symbols(std::move(Symbols)) {
  assert(!this->Symbols.empty() && "Can not fail to finalize an empty set");
}

std::error_code FailedToFinalize::convertToErrorCode() const {
  return orcError(OrcErrorCode::UnknownORCError);
}

void FailedToFinalize::log(raw_ostream &OS) const {
  OS << "Failed to finalize symbols: " << Symbols;
}

AsynchronousSymbolQuery::AsynchronousSymbolQuery(
    const SymbolNameSet &Symbols, SymbolsResolvedCallback NotifySymbolsResolved,
    SymbolsReadyCallback NotifySymbolsReady)
    : NotifySymbolsResolved(std::move(NotifySymbolsResolved)),
      NotifySymbolsReady(std::move(NotifySymbolsReady)) {
  assert(this->NotifySymbolsResolved &&
         "Symbols resolved callback must be set");
  assert(this->NotifySymbolsReady && "Symbols ready callback must be set");
  OutstandingResolutions = OutstandingFinalizations = Symbols.size();
}

void AsynchronousSymbolQuery::notifyMaterializationFailed(Error Err) {
  if (OutstandingResolutions != 0)
    NotifySymbolsResolved(std::move(Err));
  else if (OutstandingFinalizations != 0)
    NotifySymbolsReady(std::move(Err));
  else
    consumeError(std::move(Err));
  OutstandingResolutions = OutstandingFinalizations = 0;
}

void AsynchronousSymbolQuery::resolve(SymbolStringPtr Name,
                                      JITEvaluatedSymbol Sym) {
  // If OutstandingResolutions is zero we must have errored out already. Just
  // ignore this.
  if (OutstandingResolutions == 0)
    return;

  assert(!Symbols.count(Name) && "Symbol has already been assigned an address");
  Symbols.insert(std::make_pair(std::move(Name), std::move(Sym)));
  --OutstandingResolutions;
  if (OutstandingResolutions == 0)
    NotifySymbolsResolved(std::move(Symbols));
}

void AsynchronousSymbolQuery::finalizeSymbol() {
  // If OutstandingFinalizations is zero we must have errored out already. Just
  // ignore this.
  if (OutstandingFinalizations == 0)
    return;

  assert(OutstandingFinalizations > 0 && "All symbols already finalized");
  --OutstandingFinalizations;
  if (OutstandingFinalizations == 0)
    NotifySymbolsReady(Error::success());
}

MaterializationResponsibility::MaterializationResponsibility(
    VSO &V, SymbolFlagsMap SymbolFlags)
    : V(V), SymbolFlags(std::move(SymbolFlags)) {
  assert(!this->SymbolFlags.empty() && "Materializing nothing?");
}

MaterializationResponsibility::~MaterializationResponsibility() {
  assert(SymbolFlags.empty() &&
         "All symbols should have been explicitly materialized or failed");
}

void MaterializationResponsibility::resolve(const SymbolMap &Symbols) {
#ifndef NDEBUG
  for (auto &KV : Symbols) {
    auto I = SymbolFlags.find(KV.first);
    assert(I != SymbolFlags.end() &&
           "Resolving symbol outside this responsibility set");
    assert(KV.second.getFlags() == I->second &&
           "Resolving symbol with incorrect flags");
  }
#endif
  V.resolve(Symbols);
}

void MaterializationResponsibility::finalize() {
  SymbolNameSet SymbolNames;
  for (auto &KV : SymbolFlags)
    SymbolNames.insert(KV.first);
  SymbolFlags.clear();
  V.finalize(SymbolNames);
}

void MaterializationResponsibility::notifyMaterializationFailed() {
  SymbolNameSet SymbolNames;
  for (auto &KV : SymbolFlags)
    SymbolNames.insert(KV.first);
  SymbolFlags.clear();
  V.notifyMaterializationFailed(SymbolNames);
}

MaterializationResponsibility
MaterializationResponsibility::delegate(SymbolNameSet Symbols) {
  SymbolFlagsMap ExtractedFlags;

  for (auto &S : Symbols) {
    auto I = SymbolFlags.find(S);
    ExtractedFlags.insert(*I);
    SymbolFlags.erase(I);
  }

  return MaterializationResponsibility(V, std::move(ExtractedFlags));
}

VSO::Materializer::Materializer(std::unique_ptr<MaterializationUnit> MU,
                                MaterializationResponsibility R)
    : MU(std::move(MU)), R(std::move(R)) {}

void VSO::Materializer::operator()() { MU->materialize(std::move(R)); }

VSO::UnmaterializedInfo::UnmaterializedInfo(
    std::unique_ptr<MaterializationUnit> MU)
    : MU(std::move(MU)), Symbols(this->MU->getSymbols()) {}

void VSO::UnmaterializedInfo::discard(VSO &V, SymbolStringPtr Name) {
  assert(MU && "No materializer attached");
  MU->discard(V, Name);
  auto I = Symbols.find(Name);
  assert(I != Symbols.end() && "Symbol not found in this MU");
  Symbols.erase(I);
}

VSO::SymbolTableEntry::SymbolTableEntry(JITSymbolFlags Flags,
                                        UnmaterializedInfoIterator UMII)
    : Flags(Flags), UMII(std::move(UMII)) {
  // We *don't* expect isLazy to be set here. That's for the VSO to do.
  assert(!Flags.isLazy() && "Initial flags include lazy?");
  assert(!Flags.isMaterializing() && "Initial flags include materializing");
  this->Flags |= JITSymbolFlags::Lazy;
}

VSO::SymbolTableEntry::SymbolTableEntry(JITSymbolFlags Flags)
    : Flags(Flags), Address(0) {
  // We *don't* expect isMaterializing to be set here. That's for the VSO to do.
  assert(!Flags.isLazy() && "Initial flags include lazy?");
  assert(!Flags.isMaterializing() && "Initial flags include materializing");
  this->Flags |= JITSymbolFlags::Materializing;
}

VSO::SymbolTableEntry::SymbolTableEntry(JITEvaluatedSymbol Sym)
    : Flags(Sym.getFlags()), Address(Sym.getAddress()) {
  assert(!Flags.isLazy() && !Flags.isMaterializing() &&
         "This constructor is for final symbols only");
}

VSO::SymbolTableEntry::SymbolTableEntry(SymbolTableEntry &&Other)
    : Flags(Other.Flags), Address(0) {
  if (this->Flags.isLazy())
    UMII = std::move(Other.UMII);
  else
    Address = Other.Address;
}

VSO::SymbolTableEntry &VSO::SymbolTableEntry::
operator=(SymbolTableEntry &&Other) {
  destroy();
  Flags = std::move(Other.Flags);
  if (Other.Flags.isLazy()) {
    UMII = std::move(Other.UMII);
  } else
    Address = Other.Address;
  return *this;
}

VSO::SymbolTableEntry::~SymbolTableEntry() { destroy(); }

void VSO::SymbolTableEntry::replaceWith(VSO &V, SymbolStringPtr Name,
                                        JITEvaluatedSymbol Sym) {
  assert(!Flags.isMaterializing() &&
         "Attempting to replace definition during materialization?");
  if (Flags.isLazy()) {
    UMII->discard(V, Name);
    if (UMII->Symbols.empty())
      V.UnmaterializedInfos.erase(UMII);
  }
  destroy();
  Flags = Sym.getFlags();
  Address = Sym.getAddress();
}

void VSO::SymbolTableEntry::replaceWith(VSO &V, SymbolStringPtr Name,
                                        JITSymbolFlags NewFlags,
                                        UnmaterializedInfoIterator NewUMII) {
  assert(!Flags.isMaterializing() &&
         "Attempting to replace definition during materialization?");
  if (Flags.isLazy()) {
    UMII->discard(V, Name);
    if (UMII->Symbols.empty())
      V.UnmaterializedInfos.erase(UMII);
  }
  destroy();
  Flags = NewFlags;
  UMII = std::move(NewUMII);
}

void VSO::SymbolTableEntry::replaceMaterializing(VSO &V, SymbolStringPtr Name,
                                                 JITSymbolFlags NewFlags) {
  assert(!NewFlags.isWeak() &&
         "Can't define a lazy symbol in materializing mode");
  assert(!NewFlags.isLazy() && !NewFlags.isMaterializing() &&
         "Flags should not be in lazy or materializing state");
  if (Flags.isLazy()) {
    UMII->discard(V, Name);
    if (UMII->Symbols.empty())
      V.UnmaterializedInfos.erase(UMII);
  }
  destroy();
  Flags = NewFlags;
  Flags |= JITSymbolFlags::Materializing;
  Address = 0;
}

void VSO::SymbolTableEntry::notifyMaterializing() {
  assert(Flags.isLazy() && "Can only start materializing from lazy state");
  UMII.~UnmaterializedInfoIterator();
  Flags &= ~JITSymbolFlags::Lazy;
  Flags |= JITSymbolFlags::Materializing;
  Address = 0;
}

void VSO::SymbolTableEntry::resolve(VSO &V, JITEvaluatedSymbol Sym) {
  assert(!Flags.isLazy() && Flags.isMaterializing() &&
         "Can only resolve in materializing state");
  Flags = Sym.getFlags();
  Flags |= JITSymbolFlags::Materializing;
  Address = Sym.getAddress();
  assert(Address != 0 && "Can not resolve to null");
}

void VSO::SymbolTableEntry::finalize() {
  assert(Address != 0 && "Cannot finalize with null address");
  assert(Flags.isMaterializing() && !Flags.isLazy() &&
         "Symbol should be in materializing state");
  Flags &= ~JITSymbolFlags::Materializing;
}

void VSO::SymbolTableEntry::destroy() {
  if (Flags.isLazy())
    UMII.~UnmaterializedInfoIterator();
}

VSO::RelativeLinkageStrength VSO::compareLinkage(Optional<JITSymbolFlags> Old,
                                                 JITSymbolFlags New) {
  if (Old == None)
    return llvm::orc::VSO::NewDefinitionIsStronger;

  if (Old->isStrong()) {
    if (New.isStrong())
      return llvm::orc::VSO::DuplicateDefinition;
    else
      return llvm::orc::VSO::ExistingDefinitionIsStronger;
  } else {
    if (New.isStrong())
      return llvm::orc::VSO::NewDefinitionIsStronger;
    else
      return llvm::orc::VSO::ExistingDefinitionIsStronger;
  }
}

VSO::RelativeLinkageStrength
VSO::compareLinkage(SymbolStringPtr Name, JITSymbolFlags NewFlags) const {
  auto I = Symbols.find(Name);
  return compareLinkage(
      I == Symbols.end() ? None : Optional<JITSymbolFlags>(I->second.Flags),
      NewFlags);
}

Error VSO::define(SymbolMap NewSymbols) {
  Error Err = Error::success();
  for (auto &KV : NewSymbols) {
    auto I = Symbols.find(KV.first);
    auto LinkageResult = compareLinkage(
        I == Symbols.end() ? None : Optional<JITSymbolFlags>(I->second.Flags),
        KV.second.getFlags());

    // Silently discard weaker definitions.
    if (LinkageResult == ExistingDefinitionIsStronger)
      continue;

    // Report duplicate definition errors.
    if (LinkageResult == DuplicateDefinition) {
      Err = joinErrors(std::move(Err),
                       make_error<orc::DuplicateDefinition>(*KV.first));
      continue;
    }

    if (I != Symbols.end())
      I->second.replaceWith(*this, I->first, KV.second);
    else
      Symbols.insert(std::make_pair(KV.first, std::move(KV.second)));
  }
  return Err;
}

Error VSO::defineLazy(std::unique_ptr<MaterializationUnit> MU) {
  auto UMII = UnmaterializedInfos.insert(UnmaterializedInfos.end(),
                                         UnmaterializedInfo(std::move(MU)));

  Error Err = Error::success();
  for (auto &KV : UMII->Symbols) {
    auto I = Symbols.find(KV.first);

    assert((I == Symbols.end() ||
            !I->second.Flags.isMaterializing()) &&
               "Attempt to replace materializing symbol definition");

    auto LinkageResult = compareLinkage(
        I == Symbols.end() ? None : Optional<JITSymbolFlags>(I->second.Flags),
        KV.second);

    // Discard weaker definitions.
    if (LinkageResult == ExistingDefinitionIsStronger) {
      UMII->discard(*this, KV.first);
      continue;
    }

    // Report duplicate definition errors.
    if (LinkageResult == DuplicateDefinition) {
      Err = joinErrors(std::move(Err),
                       make_error<orc::DuplicateDefinition>(*KV.first));
      // Duplicate definitions are discarded, so remove the duplicates from
      // materializer.
      UMII->discard(*this, KV.first);
      continue;
    }

    // Existing definition was weaker. Replace it.
    if (I != Symbols.end())
      I->second.replaceWith(*this, KV.first, KV.second, UMII);
    else
      Symbols.insert(
          std::make_pair(KV.first, SymbolTableEntry(KV.second, UMII)));
  }

  if (UMII->Symbols.empty())
    UnmaterializedInfos.erase(UMII);

  return Err;
}

SymbolNameSet VSO::lookupFlags(SymbolFlagsMap &Flags, SymbolNameSet Names) {

  for (SymbolNameSet::iterator I = Names.begin(), E = Names.end(); I != E;) {
    auto Tmp = I++;
    auto SymI = Symbols.find(*Tmp);

    // If the symbol isn't in this dylib then just continue.
    if (SymI == Symbols.end())
      continue;

    Names.erase(Tmp);

    Flags[SymI->first] =
        JITSymbolFlags::stripTransientFlags(SymI->second.Flags);
  }

  return Names;
}

VSO::LookupResult VSO::lookup(std::shared_ptr<AsynchronousSymbolQuery> Query,
                              SymbolNameSet Names) {
  MaterializerList Materializers;

  for (SymbolNameSet::iterator I = Names.begin(), E = Names.end(); I != E;) {
    auto Tmp = I++;
    auto SymI = Symbols.find(*Tmp);

    // If the symbol isn't in this dylib then just continue.
    if (SymI == Symbols.end())
      continue;

    // The symbol is in the VSO. Erase it from Names and proceed.
    Names.erase(Tmp);

    // If this symbol has not been materialized yet grab its materializer,
    // move all of its sibling symbols to the materializing state, and
    // delete its unmaterialized info.
    if (SymI->second.Flags.isLazy()) {
      assert(SymI->second.UMII->MU &&
             "Lazy symbol has no materializer attached");
      auto MU = std::move(SymI->second.UMII->MU);
      auto SymbolFlags = std::move(SymI->second.UMII->Symbols);
      UnmaterializedInfos.erase(SymI->second.UMII);

      for (auto &KV : SymbolFlags) {
        auto SiblingI = Symbols.find(KV.first);
        MaterializingInfos.insert(
            std::make_pair(SiblingI->first, MaterializingInfo()));
        SiblingI->second.notifyMaterializing();
      }

      Materializers.push_back(Materializer(
          std::move(MU),
          MaterializationResponsibility(*this, std::move(SymbolFlags))));
    }

    // If this symbol already has a fully materialized value, just use it.
    if (!SymI->second.Flags.isMaterializing()) {
      Query->resolve(SymI->first, JITEvaluatedSymbol(SymI->second.Address,
                                                     SymI->second.Flags));
      Query->finalizeSymbol();
      continue;
    }

    // If this symbol is materializing, then get (or create) its
    // MaterializingInfo struct and appaend the query.
    auto J = MaterializingInfos.find(SymI->first);
    assert(J != MaterializingInfos.end() && "Missing MaterializingInfo");

    if (SymI->second.Address) {
      auto Sym = JITEvaluatedSymbol(SymI->second.Address, SymI->second.Flags);
      Query->resolve(SymI->first, Sym);
      assert(J->second.PendingResolution.empty() &&
             "Queries still pending resolution on resolved symbol?");
      J->second.PendingFinalization.push_back(Query);
    } else {
      assert(J->second.PendingFinalization.empty() &&
             "Queries pendiing finalization on unresolved symbol?");
      J->second.PendingResolution.push_back(Query);
    }
  }

  return {std::move(Materializers), std::move(Names)};
}

void VSO::resolve(const SymbolMap &SymbolValues) {
  for (auto &KV : SymbolValues) {
    auto I = Symbols.find(KV.first);
    assert(I != Symbols.end() && "Resolving symbol not present in this dylib");
    I->second.resolve(*this, KV.second);

    auto J = MaterializingInfos.find(KV.first);
    if (J == MaterializingInfos.end())
      continue;

    assert(J->second.PendingFinalization.empty() &&
           "Queries already pending finalization?");
    for (auto &Q : J->second.PendingResolution)
      Q->resolve(KV.first, KV.second);
    J->second.PendingFinalization = std::move(J->second.PendingResolution);
    J->second.PendingResolution = MaterializingInfo::QueryList();
  }
}

void VSO::notifyMaterializationFailed(const SymbolNameSet &Names) {
  assert(!Names.empty() && "Failed to materialize empty set?");

  std::map<std::shared_ptr<AsynchronousSymbolQuery>, SymbolNameSet>
      ResolutionFailures;
  std::map<std::shared_ptr<AsynchronousSymbolQuery>, SymbolNameSet>
      FinalizationFailures;

  for (auto &S : Names) {
    auto I = Symbols.find(S);
    assert(I != Symbols.end() && "Symbol not present in this VSO");

    auto J = MaterializingInfos.find(S);
    if (J != MaterializingInfos.end()) {
      if (J->second.PendingFinalization.empty()) {
        for (auto &Q : J->second.PendingResolution)
          ResolutionFailures[Q].insert(S);
      } else {
        for (auto &Q : J->second.PendingFinalization)
          FinalizationFailures[Q].insert(S);
      }
      MaterializingInfos.erase(J);
    }
    Symbols.erase(I);
  }

  for (auto &KV : ResolutionFailures)
    KV.first->notifyMaterializationFailed(
        make_error<FailedToResolve>(std::move(KV.second)));

  for (auto &KV : FinalizationFailures)
    KV.first->notifyMaterializationFailed(
        make_error<FailedToFinalize>(std::move(KV.second)));
}

void VSO::finalize(const SymbolNameSet &SymbolsToFinalize) {
  for (auto &S : SymbolsToFinalize) {
    auto I = Symbols.find(S);
    assert(I != Symbols.end() && "Finalizing symbol not present in this dylib");

    auto J = MaterializingInfos.find(S);
    if (J != MaterializingInfos.end()) {
      assert(J->second.PendingResolution.empty() &&
             "Queries still pending resolution?");
      for (auto &Q : J->second.PendingFinalization)
        Q->finalizeSymbol();
      MaterializingInfos.erase(J);
    }
    I->second.finalize();
  }
}

Expected<SymbolMap> lookup(const std::vector<VSO *> &VSOs, SymbolNameSet Names,
                           MaterializationDispatcher DispatchMaterialization) {
#if LLVM_ENABLE_THREADS
  // In the threaded case we use promises to return the results.
  std::promise<SymbolMap> PromisedResult;
  std::mutex ErrMutex;
  Error ResolutionError = Error::success();
  std::promise<void> PromisedReady;
  Error ReadyError = Error::success();
  auto OnResolve = [&](Expected<SymbolMap> Result) {
    if (Result)
      PromisedResult.set_value(std::move(*Result));
    else {
      {
        ErrorAsOutParameter _(&ResolutionError);
        std::lock_guard<std::mutex> Lock(ErrMutex);
        ResolutionError = Result.takeError();
      }
      PromisedResult.set_value(SymbolMap());
    }
  };
  auto OnReady = [&](Error Err) {
    if (Err) {
      ErrorAsOutParameter _(&ReadyError);
      std::lock_guard<std::mutex> Lock(ErrMutex);
      ReadyError = std::move(Err);
    }
    PromisedReady.set_value();
  };
#else
  SymbolMap Result;
  Error ResolutionError = Error::success();
  Error ReadyError = Error::success();

  auto OnResolve = [&](Expected<SymbolMap> R) {
    ErrorAsOutParameter _(&ResolutionError);
    if (R)
      Result = std::move(*R);
    else
      ResolutionError = R.takeError();
  };
  auto OnReady = [&](Error Err) {
    ErrorAsOutParameter _(&ReadyError);
    if (Err)
      ReadyError = std::move(Err);
  };
#endif

  auto Query = std::make_shared<AsynchronousSymbolQuery>(
      Names, std::move(OnResolve), std::move(OnReady));
  SymbolNameSet UnresolvedSymbols(std::move(Names));

  for (auto *V : VSOs) {

    if (UnresolvedSymbols.empty())
      break;

    assert(V && "VSO pointers in VSOs list should be non-null");
    auto LR = V->lookup(Query, UnresolvedSymbols);
    UnresolvedSymbols = std::move(LR.UnresolvedSymbols);

    for (auto &M : LR.Materializers)
      DispatchMaterialization(std::move(M));
  }

#if LLVM_ENABLE_THREADS
  auto ResultFuture = PromisedResult.get_future();
  auto Result = ResultFuture.get();

  {
    std::lock_guard<std::mutex> Lock(ErrMutex);
    if (ResolutionError) {
      // ReadyError will never be assigned. Consume the success value.
      cantFail(std::move(ReadyError));
      return std::move(ResolutionError);
    }
  }

  auto ReadyFuture = PromisedReady.get_future();
  ReadyFuture.get();

  {
    std::lock_guard<std::mutex> Lock(ErrMutex);
    if (ReadyError)
      return std::move(ReadyError);
  }

  return std::move(Result);

#else
  if (ResolutionError) {
    // ReadyError will never be assigned. Consume the success value.
    cantFail(std::move(ReadyError));
    return std::move(ResolutionError);
  }

  if (ReadyError)
    return std::move(ReadyError);

  return Result;
#endif
}

/// @brief Look up a symbol by searching a list of VSOs.
Expected<JITEvaluatedSymbol>
lookup(const std::vector<VSO *> VSOs, SymbolStringPtr Name,
       MaterializationDispatcher DispatchMaterialization) {
  SymbolNameSet Names({Name});
  if (auto ResultMap =
          lookup(VSOs, std::move(Names), std::move(DispatchMaterialization))) {
    assert(ResultMap->size() == 1 && "Unexpected number of results");
    assert(ResultMap->count(Name) && "Missing result for symbol");
    return ResultMap->begin()->second;
  } else
    return ResultMap.takeError();
}

void ExecutionSession::logErrorsToStdErr(Error Err) {
  logAllUnhandledErrors(std::move(Err), errs(), "JIT session error: ");
}

} // End namespace orc.
} // End namespace llvm.
