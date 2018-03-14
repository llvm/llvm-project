//===--------- Core.cpp - Core ORC APIs (SymbolSource, VSO, etc.) ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"

#if LLVM_ENABLE_THREADS
#include <future>
#endif

namespace llvm {
namespace orc {

void SymbolResolver::anchor() {}
void SymbolSource::anchor() {}

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

void AsynchronousSymbolQuery::setFailed(Error Err) {
  OutstandingResolutions = OutstandingFinalizations = 0;
  if (NotifySymbolsResolved)
    NotifySymbolsResolved(std::move(Err));
  else
    NotifySymbolsReady(std::move(Err));
}

void AsynchronousSymbolQuery::setDefinition(SymbolStringPtr Name,
                                            JITEvaluatedSymbol Sym) {
  // If OutstandingResolutions is zero we must have errored out already. Just
  // ignore this.
  if (OutstandingResolutions == 0)
    return;

  assert(!Symbols.count(Name) && "Symbol has already been assigned an address");
  Symbols.insert(std::make_pair(std::move(Name), std::move(Sym)));
  --OutstandingResolutions;
  if (OutstandingResolutions == 0) {
    NotifySymbolsResolved(std::move(Symbols));
    // Null out NotifySymbolsResolved to indicate that we've already called it.
    NotifySymbolsResolved = {};
  }
}

void AsynchronousSymbolQuery::notifySymbolFinalized() {
  // If OutstandingFinalizations is zero we must have errored out already. Just
  // ignore this.
  if (OutstandingFinalizations == 0)
    return;

  assert(OutstandingFinalizations > 0 && "All symbols already finalized");
  --OutstandingFinalizations;
  if (OutstandingFinalizations == 0)
    NotifySymbolsReady(Error::success());
}

VSO::MaterializationInfo::MaterializationInfo(
    JITSymbolFlags Flags, std::shared_ptr<SymbolSource> Source)
    : Flags(std::move(Flags)), Source(std::move(Source)) {}

JITSymbolFlags VSO::MaterializationInfo::getFlags() const { return Flags; }

JITTargetAddress VSO::MaterializationInfo::getAddress() const {
  return Address;
}

void VSO::MaterializationInfo::replaceWithSource(
    VSO &V, SymbolStringPtr Name, JITSymbolFlags NewFlags,
    std::shared_ptr<SymbolSource> NewSource) {
  assert(Address == 0 && PendingResolution.empty() &&
         PendingFinalization.empty() &&
         "Cannot replace source during or after materialization");
  Source->discard(V, Name);
  Flags = std::move(NewFlags);
  Source = std::move(NewSource);
}

std::shared_ptr<SymbolSource> VSO::MaterializationInfo::query(
    SymbolStringPtr Name, std::shared_ptr<AsynchronousSymbolQuery> Query) {
  if (Address == 0) {
    PendingResolution.push_back(std::move(Query));
    auto S = std::move(Source);
    Source = nullptr;
    return S;
  }

  Query->setDefinition(Name, JITEvaluatedSymbol(Address, Flags));
  PendingFinalization.push_back(std::move(Query));
  return nullptr;
}

void VSO::MaterializationInfo::resolve(VSO &V, SymbolStringPtr Name,
                                       JITEvaluatedSymbol Sym) {
  if (Source) {
    Source->discard(V, Name);
    Source = nullptr;
  }

  // FIXME: Sanity check flags?
  Flags = Sym.getFlags();
  Address = Sym.getAddress();
  for (auto &Query : PendingResolution) {
    Query->setDefinition(Name, std::move(Sym));
    PendingFinalization.push_back(std::move(Query));
  }
  PendingResolution = {};
}

void VSO::MaterializationInfo::finalize() {
  for (auto &Query : PendingFinalization)
    Query->notifySymbolFinalized();
  PendingFinalization = {};
}

VSO::SymbolTableEntry::SymbolTableEntry(JITSymbolFlags Flags,
                                        std::shared_ptr<SymbolSource> Source)
    : Flags(JITSymbolFlags::FlagNames(Flags | JITSymbolFlags::NotMaterialized)),
      MatInfo(
          llvm::make_unique<MaterializationInfo>(Flags, std::move(Source))) {
  // FIXME: Assert flag sanity.
}

VSO::SymbolTableEntry::SymbolTableEntry(JITEvaluatedSymbol Sym)
    : Flags(Sym.getFlags()), Address(Sym.getAddress()) {
  // FIXME: Assert flag sanity.
}

VSO::SymbolTableEntry::SymbolTableEntry(SymbolTableEntry &&Other)
    : Flags(Other.Flags), Address(0) {
  if (Flags.isMaterialized())
    Address = Other.Address;
  else
    MatInfo = std::move(Other.MatInfo);
}

VSO::SymbolTableEntry::~SymbolTableEntry() {
  if (!Flags.isMaterialized())
    MatInfo.std::unique_ptr<MaterializationInfo>::~unique_ptr();
}

JITSymbolFlags VSO::SymbolTableEntry::getFlags() const { return Flags; }

void VSO::SymbolTableEntry::replaceWithSource(
    VSO &V, SymbolStringPtr Name, JITSymbolFlags NewFlags,
    std::shared_ptr<SymbolSource> NewSource) {
  bool ReplaceExisting = !Flags.isMaterialized();
  Flags = NewFlags;
  if (ReplaceExisting)
    MatInfo->replaceWithSource(V, Name, Flags, std::move(NewSource));
  else
    new (&MatInfo) std::unique_ptr<MaterializationInfo>(
        llvm::make_unique<MaterializationInfo>(Flags, std::move(NewSource)));
}

std::shared_ptr<SymbolSource>
VSO::SymbolTableEntry::query(SymbolStringPtr Name,
                             std::shared_ptr<AsynchronousSymbolQuery> Query) {
  if (Flags.isMaterialized()) {
    Query->setDefinition(std::move(Name), JITEvaluatedSymbol(Address, Flags));
    Query->notifySymbolFinalized();
    return nullptr;
  } else
    return MatInfo->query(std::move(Name), std::move(Query));
}

void VSO::SymbolTableEntry::resolve(VSO &V, SymbolStringPtr Name,
                                    JITEvaluatedSymbol Sym) {
  if (Flags.isMaterialized()) {
    // FIXME: Should we assert flag state here (flags must match except for
    //        materialization state, overrides must be legal) or in the caller
    //        in VSO?
    Flags = Sym.getFlags();
    Address = Sym.getAddress();
  } else
    MatInfo->resolve(V, std::move(Name), std::move(Sym));
}

void VSO::SymbolTableEntry::finalize() {
  if (!Flags.isMaterialized()) {
    auto TmpMatInfo = std::move(MatInfo);
    MatInfo.std::unique_ptr<MaterializationInfo>::~unique_ptr();
    // FIXME: Assert flag sanity?
    Flags = TmpMatInfo->getFlags();
    Address = TmpMatInfo->getAddress();
    TmpMatInfo->finalize();
  }
  assert(Flags.isMaterialized() && "Trying to finalize not-emitted symbol");
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
  return compareLinkage(I == Symbols.end()
                            ? None
                            : Optional<JITSymbolFlags>(I->second.getFlags()),
                        NewFlags);
}

Error VSO::define(SymbolMap NewSymbols) {
  Error Err = Error::success();
  for (auto &KV : NewSymbols) {
    auto I = Symbols.find(KV.first);
    auto LinkageResult = compareLinkage(
        I == Symbols.end() ? None
                           : Optional<JITSymbolFlags>(I->second.getFlags()),
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

    if (I != Symbols.end()) {
      I->second.resolve(*this, KV.first, std::move(KV.second));
      I->second.finalize();
    } else
      Symbols.insert(std::make_pair(KV.first, std::move(KV.second)));
  }
  return Err;
}

Error VSO::defineLazy(const SymbolFlagsMap &NewSymbols,
                      std::shared_ptr<SymbolSource> Source) {
  Error Err = Error::success();
  for (auto &KV : NewSymbols) {
    auto I = Symbols.find(KV.first);

    auto LinkageResult = compareLinkage(
        I == Symbols.end() ? None
                           : Optional<JITSymbolFlags>(I->second.getFlags()),
        KV.second);

    // Discard weaker definitions.
    if (LinkageResult == ExistingDefinitionIsStronger)
      Source->discard(*this, KV.first);

    // Report duplicate definition errors.
    if (LinkageResult == DuplicateDefinition) {
      Err = joinErrors(std::move(Err),
                       make_error<orc::DuplicateDefinition>(*KV.first));
      continue;
    }

    if (I != Symbols.end())
      I->second.replaceWithSource(*this, KV.first, KV.second, Source);
    else
      Symbols.emplace(
          std::make_pair(KV.first, SymbolTableEntry(KV.second, Source)));
  }
  return Err;
}

void VSO::resolve(SymbolMap SymbolValues) {
  for (auto &KV : SymbolValues) {
    auto I = Symbols.find(KV.first);
    assert(I != Symbols.end() && "Resolving symbol not present in this dylib");
    I->second.resolve(*this, KV.first, std::move(KV.second));
  }
}

void VSO::finalize(SymbolNameSet SymbolsToFinalize) {
  for (auto &S : SymbolsToFinalize) {
    auto I = Symbols.find(S);
    assert(I != Symbols.end() && "Finalizing symbol not present in this dylib");
    I->second.finalize();
  }
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
        JITSymbolFlags::stripTransientFlags(SymI->second.getFlags());
  }

  return Names;
}

VSO::LookupResult VSO::lookup(std::shared_ptr<AsynchronousSymbolQuery> Query,
                              SymbolNameSet Names) {
  SourceWorkMap MaterializationWork;

  for (SymbolNameSet::iterator I = Names.begin(), E = Names.end(); I != E;) {
    auto Tmp = I++;
    auto SymI = Symbols.find(*Tmp);

    // If the symbol isn't in this dylib then just continue.
    if (SymI == Symbols.end())
      continue;

    // The symbol is in the dylib. Erase it from Names and proceed.
    Names.erase(Tmp);

    // Forward the query to the given SymbolTableEntry, and if it return a
    // layer to perform materialization with, add that to the
    // MaterializationWork map.
    if (auto Source = SymI->second.query(SymI->first, Query))
      MaterializationWork[Source].insert(SymI->first);
  }

  return {std::move(MaterializationWork), std::move(Names)};
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

  for (auto *VSO : VSOs) {

    if (UnresolvedSymbols.empty())
      break;

    assert(VSO && "VSO pointers in VSOs list should be non-null");
    auto LR = VSO->lookup(Query, UnresolvedSymbols);
    UnresolvedSymbols = std::move(LR.UnresolvedSymbols);

    for (auto I = LR.MaterializationWork.begin(),
              E = LR.MaterializationWork.end();
         I != E;) {
      auto Tmp = I++;
      std::shared_ptr<SymbolSource> Source = Tmp->first;
      SymbolNameSet Names = std::move(Tmp->second);
      LR.MaterializationWork.erase(Tmp);
      DispatchMaterialization(*VSO, std::move(Source), std::move(Names));
    }
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

ExecutionSession::ExecutionSession(SymbolStringPool &SSP) : SSP(SSP) {}

VModuleKey ExecutionSession::allocateVModule() { return ++LastKey; }

void ExecutionSession::releaseVModule(VModuleKey VMod) {
  // FIXME: Recycle keys.
}

void ExecutionSession::logErrorsToStdErr(Error Err) {
  logAllUnhandledErrors(std::move(Err), errs(), "JIT session error: ");
}

} // End namespace orc.
} // End namespace llvm.
