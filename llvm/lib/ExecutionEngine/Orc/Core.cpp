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

#if LLVM_ENABLE_THREADS
#include <future>
#endif

namespace llvm {
namespace orc {

void MaterializationUnit::anchor() {}
void SymbolResolver::anchor() {}

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
    size_t SymbolsRemaining, std::unique_ptr<MaterializationUnit> MU)
    : SymbolsRemaining(SymbolsRemaining), MU(std::move(MU)) {}

VSO::SymbolTableEntry::SymbolTableEntry(
    JITSymbolFlags Flags, MaterializationInfoIterator MaterializationInfoItr)
    : Flags(JITSymbolFlags::FlagNames(Flags | JITSymbolFlags::NotMaterialized)),
      MaterializationInfoItr(std::move(MaterializationInfoItr)) {
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
    MaterializationInfoItr = std::move(Other.MaterializationInfoItr);
}

VSO::SymbolTableEntry::~SymbolTableEntry() { destroy(); }

VSO::SymbolTableEntry &VSO::SymbolTableEntry::
operator=(JITEvaluatedSymbol Sym) {
  destroy();
  Flags = Sym.getFlags();
  Address = Sym.getAddress();
  return *this;
}

void VSO::SymbolTableEntry::destroy() {
  if (!Flags.isMaterialized())
    MaterializationInfoItr.~MaterializationInfoIterator();
}

JITSymbolFlags VSO::SymbolTableEntry::getFlags() const { return Flags; }

void VSO::SymbolTableEntry::replaceWith(
    VSO &V, SymbolStringPtr Name, JITSymbolFlags NewFlags,
    MaterializationInfoIterator NewMaterializationInfoItr) {
  bool ReplaceExistingLazyDefinition = !Flags.isMaterialized();
  Flags = NewFlags;
  if (ReplaceExistingLazyDefinition) {
    // If we are replacing an existing lazy definition with a stronger one,
    // we need to notify the old lazy definition to discard its definition.
    assert((*MaterializationInfoItr)->MU != nullptr &&
           (*MaterializationInfoItr)->Symbols.count(Name) == 0 &&
           (*MaterializationInfoItr)->PendingResolution.count(Name) == 0 &&
           (*MaterializationInfoItr)->PendingFinalization.count(Name) == 0 &&
           "Attempt to replace materializer during materialization");

    if (--(*MaterializationInfoItr)->SymbolsRemaining == 0)
      V.MaterializationInfos.erase(MaterializationInfoItr);
  }
  MaterializationInfoItr = std::move(NewMaterializationInfoItr);
}

std::unique_ptr<MaterializationUnit>
VSO::SymbolTableEntry::query(SymbolStringPtr Name,
                             std::shared_ptr<AsynchronousSymbolQuery> Query) {
  if (Flags.isMaterialized()) {
    Query->setDefinition(std::move(Name), JITEvaluatedSymbol(Address, Flags));
    Query->notifySymbolFinalized();
    return nullptr;
  } else {
    if ((*MaterializationInfoItr)->MU) {
      assert((*MaterializationInfoItr)->PendingResolution.count(Name) == 0 &&
             (*MaterializationInfoItr)->PendingFinalization.count(Name) == 0 &&
             "Materializer should have been activated on first query");
      (*MaterializationInfoItr)
          ->PendingResolution[Name]
          .push_back(std::move(Query));
      return std::move((*MaterializationInfoItr)->MU);
    } else {
      assert((*MaterializationInfoItr)->MU == nullptr &&
             "Materializer should have been activated on first query");
      auto SymValueItr = (*MaterializationInfoItr)->Symbols.find(Name);
      if (SymValueItr == (*MaterializationInfoItr)->Symbols.end()) {
        // Symbol has not been resolved yet.
        (*MaterializationInfoItr)
            ->PendingResolution[Name]
            .push_back(std::move(Query));
        return nullptr;
      } else {
        // Symbol has already resolved, is just waiting on finalization.
        Query->setDefinition(Name, SymValueItr->second);
        (*MaterializationInfoItr)
            ->PendingFinalization[Name]
            .push_back(std::move(Query));
        return nullptr;
      }
    }
  }
}

void VSO::SymbolTableEntry::resolve(VSO &V, SymbolStringPtr Name,
                                    JITEvaluatedSymbol Sym) {
  if (Flags.isMaterialized()) {
    // FIXME: Should we assert flag state here (flags must match except for
    //        materialization state, overrides must be legal) or in the caller
    //        in VSO?
    Flags = Sym.getFlags();
    Address = Sym.getAddress();
  } else {
    assert((*MaterializationInfoItr)->MU == nullptr &&
           "Can not resolve a symbol that has not been materialized");
    assert((*MaterializationInfoItr)->Symbols.count(Name) == 0 &&
           "Symbol resolved more than once");

    // Add the symbol to the MaterializationInfo Symbols table.
    (*MaterializationInfoItr)->Symbols[Name] = Sym;

    // If there are any queries waiting on this symbol then notify them that it
    // has been resolved, then move them to the PendingFinalization list.
    auto I = (*MaterializationInfoItr)->PendingResolution.find(Name);
    if (I != (*MaterializationInfoItr)->PendingResolution.end()) {
      assert((*MaterializationInfoItr)->PendingFinalization.count(Name) == 0 &&
             "Queries already pending finalization on newly resolved symbol");
      auto &PendingFinalization =
          (*MaterializationInfoItr)->PendingFinalization[Name];

      for (auto &Query : I->second) {
        Query->setDefinition(Name, Sym);
        PendingFinalization.push_back(Query);
      }

      // Clear the PendingResolution list for this symbol.
      (*MaterializationInfoItr)->PendingResolution.erase(I);
    }
  }
}

void VSO::SymbolTableEntry::finalize(VSO &V, SymbolStringPtr Name) {
  if (!Flags.isMaterialized()) {
    auto SymI = (*MaterializationInfoItr)->Symbols.find(Name);
    assert(SymI != (*MaterializationInfoItr)->Symbols.end() &&
           "Finalizing an unresolved symbol");
    auto Sym = SymI->second;
    (*MaterializationInfoItr)->Symbols.erase(SymI);
    auto I = (*MaterializationInfoItr)->PendingFinalization.find(Name);
    if (I != (*MaterializationInfoItr)->PendingFinalization.end()) {
      for (auto &Query : I->second)
        Query->notifySymbolFinalized();
      (*MaterializationInfoItr)->PendingFinalization.erase(I);
    }

    if (--(*MaterializationInfoItr)->SymbolsRemaining == 0)
      V.MaterializationInfos.erase(MaterializationInfoItr);

    // Destruct the iterator and re-define this entry using the final symbol
    // value.
    destroy();
    Flags = Sym.getFlags();
    Address = Sym.getAddress();
  }
  assert(Flags.isMaterialized() && "Trying to finalize not-emitted symbol");
}

void VSO::SymbolTableEntry::discard(VSO &V, SymbolStringPtr Name) {
  assert((*MaterializationInfoItr)->MU != nullptr &&
         "Can not override a symbol after it has been materialized");
  (*MaterializationInfoItr)->MU->discard(V, Name);
  --(*MaterializationInfoItr)->SymbolsRemaining;
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
      // This is an override -- discard the overridden definition and overwrite.
      I->second.discard(*this, KV.first);
      I->second = std::move(KV.second);
    } else
      Symbols.insert(std::make_pair(KV.first, std::move(KV.second)));
  }
  return Err;
}

Error VSO::defineLazy(std::unique_ptr<MaterializationUnit> MU) {

  auto NewSymbols = MU->getSymbols();

  auto MaterializationInfoItr =
      MaterializationInfos
          .insert(llvm::make_unique<MaterializationInfo>(NewSymbols.size(),
                                                         std::move(MU)))
          .first;

  Error Err = Error::success();
  for (auto &KV : NewSymbols) {
    auto I = Symbols.find(KV.first);

    auto LinkageResult = compareLinkage(
        I == Symbols.end() ? None
                           : Optional<JITSymbolFlags>(I->second.getFlags()),
        KV.second);

    // Discard weaker definitions.
    if (LinkageResult == ExistingDefinitionIsStronger) {
      (*MaterializationInfoItr)->MU->discard(*this, KV.first);
      assert((*MaterializationInfoItr)->SymbolsRemaining > 0 &&
             "Discarding non-existant symbols?");
      --(*MaterializationInfoItr)->SymbolsRemaining;
      continue;
    }

    // Report duplicate definition errors.
    if (LinkageResult == DuplicateDefinition) {
      Err = joinErrors(std::move(Err),
                       make_error<orc::DuplicateDefinition>(*KV.first));
      // Duplicate definitions are discarded, so remove the duplicates from
      // materializer.
      assert((*MaterializationInfoItr)->SymbolsRemaining > 0 &&
             "Discarding non-existant symbols?");
      --(*MaterializationInfoItr)->SymbolsRemaining;
      continue;
    }

    if (I != Symbols.end())
      I->second.replaceWith(*this, KV.first, KV.second, MaterializationInfoItr);
    else
      Symbols.emplace(std::make_pair(
          KV.first, SymbolTableEntry(KV.second, MaterializationInfoItr)));
  }

  // If we ended up overriding all definitions in this materializer then delete
  // it.
  if ((*MaterializationInfoItr)->SymbolsRemaining == 0)
    MaterializationInfos.erase(MaterializationInfoItr);

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
    I->second.finalize(*this, S);
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
  MaterializationUnitList MaterializationUnits;

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
    if (auto MU = SymI->second.query(SymI->first, Query))
      MaterializationUnits.push_back(std::move(MU));
  }

  return {std::move(MaterializationUnits), std::move(Names)};
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

    for (auto &MU : LR.MaterializationUnits)
      DispatchMaterialization(*V, std::move(MU));
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
