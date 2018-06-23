//===----- Core.cpp - Core ORC APIs (MaterializationUnit, VSO, etc.) ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"
#include "llvm/Support/Format.h"

#if LLVM_ENABLE_THREADS
#include <future>
#endif

namespace llvm {
namespace orc {

char FailedToMaterialize::ID = 0;
char SymbolsNotFound::ID = 0;

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
  if (!SymbolFlags.empty()) {
    OS << " {\"" << *SymbolFlags.begin()->first
       << "\": " << SymbolFlags.begin()->second << "}";
    for (auto &KV :
         make_range(std::next(SymbolFlags.begin()), SymbolFlags.end()))
      OS << ", {\"" << *KV.first << "\": " << KV.second << "}";
  }
  OS << " }";
  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolDependenceMap &Deps) {
  OS << "{";
  if (!Deps.empty()) {
    OS << " { " << Deps.begin()->first->getName() << ": "
       << Deps.begin()->second << " }";
    for (auto &KV : make_range(std::next(Deps.begin()), Deps.end()))
      OS << ", { " << KV.first->getName() << ": " << KV.second << " }";
  }
  OS << " }";
  return OS;
}

FailedToMaterialize::FailedToMaterialize(SymbolNameSet Symbols)
    : Symbols(std::move(Symbols)) {
  assert(!this->Symbols.empty() && "Can not fail to resolve an empty set");
}

std::error_code FailedToMaterialize::convertToErrorCode() const {
  return orcError(OrcErrorCode::UnknownORCError);
}

void FailedToMaterialize::log(raw_ostream &OS) const {
  OS << "Failed to materialize symbols: " << Symbols;
}

SymbolsNotFound::SymbolsNotFound(SymbolNameSet Symbols)
    : Symbols(std::move(Symbols)) {
  assert(!this->Symbols.empty() && "Can not fail to resolve an empty set");
}

std::error_code SymbolsNotFound::convertToErrorCode() const {
  return orcError(OrcErrorCode::UnknownORCError);
}

void SymbolsNotFound::log(raw_ostream &OS) const {
  OS << "Symbols not found: " << Symbols;
}

void ExecutionSessionBase::failQuery(AsynchronousSymbolQuery &Q, Error Err) {
  bool DeliveredError = true;
  runSessionLocked([&]() -> void {
    Q.detach();
    if (Q.canStillFail())
      Q.handleFailed(std::move(Err));
    else
      DeliveredError = false;
  });

  if (!DeliveredError)
    reportError(std::move(Err));
}

AsynchronousSymbolQuery::AsynchronousSymbolQuery(
    const SymbolNameSet &Symbols, SymbolsResolvedCallback NotifySymbolsResolved,
    SymbolsReadyCallback NotifySymbolsReady)
    : NotifySymbolsResolved(std::move(NotifySymbolsResolved)),
      NotifySymbolsReady(std::move(NotifySymbolsReady)) {
  NotYetResolvedCount = NotYetReadyCount = Symbols.size();

  for (auto &S : Symbols)
    ResolvedSymbols[S] = nullptr;

  // If the query is empty it is trivially resolved/ready.
  if (Symbols.empty()) {
    handleFullyResolved();
    handleFullyReady();
  }
}

void AsynchronousSymbolQuery::resolve(const SymbolStringPtr &Name,
                                      JITEvaluatedSymbol Sym) {
  auto I = ResolvedSymbols.find(Name);
  assert(I != ResolvedSymbols.end() &&
         "Resolving symbol outside the requested set");
  assert(I->second.getAddress() == 0 && "Redundantly resolving symbol Name");
  I->second = std::move(Sym);
  --NotYetResolvedCount;
}

void AsynchronousSymbolQuery::handleFullyResolved() {
  assert(NotYetResolvedCount == 0 && "Not fully resolved?");
  assert(NotifySymbolsResolved &&
         "NotifySymbolsResolved already called or error occurred");
  NotifySymbolsResolved(
      ResolutionResult(std::move(ResolvedSymbols), QueryRegistrations));
  NotifySymbolsResolved = SymbolsResolvedCallback();
}

void AsynchronousSymbolQuery::notifySymbolReady() {
  assert(NotYetReadyCount != 0 && "All symbols already finalized");
  --NotYetReadyCount;
}

void AsynchronousSymbolQuery::handleFullyReady() {
  assert(QueryRegistrations.empty() &&
         "Query is still registered with some symbols");
  assert(!NotifySymbolsResolved && "Resolution not applied yet");
  NotifySymbolsReady(Error::success());
  NotifySymbolsReady = SymbolsReadyCallback();
}

bool AsynchronousSymbolQuery::canStillFail() {
  return (NotifySymbolsResolved || NotifySymbolsReady);
}

void AsynchronousSymbolQuery::handleFailed(Error Err) {
  assert(QueryRegistrations.empty() && ResolvedSymbols.empty() &&
         NotYetResolvedCount == 0 && NotYetReadyCount == 0 &&
         "Query should already have been abandoned");
  if (NotifySymbolsResolved)
    NotifySymbolsResolved(std::move(Err));
  else {
    assert(NotifySymbolsReady && "Failed after both callbacks issued?");
    NotifySymbolsReady(std::move(Err));
    NotifySymbolsReady = SymbolsReadyCallback();
  }
}

void AsynchronousSymbolQuery::addQueryDependence(VSO &V, SymbolStringPtr Name) {
  bool Added = QueryRegistrations[&V].insert(std::move(Name)).second;
  (void)Added;
  assert(Added && "Duplicate dependence notification?");
}

void AsynchronousSymbolQuery::removeQueryDependence(
    VSO &V, const SymbolStringPtr &Name) {
  auto QRI = QueryRegistrations.find(&V);
  assert(QRI != QueryRegistrations.end() && "No dependencies registered for V");
  assert(QRI->second.count(Name) && "No dependency on Name in V");
  QRI->second.erase(Name);
  if (QRI->second.empty())
    QueryRegistrations.erase(QRI);
}

void AsynchronousSymbolQuery::detach() {
  ResolvedSymbols.clear();
  NotYetResolvedCount = 0;
  NotYetReadyCount = 0;
  for (auto &KV : QueryRegistrations)
    KV.first->detachQueryHelper(*this, KV.second);
  QueryRegistrations.clear();
}

MaterializationResponsibility::MaterializationResponsibility(
    VSO &V, SymbolFlagsMap SymbolFlags)
    : V(V), SymbolFlags(std::move(SymbolFlags)) {
  assert(!this->SymbolFlags.empty() && "Materializing nothing?");

#ifndef NDEBUG
  for (auto &KV : this->SymbolFlags)
    KV.second |= JITSymbolFlags::Materializing;
#endif
}

MaterializationResponsibility::~MaterializationResponsibility() {
  assert(SymbolFlags.empty() &&
         "All symbols should have been explicitly materialized or failed");
}

SymbolNameSet MaterializationResponsibility::getRequestedSymbols() {
  return V.getRequestedSymbols(SymbolFlags);
}

void MaterializationResponsibility::resolve(const SymbolMap &Symbols) {
#ifndef NDEBUG
  for (auto &KV : Symbols) {
    auto I = SymbolFlags.find(KV.first);
    assert(I != SymbolFlags.end() &&
           "Resolving symbol outside this responsibility set");
    assert(I->second.isMaterializing() && "Duplicate resolution");
    I->second &= ~JITSymbolFlags::Materializing;
    if (I->second.isWeak())
      assert(I->second == (KV.second.getFlags() | JITSymbolFlags::Weak) &&
             "Resolving symbol with incorrect flags");
    else
      assert(I->second == KV.second.getFlags() &&
             "Resolving symbol with incorrect flags");
  }
#endif

  V.resolve(Symbols);
}

void MaterializationResponsibility::finalize() {
#ifndef NDEBUG
  for (auto &KV : SymbolFlags)
    assert(!KV.second.isMaterializing() &&
           "Failed to resolve symbol before finalization");
#endif // NDEBUG

  V.finalize(SymbolFlags);
  SymbolFlags.clear();
}

Error MaterializationResponsibility::defineMaterializing(
    const SymbolFlagsMap &NewSymbolFlags) {
  // Add the given symbols to this responsibility object.
  // It's ok if we hit a duplicate here: In that case the new version will be
  // discarded, and the VSO::defineMaterializing method will return a duplicate
  // symbol error.
  for (auto &KV : NewSymbolFlags) {
    auto I = SymbolFlags.insert(KV).first;
    (void)I;
#ifndef NDEBUG
    I->second |= JITSymbolFlags::Materializing;
#endif
  }

  return V.defineMaterializing(NewSymbolFlags);
}

void MaterializationResponsibility::failMaterialization() {

  SymbolNameSet FailedSymbols;
  for (auto &KV : SymbolFlags)
    FailedSymbols.insert(KV.first);

  V.notifyFailed(FailedSymbols);
  SymbolFlags.clear();
}

void MaterializationResponsibility::delegate(
    std::unique_ptr<MaterializationUnit> MU) {
  for (auto &KV : MU->getSymbols())
    SymbolFlags.erase(KV.first);

  V.replace(std::move(MU));
}

void MaterializationResponsibility::addDependencies(
    const SymbolDependenceMap &Dependencies) {
  V.addDependencies(SymbolFlags, Dependencies);
}

AbsoluteSymbolsMaterializationUnit::AbsoluteSymbolsMaterializationUnit(
    SymbolMap Symbols)
    : MaterializationUnit(extractFlags(Symbols)), Symbols(std::move(Symbols)) {}

void AbsoluteSymbolsMaterializationUnit::materialize(
    MaterializationResponsibility R) {
  R.resolve(Symbols);
  R.finalize();
}

void AbsoluteSymbolsMaterializationUnit::discard(const VSO &V,
                                                 SymbolStringPtr Name) {
  assert(Symbols.count(Name) && "Symbol is not part of this MU");
  Symbols.erase(Name);
}

SymbolFlagsMap
AbsoluteSymbolsMaterializationUnit::extractFlags(const SymbolMap &Symbols) {
  SymbolFlagsMap Flags;
  for (const auto &KV : Symbols)
    Flags[KV.first] = KV.second.getFlags();
  return Flags;
}

Error VSO::defineMaterializing(const SymbolFlagsMap &SymbolFlags) {
  return ES.runSessionLocked([&]() -> Error {
    std::vector<SymbolMap::iterator> AddedSyms;

    for (auto &KV : SymbolFlags) {
      SymbolMap::iterator EntryItr;
      bool Added;

      auto NewFlags = KV.second;
      NewFlags |= JITSymbolFlags::Materializing;

      std::tie(EntryItr, Added) = Symbols.insert(
          std::make_pair(KV.first, JITEvaluatedSymbol(0, NewFlags)));

      if (Added)
        AddedSyms.push_back(EntryItr);
      else {
        // Remove any symbols already added.
        for (auto &SI : AddedSyms)
          Symbols.erase(SI);

        // FIXME: Return all duplicates.
        return make_error<DuplicateDefinition>(*KV.first);
      }
    }

    return Error::success();
  });
}

void VSO::replace(std::unique_ptr<MaterializationUnit> MU) {
  assert(MU != nullptr && "Can not replace with a null MaterializationUnit");

  auto MustRunMU =
      ES.runSessionLocked([&, this]() -> std::unique_ptr<MaterializationUnit> {

#ifndef NDEBUG
        for (auto &KV : MU->getSymbols()) {
          auto SymI = Symbols.find(KV.first);
          assert(SymI != Symbols.end() && "Replacing unknown symbol");
          assert(!SymI->second.getFlags().isLazy() &&
                 SymI->second.getFlags().isMaterializing() &&
                 "Can not replace symbol that is not materializing");
          assert(UnmaterializedInfos.count(KV.first) == 0 &&
                 "Symbol being replaced should have no UnmaterializedInfo");
        }
#endif // NDEBUG

        // If any symbol has pending queries against it then we need to
        // materialize MU immediately.
        for (auto &KV : MU->getSymbols()) {
          auto MII = MaterializingInfos.find(KV.first);
          if (MII != MaterializingInfos.end()) {
            if (!MII->second.PendingQueries.empty())
              return std::move(MU);
          }
        }

        // Otherwise, make MU responsible for all the symbols.
        auto UMI = std::make_shared<UnmaterializedInfo>(std::move(MU));
        for (auto &KV : UMI->MU->getSymbols()) {
          assert(!KV.second.isLazy() &&
                 "Lazy flag should be managed internally.");
          assert(!KV.second.isMaterializing() &&
                 "Materializing flags should be managed internally.");

          auto SymI = Symbols.find(KV.first);
          JITSymbolFlags ReplaceFlags = KV.second;
          ReplaceFlags |= JITSymbolFlags::Lazy;
          SymI->second = JITEvaluatedSymbol(SymI->second.getAddress(),
                                            std::move(ReplaceFlags));
          UnmaterializedInfos[KV.first] = UMI;
        }

        return nullptr;
      });

  if (MustRunMU)
    ES.dispatchMaterialization(*this, std::move(MustRunMU));
}

SymbolNameSet VSO::getRequestedSymbols(const SymbolFlagsMap &SymbolFlags) {
  return ES.runSessionLocked([&]() {
    SymbolNameSet RequestedSymbols;

    for (auto &KV : SymbolFlags) {
      assert(Symbols.count(KV.first) && "VSO does not cover this symbol?");
      assert(Symbols[KV.first].getFlags().isMaterializing() &&
             "getRequestedSymbols can only be called for materializing "
             "symbols");
      auto I = MaterializingInfos.find(KV.first);
      if (I == MaterializingInfos.end())
        continue;

      if (!I->second.PendingQueries.empty())
        RequestedSymbols.insert(KV.first);
    }

    return RequestedSymbols;
  });
}

void VSO::addDependencies(const SymbolFlagsMap &Dependants,
                          const SymbolDependenceMap &Dependencies) {
  ES.runSessionLocked([&, this]() {
    for (auto &KV : Dependants) {
      const auto &Name = KV.first;
      assert(Symbols.count(Name) && "Name not in symbol table");
      assert((Symbols[Name].getFlags().isLazy() ||
              Symbols[Name].getFlags().isMaterializing()) &&
             "Symbol is not lazy or materializing");

      auto &MI = MaterializingInfos[Name];
      assert(!MI.IsFinalized && "Can not add dependencies to finalized symbol");

      for (auto &KV : Dependencies) {
        assert(KV.first && "Null VSO in dependency?");
        auto &OtherVSO = *KV.first;
        auto &DepsOnOtherVSO = MI.UnfinalizedDependencies[&OtherVSO];

        for (auto &OtherSymbol : KV.second) {
          auto &OtherMI = OtherVSO.MaterializingInfos[OtherSymbol];

          if (OtherMI.IsFinalized)
            transferFinalizedNodeDependencies(MI, Name, OtherMI);
          else if (&OtherVSO != this || OtherSymbol != Name) {
            OtherMI.Dependants[this].insert(Name);
            DepsOnOtherVSO.insert(OtherSymbol);
          }
        }

        if (DepsOnOtherVSO.empty())
          MI.UnfinalizedDependencies.erase(&OtherVSO);
      }
    }
  });
}

void VSO::resolve(const SymbolMap &Resolved) {
  auto FullyResolvedQueries = ES.runSessionLocked([&, this]() {
    AsynchronousSymbolQuerySet FullyResolvedQueries;
    for (const auto &KV : Resolved) {
      auto &Name = KV.first;
      auto Sym = KV.second;

      assert(!Sym.getFlags().isLazy() && !Sym.getFlags().isMaterializing() &&
             "Materializing flags should be managed internally");

      auto I = Symbols.find(Name);

      assert(I != Symbols.end() && "Symbol not found");
      assert(!I->second.getFlags().isLazy() &&
             I->second.getFlags().isMaterializing() &&
             "Symbol should be materializing");
      assert(I->second.getAddress() == 0 && "Symbol has already been resolved");

      assert((Sym.getFlags() & ~JITSymbolFlags::Weak) ==
                 (JITSymbolFlags::stripTransientFlags(I->second.getFlags()) &
                  ~JITSymbolFlags::Weak) &&
             "Resolved flags should match the declared flags");

      // Once resolved, symbols can never be weak.
      JITSymbolFlags ResolvedFlags = Sym.getFlags();
      ResolvedFlags &= ~JITSymbolFlags::Weak;
      ResolvedFlags |= JITSymbolFlags::Materializing;
      I->second = JITEvaluatedSymbol(Sym.getAddress(), ResolvedFlags);

      auto &MI = MaterializingInfos[Name];
      for (auto &Q : MI.PendingQueries) {
        Q->resolve(Name, Sym);
        if (Q->isFullyResolved())
          FullyResolvedQueries.insert(Q);
      }
    }

    return FullyResolvedQueries;
  });

  for (auto &Q : FullyResolvedQueries) {
    assert(Q->isFullyResolved() && "Q not fully resolved");
    Q->handleFullyResolved();
  }
}

void VSO::finalize(const SymbolFlagsMap &Finalized) {
  auto FullyReadyQueries = ES.runSessionLocked([&, this]() {
    AsynchronousSymbolQuerySet ReadyQueries;

    for (const auto &KV : Finalized) {
      const auto &Name = KV.first;

      auto MII = MaterializingInfos.find(Name);
      assert(MII != MaterializingInfos.end() &&
             "Missing MaterializingInfo entry");

      auto &MI = MII->second;

      // For each dependant, transfer this node's unfinalized dependencies to
      // it. If the dependant node is fully finalized then notify any pending
      // queries.
      for (auto &KV : MI.Dependants) {
        auto &DependantVSO = *KV.first;
        for (auto &DependantName : KV.second) {
          auto DependantMII =
              DependantVSO.MaterializingInfos.find(DependantName);
          assert(DependantMII != DependantVSO.MaterializingInfos.end() &&
                 "Dependant should have MaterializingInfo");

          auto &DependantMI = DependantMII->second;

          // Remove the dependant's dependency on this node.
          assert(DependantMI.UnfinalizedDependencies[this].count(Name) &&
                 "Dependant does not count this symbol as a dependency?");
          DependantMI.UnfinalizedDependencies[this].erase(Name);
          if (DependantMI.UnfinalizedDependencies[this].empty())
            DependantMI.UnfinalizedDependencies.erase(this);

          // Transfer unfinalized dependencies from this node to the dependant.
          DependantVSO.transferFinalizedNodeDependencies(DependantMI,
                                                         DependantName, MI);

          // If the dependant is finalized and this node was the last of its
          // unfinalized dependencies then notify any pending queries on the
          // dependant node.
          if (DependantMI.IsFinalized &&
              DependantMI.UnfinalizedDependencies.empty()) {
            assert(DependantMI.Dependants.empty() &&
                   "Dependants should be empty by now");
            for (auto &Q : DependantMI.PendingQueries) {
              Q->notifySymbolReady();
              if (Q->isFullyReady())
                ReadyQueries.insert(Q);
              Q->removeQueryDependence(DependantVSO, DependantName);
            }

            // If this dependant node was fully finalized we can erase its
            // MaterializingInfo and update its materializing state.
            assert(DependantVSO.Symbols.count(DependantName) &&
                   "Dependant has no entry in the Symbols table");
            auto &DependantSym = DependantVSO.Symbols[DependantName];
            DependantSym.setFlags(static_cast<JITSymbolFlags::FlagNames>(
                DependantSym.getFlags() & ~JITSymbolFlags::Materializing));
            DependantVSO.MaterializingInfos.erase(DependantMII);
          }
        }
      }
      MI.Dependants.clear();
      MI.IsFinalized = true;

      if (MI.UnfinalizedDependencies.empty()) {
        for (auto &Q : MI.PendingQueries) {
          Q->notifySymbolReady();
          if (Q->isFullyReady())
            ReadyQueries.insert(Q);
          Q->removeQueryDependence(*this, Name);
        }
        assert(Symbols.count(Name) &&
               "Symbol has no entry in the Symbols table");
        auto &Sym = Symbols[Name];
        Sym.setFlags(static_cast<JITSymbolFlags::FlagNames>(
            Sym.getFlags() & ~JITSymbolFlags::Materializing));
        MaterializingInfos.erase(MII);
      }
    }

    return ReadyQueries;
  });

  for (auto &Q : FullyReadyQueries) {
    assert(Q->isFullyReady() && "Q is not fully ready");
    Q->handleFullyReady();
  }
}

void VSO::notifyFailed(const SymbolNameSet &FailedSymbols) {

  // FIXME: This should fail any transitively dependant symbols too.

  auto FailedQueriesToNotify = ES.runSessionLocked([&, this]() {
    AsynchronousSymbolQuerySet FailedQueries;

    for (auto &Name : FailedSymbols) {
      auto I = Symbols.find(Name);
      assert(I != Symbols.end() && "Symbol not present in this VSO");
      Symbols.erase(I);

      auto MII = MaterializingInfos.find(Name);

      // If we have not created a MaterializingInfo for this symbol yet then
      // there is nobody to notify.
      if (MII == MaterializingInfos.end())
        continue;

      // Copy all the queries to the FailedQueries list, then abandon them.
      // This has to be a copy, and the copy has to come before the abandon
      // operation: Each Q.detach() call will reach back into this
      // PendingQueries list to remove Q.
      for (auto &Q : MII->second.PendingQueries)
        FailedQueries.insert(Q);

      for (auto &Q : FailedQueries)
        Q->detach();

      assert(MII->second.PendingQueries.empty() &&
             "Queries remain after symbol was failed");

      MaterializingInfos.erase(MII);
    }

    return FailedQueries;
  });

  for (auto &Q : FailedQueriesToNotify)
    Q->handleFailed(make_error<FailedToMaterialize>(FailedSymbols));
}

SymbolNameSet VSO::lookupFlags(SymbolFlagsMap &Flags,
                               const SymbolNameSet &Names) {
  return ES.runSessionLocked([&, this]() {
    auto Unresolved = lookupFlagsImpl(Flags, Names);
    if (FallbackDefinitionGenerator && !Unresolved.empty()) {
      auto FallbackDefs = FallbackDefinitionGenerator(*this, Unresolved);
      if (!FallbackDefs.empty()) {
        auto Unresolved2 = lookupFlagsImpl(Flags, FallbackDefs);
        (void)Unresolved2;
        assert(Unresolved2.empty() &&
               "All fallback defs should have been found by lookupFlagsImpl");
        for (auto &D : FallbackDefs)
          Unresolved.erase(D);
      }
    };
    return Unresolved;
  });
}

SymbolNameSet VSO::lookupFlagsImpl(SymbolFlagsMap &Flags,
                                   const SymbolNameSet &Names) {
  SymbolNameSet Unresolved;

  for (auto &Name : Names) {
    auto I = Symbols.find(Name);

    if (I == Symbols.end()) {
      Unresolved.insert(Name);
      continue;
    }

    assert(!Flags.count(Name) && "Symbol already present in Flags map");
    Flags[Name] = JITSymbolFlags::stripTransientFlags(I->second.getFlags());
  }

  return Unresolved;
}

SymbolNameSet VSO::lookup(std::shared_ptr<AsynchronousSymbolQuery> Q,
                          SymbolNameSet Names) {
  assert(Q && "Query can not be null");

  LookupImplActionFlags ActionFlags = None;
  std::vector<std::unique_ptr<MaterializationUnit>> MUs;

  SymbolNameSet Unresolved = std::move(Names);
  ES.runSessionLocked([&, this]() {
    ActionFlags = lookupImpl(Q, MUs, Unresolved);
    if (FallbackDefinitionGenerator && !Unresolved.empty()) {
      assert(ActionFlags == None &&
             "ActionFlags set but unresolved symbols remain?");
      auto FallbackDefs = FallbackDefinitionGenerator(*this, Unresolved);
      if (!FallbackDefs.empty()) {
        for (auto &D : FallbackDefs)
          Unresolved.erase(D);
        ActionFlags = lookupImpl(Q, MUs, FallbackDefs);
        assert(FallbackDefs.empty() &&
               "All fallback defs should have been found by lookupImpl");
      }
    }
  });

  assert((MUs.empty() || ActionFlags == None) &&
         "If action flags are set, there should be no work to do (so no MUs)");

  if (ActionFlags & NotifyFullyResolved)
    Q->handleFullyResolved();

  if (ActionFlags & NotifyFullyReady)
    Q->handleFullyReady();

  // Dispatch any required MaterializationUnits for materialization.
  for (auto &MU : MUs)
    ES.dispatchMaterialization(*this, std::move(MU));

  return Unresolved;
}

VSO::LookupImplActionFlags
VSO::lookupImpl(std::shared_ptr<AsynchronousSymbolQuery> &Q,
                std::vector<std::unique_ptr<MaterializationUnit>> &MUs,
                SymbolNameSet &Unresolved) {
  LookupImplActionFlags ActionFlags = None;

  for (auto I = Unresolved.begin(), E = Unresolved.end(); I != E;) {
    auto TmpI = I++;
    auto Name = *TmpI;

    // Search for the name in Symbols. Skip it if not found.
    auto SymI = Symbols.find(Name);
    if (SymI == Symbols.end())
      continue;

    // If we found Name in V, remove it frome the Unresolved set and add it
    // to the dependencies set.
    Unresolved.erase(TmpI);

    // If the symbol has an address then resolve it.
    if (SymI->second.getAddress() != 0) {
      Q->resolve(Name, SymI->second);
      if (Q->isFullyResolved())
        ActionFlags |= NotifyFullyResolved;
    }

    // If the symbol is lazy, get the MaterialiaztionUnit for it.
    if (SymI->second.getFlags().isLazy()) {
      assert(SymI->second.getAddress() == 0 &&
             "Lazy symbol should not have a resolved address");
      assert(!SymI->second.getFlags().isMaterializing() &&
             "Materializing and lazy should not both be set");
      auto UMII = UnmaterializedInfos.find(Name);
      assert(UMII != UnmaterializedInfos.end() &&
             "Lazy symbol should have UnmaterializedInfo");
      auto MU = std::move(UMII->second->MU);
      assert(MU != nullptr && "Materializer should not be null");

      // Kick all symbols associated with this MaterializationUnit into
      // materializing state.
      for (auto &KV : MU->getSymbols()) {
        auto SymK = Symbols.find(KV.first);
        auto Flags = SymK->second.getFlags();
        Flags &= ~JITSymbolFlags::Lazy;
        Flags |= JITSymbolFlags::Materializing;
        SymK->second.setFlags(Flags);
        UnmaterializedInfos.erase(KV.first);
      }

      // Add MU to the list of MaterializationUnits to be materialized.
      MUs.push_back(std::move(MU));
    } else if (!SymI->second.getFlags().isMaterializing()) {
      // The symbol is neither lazy nor materializing. Finalize it and
      // continue.
      Q->notifySymbolReady();
      if (Q->isFullyReady())
        ActionFlags |= NotifyFullyReady;
      continue;
    }

    // Add the query to the PendingQueries list.
    assert(SymI->second.getFlags().isMaterializing() &&
           "By this line the symbol should be materializing");
    auto &MI = MaterializingInfos[Name];
    MI.PendingQueries.push_back(Q);
    Q->addQueryDependence(*this, Name);
  }

  return ActionFlags;
}

void VSO::dump(raw_ostream &OS) {
  ES.runSessionLocked([&, this]() {
    OS << "VSO \"" << VSOName
       << "\" (ES: " << format("0x%016x", reinterpret_cast<uintptr_t>(&ES))
       << "):\n"
       << "Symbol table:\n";

    for (auto &KV : Symbols) {
      OS << "    \"" << *KV.first
         << "\": " << format("0x%016x", KV.second.getAddress());
      if (KV.second.getFlags().isLazy() ||
          KV.second.getFlags().isMaterializing()) {
        OS << " (";
        if (KV.second.getFlags().isLazy()) {
          auto I = UnmaterializedInfos.find(KV.first);
          assert(I != UnmaterializedInfos.end() &&
                 "Lazy symbol should have UnmaterializedInfo");
          OS << " Lazy (MU=" << I->second->MU.get() << ")";
        }
        if (KV.second.getFlags().isMaterializing())
          OS << " Materializing";
        OS << " )\n";
      } else
        OS << "\n";
    }

    if (!MaterializingInfos.empty())
      OS << "  MaterializingInfos entries:\n";
    for (auto &KV : MaterializingInfos) {
      OS << "    \"" << *KV.first << "\":\n"
         << "      IsFinalized = " << (KV.second.IsFinalized ? "true" : "false")
         << "\n"
         << "      " << KV.second.PendingQueries.size()
         << " pending queries: { ";
      for (auto &Q : KV.second.PendingQueries)
        OS << Q.get() << " ";
      OS << "}\n      Dependants:\n";
      for (auto &KV2 : KV.second.Dependants)
        OS << "        " << KV2.first->getName() << ": " << KV2.second << "\n";
      OS << "      Unfinalized Dependencies:\n";
      for (auto &KV2 : KV.second.UnfinalizedDependencies)
        OS << "        " << KV2.first->getName() << ": " << KV2.second << "\n";
    }
  });
}

Error VSO::defineImpl(MaterializationUnit &MU) {
  SymbolNameSet Duplicates;
  SymbolNameSet MUDefsOverridden;
  std::vector<SymbolMap::iterator> ExistingDefsOverridden;
  for (auto &KV : MU.getSymbols()) {
    assert(!KV.second.isLazy() && "Lazy flag should be managed internally.");
    assert(!KV.second.isMaterializing() &&
           "Materializing flags should be managed internally.");

    SymbolMap::iterator EntryItr;
    bool Added;

    auto NewFlags = KV.second;
    NewFlags |= JITSymbolFlags::Lazy;

    std::tie(EntryItr, Added) = Symbols.insert(
        std::make_pair(KV.first, JITEvaluatedSymbol(0, NewFlags)));

    if (!Added) {
      if (KV.second.isStrong()) {
        if (EntryItr->second.getFlags().isStrong() ||
            (EntryItr->second.getFlags() & JITSymbolFlags::Materializing))
          Duplicates.insert(KV.first);
        else
          ExistingDefsOverridden.push_back(EntryItr);
      } else
        MUDefsOverridden.insert(KV.first);
    }
  }

  if (!Duplicates.empty()) {
    // We need to remove the symbols we added.
    for (auto &KV : MU.getSymbols()) {
      if (Duplicates.count(KV.first))
        continue;

      bool Found = false;
      for (const auto &I : ExistingDefsOverridden)
        if (I->first == KV.first)
          Found = true;

      if (!Found)
        Symbols.erase(KV.first);
    }

    // FIXME: Return all duplicates.
    return make_error<DuplicateDefinition>(**Duplicates.begin());
  }

  // Update flags on existing defs and call discard on their materializers.
  for (auto &ExistingDefItr : ExistingDefsOverridden) {
    assert(ExistingDefItr->second.getFlags().isLazy() &&
           !ExistingDefItr->second.getFlags().isMaterializing() &&
           "Overridden existing def should be in the Lazy state");

    auto UMII = UnmaterializedInfos.find(ExistingDefItr->first);
    assert(UMII != UnmaterializedInfos.end() &&
           "Overridden existing def should have an UnmaterializedInfo");

    UMII->second->MU->doDiscard(*this, ExistingDefItr->first);
  }

  // Discard overridden symbols povided by MU.
  for (auto &Sym : MUDefsOverridden)
    MU.doDiscard(*this, Sym);

  return Error::success();
}

void VSO::detachQueryHelper(AsynchronousSymbolQuery &Q,
                            const SymbolNameSet &QuerySymbols) {
  for (auto &QuerySymbol : QuerySymbols) {
    assert(MaterializingInfos.count(QuerySymbol) &&
           "QuerySymbol does not have MaterializingInfo");
    auto &MI = MaterializingInfos[QuerySymbol];

    auto IdenticalQuery =
        [&](const std::shared_ptr<AsynchronousSymbolQuery> &R) {
          return R.get() == &Q;
        };

    auto I = std::find_if(MI.PendingQueries.begin(), MI.PendingQueries.end(),
                          IdenticalQuery);
    assert(I != MI.PendingQueries.end() &&
           "Query Q should be in the PendingQueries list for QuerySymbol");
    MI.PendingQueries.erase(I);
  }
}

void VSO::transferFinalizedNodeDependencies(
    MaterializingInfo &DependantMI, const SymbolStringPtr &DependantName,
    MaterializingInfo &FinalizedMI) {
  for (auto &KV : FinalizedMI.UnfinalizedDependencies) {
    auto &DependencyVSO = *KV.first;
    SymbolNameSet *UnfinalizedDependenciesOnDependencyVSO = nullptr;

    for (auto &DependencyName : KV.second) {
      auto &DependencyMI = DependencyVSO.MaterializingInfos[DependencyName];

      // Do not add self dependencies.
      if (&DependencyMI == &DependantMI)
        continue;

      // If we haven't looked up the dependencies for DependencyVSO yet, do it
      // now and cache the result.
      if (!UnfinalizedDependenciesOnDependencyVSO)
        UnfinalizedDependenciesOnDependencyVSO =
            &DependantMI.UnfinalizedDependencies[&DependencyVSO];

      DependencyMI.Dependants[this].insert(DependantName);
      UnfinalizedDependenciesOnDependencyVSO->insert(DependencyName);
    }
  }
}

VSO &ExecutionSession::createVSO(std::string Name) {
  return runSessionLocked([&, this]() -> VSO & {
      VSOs.push_back(std::unique_ptr<VSO>(new VSO(*this, std::move(Name))));
    return *VSOs.back();
  });
}

Expected<SymbolMap> blockingLookup(ExecutionSessionBase &ES,
                                   AsynchronousLookupFunction AsyncLookup,
                                   SymbolNameSet Names, bool WaitUntilReady,
                                   MaterializationResponsibility *MR) {

#if LLVM_ENABLE_THREADS
  // In the threaded case we use promises to return the results.
  std::promise<SymbolMap> PromisedResult;
  std::mutex ErrMutex;
  Error ResolutionError = Error::success();
  std::promise<void> PromisedReady;
  Error ReadyError = Error::success();
  auto OnResolve =
      [&](Expected<AsynchronousSymbolQuery::ResolutionResult> Result) {
        if (Result) {
          if (MR)
            MR->addDependencies(Result->Dependencies);
          PromisedResult.set_value(std::move(Result->Symbols));
        } else {
          {
            ErrorAsOutParameter _(&ResolutionError);
            std::lock_guard<std::mutex> Lock(ErrMutex);
            ResolutionError = Result.takeError();
          }
          PromisedResult.set_value(SymbolMap());
        }
      };

  std::function<void(Error)> OnReady;
  if (WaitUntilReady) {
    OnReady = [&](Error Err) {
      if (Err) {
        ErrorAsOutParameter _(&ReadyError);
        std::lock_guard<std::mutex> Lock(ErrMutex);
        ReadyError = std::move(Err);
      }
      PromisedReady.set_value();
    };
  } else {
    OnReady = [&](Error Err) {
      if (Err)
        ES.reportError(std::move(Err));
    };
  }

#else
  SymbolMap Result;
  Error ResolutionError = Error::success();
  Error ReadyError = Error::success();

  auto OnResolve = [&](Expected<AsynchronousSymbolQuery::ResolutionResult> R) {
    ErrorAsOutParameter _(&ResolutionError);
    if (R) {
      if (MR)
        MR->addDependencies(R->Dependencies);
      Result = std::move(R->Symbols);
    } else
      ResolutionError = R.takeError();
  };

  std::function<void(Error)> OnReady;
  if (WaitUntilReady) {
    OnReady = [&](Error Err) {
      ErrorAsOutParameter _(&ReadyError);
      if (Err)
        ReadyError = std::move(Err);
    };
  } else {
    OnReady = [&](Error Err) {
      if (Err)
        ES.reportError(std::move(Err));
    };
  }
#endif

  auto Query = std::make_shared<AsynchronousSymbolQuery>(
      Names, std::move(OnResolve), std::move(OnReady));

  SymbolNameSet UnresolvedSymbols = AsyncLookup(Query, std::move(Names));

  // If there are unresolved symbols then the query will never return.
  // Fail it with ES.failQuery.
  if (!UnresolvedSymbols.empty())
    ES.failQuery(*Query,
                 make_error<SymbolsNotFound>(std::move(UnresolvedSymbols)));

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

  if (WaitUntilReady) {
    auto ReadyFuture = PromisedReady.get_future();
    ReadyFuture.get();

    {
      std::lock_guard<std::mutex> Lock(ErrMutex);
      if (ReadyError)
        return std::move(ReadyError);
    }
  } else
    cantFail(std::move(ReadyError));

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

Expected<SymbolMap> lookup(const VSO::VSOList &VSOs, SymbolNameSet Names) {

  if (VSOs.empty())
    return SymbolMap();

  auto &ES = (*VSOs.begin())->getExecutionSession();

  auto LookupFn = [&](std::shared_ptr<AsynchronousSymbolQuery> Q,
                      SymbolNameSet Unresolved) {
    for (auto *V : VSOs) {
      assert(V && "VSOs entries must not be null");
      if (Unresolved.empty())
        break;
      Unresolved = V->lookup(Q, std::move(Unresolved));
    }
    return Unresolved;
  };

  return blockingLookup(ES, std::move(LookupFn), Names, true);
}

/// Look up a symbol by searching a list of VSOs.
Expected<JITEvaluatedSymbol> lookup(const VSO::VSOList &VSOs,
                                    SymbolStringPtr Name) {
  SymbolNameSet Names({Name});
  if (auto ResultMap = lookup(VSOs, std::move(Names))) {
    assert(ResultMap->size() == 1 && "Unexpected number of results");
    assert(ResultMap->count(Name) && "Missing result for symbol");
    return std::move(ResultMap->begin()->second);
  } else
    return ResultMap.takeError();
}

} // End namespace orc.
} // End namespace llvm.
