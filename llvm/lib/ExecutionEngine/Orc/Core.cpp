//===--- Core.cpp - Core ORC APIs (MaterializationUnit, JITDylib, etc.) ---===//
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
#include "llvm/IR/Mangler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#if LLVM_ENABLE_THREADS
#include <future>
#endif

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

char FailedToMaterialize::ID = 0;
char SymbolsNotFound::ID = 0;

RegisterDependenciesFunction NoDependenciesToRegister =
    RegisterDependenciesFunction();

void MaterializationUnit::anchor() {}

raw_ostream &operator<<(raw_ostream &OS, const JITSymbolFlags &Flags) {
  if (Flags.isCallable())
    OS << "[Callable]";
  else
    OS << "[Data]";
  if (Flags.isWeak())
    OS << "[Weak]";
  else if (Flags.isCommon())
    OS << "[Common]";

  if (!Flags.isExported())
    OS << "[Hidden]";

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

raw_ostream &operator<<(raw_ostream &OS, const JITDylibList &JDs) {
  OS << "[";
  if (!JDs.empty()) {
    assert(JDs.front() && "JITDylibList entries must not be null");
    OS << " " << JDs.front()->getName();
    for (auto *JD : make_range(std::next(JDs.begin()), JDs.end())) {
      assert(JD && "JITDylibList entries must not be null");
      OS << ", " << JD->getName();
    }
  }
  OS << " ]";
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

AsynchronousSymbolQuery::AsynchronousSymbolQuery(
    const SymbolNameSet &Symbols, SymbolsResolvedCallback NotifySymbolsResolved,
    SymbolsReadyCallback NotifySymbolsReady)
    : NotifySymbolsResolved(std::move(NotifySymbolsResolved)),
      NotifySymbolsReady(std::move(NotifySymbolsReady)) {
  NotYetResolvedCount = NotYetReadyCount = Symbols.size();

  for (auto &S : Symbols)
    ResolvedSymbols[S] = nullptr;
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

  if (!NotifySymbolsResolved) {
    // handleFullyResolved may be called by handleFullyReady (see comments in
    // that method), in which case this is a no-op, so bail out.
    assert(!NotifySymbolsReady &&
           "NotifySymbolsResolved already called or an error occurred");
    return;
  }

  auto TmpNotifySymbolsResolved = std::move(NotifySymbolsResolved);
  NotifySymbolsResolved = SymbolsResolvedCallback();
  TmpNotifySymbolsResolved(std::move(ResolvedSymbols));
}

void AsynchronousSymbolQuery::notifySymbolReady() {
  assert(NotYetReadyCount != 0 && "All symbols already emitted");
  --NotYetReadyCount;
}

void AsynchronousSymbolQuery::handleFullyReady() {
  assert(NotifySymbolsReady &&
         "NotifySymbolsReady already called or an error occurred");

  auto TmpNotifySymbolsReady = std::move(NotifySymbolsReady);
  NotifySymbolsReady = SymbolsReadyCallback();

  if (NotYetResolvedCount == 0 && NotifySymbolsResolved) {
    // The NotifyResolved callback of one query must have caused this query to
    // become ready (i.e. there is still a handleFullyResolved callback waiting
    // to be made back up the stack). Fold the handleFullyResolved call into
    // this one before proceeding. This will cause the call further up the
    // stack to become a no-op.
    handleFullyResolved();
  }

  assert(QueryRegistrations.empty() &&
         "Query is still registered with some symbols");
  assert(!NotifySymbolsResolved && "Resolution not applied yet");
  TmpNotifySymbolsReady(Error::success());
}

bool AsynchronousSymbolQuery::canStillFail() {
  return (NotifySymbolsResolved || NotifySymbolsReady);
}

void AsynchronousSymbolQuery::handleFailed(Error Err) {
  assert(QueryRegistrations.empty() && ResolvedSymbols.empty() &&
         NotYetResolvedCount == 0 && NotYetReadyCount == 0 &&
         "Query should already have been abandoned");
  if (NotifySymbolsResolved) {
    NotifySymbolsResolved(std::move(Err));
    NotifySymbolsResolved = SymbolsResolvedCallback();
  } else {
    assert(NotifySymbolsReady && "Failed after both callbacks issued?");
    NotifySymbolsReady(std::move(Err));
  }
  NotifySymbolsReady = SymbolsReadyCallback();
}

void AsynchronousSymbolQuery::addQueryDependence(JITDylib &JD,
                                                 SymbolStringPtr Name) {
  bool Added = QueryRegistrations[&JD].insert(std::move(Name)).second;
  (void)Added;
  assert(Added && "Duplicate dependence notification?");
}

void AsynchronousSymbolQuery::removeQueryDependence(
    JITDylib &JD, const SymbolStringPtr &Name) {
  auto QRI = QueryRegistrations.find(&JD);
  assert(QRI != QueryRegistrations.end() &&
         "No dependencies registered for JD");
  assert(QRI->second.count(Name) && "No dependency on Name in JD");
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
    JITDylib &JD, SymbolFlagsMap SymbolFlags)
    : JD(JD), SymbolFlags(std::move(SymbolFlags)) {
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
  return JD.getRequestedSymbols(SymbolFlags);
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

  JD.resolve(Symbols);
}

void MaterializationResponsibility::emit() {
#ifndef NDEBUG
  for (auto &KV : SymbolFlags)
    assert(!KV.second.isMaterializing() &&
           "Failed to resolve symbol before emission");
#endif // NDEBUG

  JD.emit(SymbolFlags);
  SymbolFlags.clear();
}

Error MaterializationResponsibility::defineMaterializing(
    const SymbolFlagsMap &NewSymbolFlags) {
  // Add the given symbols to this responsibility object.
  // It's ok if we hit a duplicate here: In that case the new version will be
  // discarded, and the JITDylib::defineMaterializing method will return a
  // duplicate symbol error.
  for (auto &KV : NewSymbolFlags) {
    auto I = SymbolFlags.insert(KV).first;
    (void)I;
#ifndef NDEBUG
    I->second |= JITSymbolFlags::Materializing;
#endif
  }

  return JD.defineMaterializing(NewSymbolFlags);
}

void MaterializationResponsibility::failMaterialization() {

  SymbolNameSet FailedSymbols;
  for (auto &KV : SymbolFlags)
    FailedSymbols.insert(KV.first);

  JD.notifyFailed(FailedSymbols);
  SymbolFlags.clear();
}

void MaterializationResponsibility::replace(
    std::unique_ptr<MaterializationUnit> MU) {
  for (auto &KV : MU->getSymbols())
    SymbolFlags.erase(KV.first);

  LLVM_DEBUG(dbgs() << "For " << JD.getName() << " replacing " << MU->getSymbols() << "\n");

  JD.replace(std::move(MU));
}

MaterializationResponsibility
MaterializationResponsibility::delegate(const SymbolNameSet &Symbols) {
  SymbolFlagsMap DelegatedFlags;

  for (auto &Name : Symbols) {
    auto I = SymbolFlags.find(Name);
    assert(I != SymbolFlags.end() &&
           "Symbol is not tracked by this MaterializationResponsibility "
           "instance");

    DelegatedFlags[Name] = std::move(I->second);
    SymbolFlags.erase(I);
  }

  return MaterializationResponsibility(JD, std::move(DelegatedFlags));
}

void MaterializationResponsibility::addDependencies(
    const SymbolStringPtr &Name, const SymbolDependenceMap &Dependencies) {
  assert(SymbolFlags.count(Name) &&
         "Symbol not covered by this MaterializationResponsibility instance");
  JD.addDependencies(Name, Dependencies);
}

void MaterializationResponsibility::addDependenciesForAll(
    const SymbolDependenceMap &Dependencies) {
  for (auto &KV : SymbolFlags)
    JD.addDependencies(KV.first, Dependencies);
}

AbsoluteSymbolsMaterializationUnit::AbsoluteSymbolsMaterializationUnit(
    SymbolMap Symbols)
    : MaterializationUnit(extractFlags(Symbols)), Symbols(std::move(Symbols)) {}

void AbsoluteSymbolsMaterializationUnit::materialize(
    MaterializationResponsibility R) {
  R.resolve(Symbols);
  R.emit();
}

void AbsoluteSymbolsMaterializationUnit::discard(const JITDylib &JD,
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

ReExportsMaterializationUnit::ReExportsMaterializationUnit(
    JITDylib *SourceJD, SymbolAliasMap Aliases)
    : MaterializationUnit(extractFlags(Aliases)), SourceJD(SourceJD),
      Aliases(std::move(Aliases)) {}

void ReExportsMaterializationUnit::materialize(
    MaterializationResponsibility R) {

  auto &ES = R.getTargetJITDylib().getExecutionSession();
  JITDylib &TgtJD = R.getTargetJITDylib();
  JITDylib &SrcJD = SourceJD ? *SourceJD : TgtJD;

  // Find the set of requested aliases and aliasees. Return any unrequested
  // aliases back to the JITDylib so as to not prematurely materialize any
  // aliasees.
  auto RequestedSymbols = R.getRequestedSymbols();
  SymbolAliasMap RequestedAliases;

  for (auto &Name : RequestedSymbols) {
    auto I = Aliases.find(Name);
    assert(I != Aliases.end() && "Symbol not found in aliases map?");
    RequestedAliases[Name] = std::move(I->second);
    Aliases.erase(I);
  }

  if (!Aliases.empty()) {
    if (SourceJD)
      R.replace(reexports(*SourceJD, std::move(Aliases)));
    else
      R.replace(symbolAliases(std::move(Aliases)));
  }

  // The OnResolveInfo struct will hold the aliases and responsibilty for each
  // query in the list.
  struct OnResolveInfo {
    OnResolveInfo(MaterializationResponsibility R, SymbolAliasMap Aliases)
        : R(std::move(R)), Aliases(std::move(Aliases)) {}

    MaterializationResponsibility R;
    SymbolAliasMap Aliases;
  };

  // Build a list of queries to issue. In each round we build the largest set of
  // aliases that we can resolve without encountering a chain definition of the
  // form Foo -> Bar, Bar -> Baz. Such a form would deadlock as the query would
  // be waitin on a symbol that it itself had to resolve. Usually this will just
  // involve one round and a single query.

  std::vector<std::pair<SymbolNameSet, std::shared_ptr<OnResolveInfo>>>
      QueryInfos;
  while (!RequestedAliases.empty()) {
    SymbolNameSet ResponsibilitySymbols;
    SymbolNameSet QuerySymbols;
    SymbolAliasMap QueryAliases;

    for (auto I = RequestedAliases.begin(), E = RequestedAliases.end();
         I != E;) {
      auto Tmp = I++;

      // Chain detected. Skip this symbol for this round.
      if (&SrcJD == &TgtJD && (QueryAliases.count(Tmp->second.Aliasee) ||
                               RequestedAliases.count(Tmp->second.Aliasee)))
        continue;

      ResponsibilitySymbols.insert(Tmp->first);
      QuerySymbols.insert(Tmp->second.Aliasee);
      QueryAliases[Tmp->first] = std::move(Tmp->second);
      RequestedAliases.erase(Tmp);
    }
    assert(!QuerySymbols.empty() && "Alias cycle detected!");

    auto QueryInfo = std::make_shared<OnResolveInfo>(
        R.delegate(ResponsibilitySymbols), std::move(QueryAliases));
    QueryInfos.push_back(
        make_pair(std::move(QuerySymbols), std::move(QueryInfo)));
  }

  // Issue the queries.
  while (!QueryInfos.empty()) {
    auto QuerySymbols = std::move(QueryInfos.back().first);
    auto QueryInfo = std::move(QueryInfos.back().second);

    QueryInfos.pop_back();

    auto RegisterDependencies = [QueryInfo,
                                 &SrcJD](const SymbolDependenceMap &Deps) {
      // If there were no materializing symbols, just bail out.
      if (Deps.empty())
        return;

      // Otherwise the only deps should be on SrcJD.
      assert(Deps.size() == 1 && Deps.count(&SrcJD) &&
             "Unexpected dependencies for reexports");

      auto &SrcJDDeps = Deps.find(&SrcJD)->second;
      SymbolDependenceMap PerAliasDepsMap;
      auto &PerAliasDeps = PerAliasDepsMap[&SrcJD];

      for (auto &KV : QueryInfo->Aliases)
        if (SrcJDDeps.count(KV.second.Aliasee)) {
          PerAliasDeps = {KV.second.Aliasee};
          QueryInfo->R.addDependencies(KV.first, PerAliasDepsMap);
        }
    };

    auto OnResolve = [QueryInfo](Expected<SymbolMap> Result) {
      if (Result) {
        SymbolMap ResolutionMap;
        for (auto &KV : QueryInfo->Aliases) {
          assert(Result->count(KV.second.Aliasee) &&
                 "Result map missing entry?");
          ResolutionMap[KV.first] = JITEvaluatedSymbol(
              (*Result)[KV.second.Aliasee].getAddress(), KV.second.AliasFlags);
        }
        QueryInfo->R.resolve(ResolutionMap);
        QueryInfo->R.emit();
      } else {
        auto &ES = QueryInfo->R.getTargetJITDylib().getExecutionSession();
        ES.reportError(Result.takeError());
        QueryInfo->R.failMaterialization();
      }
    };

    auto OnReady = [&ES](Error Err) { ES.reportError(std::move(Err)); };

    ES.lookup({&SrcJD}, QuerySymbols, std::move(OnResolve), std::move(OnReady),
              std::move(RegisterDependencies));
  }
}

void ReExportsMaterializationUnit::discard(const JITDylib &JD,
                                           SymbolStringPtr Name) {
  assert(Aliases.count(Name) &&
         "Symbol not covered by this MaterializationUnit");
  Aliases.erase(Name);
}

SymbolFlagsMap
ReExportsMaterializationUnit::extractFlags(const SymbolAliasMap &Aliases) {
  SymbolFlagsMap SymbolFlags;
  for (auto &KV : Aliases)
    SymbolFlags[KV.first] = KV.second.AliasFlags;

  return SymbolFlags;
}

Expected<SymbolAliasMap>
buildSimpleReexportsAliasMap(JITDylib &SourceJD, const SymbolNameSet &Symbols) {
  auto Flags = SourceJD.lookupFlags(Symbols);

  if (Flags.size() != Symbols.size()) {
    SymbolNameSet Unresolved = Symbols;
    for (auto &KV : Flags)
      Unresolved.erase(KV.first);
    return make_error<SymbolsNotFound>(std::move(Unresolved));
  }

  SymbolAliasMap Result;
  for (auto &Name : Symbols) {
    assert(Flags.count(Name) && "Missing entry in flags map");
    Result[Name] = SymbolAliasMapEntry(Name, Flags[Name]);
  }

  return Result;
}

ReexportsFallbackDefinitionGenerator::ReexportsFallbackDefinitionGenerator(
    JITDylib &BackingJD, SymbolPredicate Allow)
    : BackingJD(BackingJD), Allow(std::move(Allow)) {}

SymbolNameSet ReexportsFallbackDefinitionGenerator::
operator()(JITDylib &JD, const SymbolNameSet &Names) {
  orc::SymbolNameSet Added;
  orc::SymbolAliasMap AliasMap;

  auto Flags = BackingJD.lookupFlags(Names);

  for (auto &KV : Flags) {
    if (!Allow(KV.first))
      continue;
    AliasMap[KV.first] = SymbolAliasMapEntry(KV.first, KV.second);
    Added.insert(KV.first);
  }

  if (!Added.empty())
    cantFail(JD.define(reexports(BackingJD, AliasMap)));

  return Added;
}

Error JITDylib::defineMaterializing(const SymbolFlagsMap &SymbolFlags) {
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

void JITDylib::replace(std::unique_ptr<MaterializationUnit> MU) {
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

SymbolNameSet JITDylib::getRequestedSymbols(const SymbolFlagsMap &SymbolFlags) {
  return ES.runSessionLocked([&]() {
    SymbolNameSet RequestedSymbols;

    for (auto &KV : SymbolFlags) {
      assert(Symbols.count(KV.first) && "JITDylib does not cover this symbol?");
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

void JITDylib::addDependencies(const SymbolStringPtr &Name,
                               const SymbolDependenceMap &Dependencies) {
  assert(Symbols.count(Name) && "Name not in symbol table");
  assert((Symbols[Name].getFlags().isLazy() ||
          Symbols[Name].getFlags().isMaterializing()) &&
         "Symbol is not lazy or materializing");

  auto &MI = MaterializingInfos[Name];
  assert(!MI.IsEmitted && "Can not add dependencies to an emitted symbol");

  for (auto &KV : Dependencies) {
    assert(KV.first && "Null JITDylib in dependency?");
    auto &OtherJITDylib = *KV.first;
    auto &DepsOnOtherJITDylib = MI.UnemittedDependencies[&OtherJITDylib];

    for (auto &OtherSymbol : KV.second) {
#ifndef NDEBUG
      // Assert that this symbol exists and has not been emitted already.
      auto SymI = OtherJITDylib.Symbols.find(OtherSymbol);
      assert(SymI != OtherJITDylib.Symbols.end() &&
             (SymI->second.getFlags().isLazy() ||
              SymI->second.getFlags().isMaterializing()) &&
             "Dependency on emitted symbol");
#endif

      auto &OtherMI = OtherJITDylib.MaterializingInfos[OtherSymbol];

      if (OtherMI.IsEmitted)
        transferEmittedNodeDependencies(MI, Name, OtherMI);
      else if (&OtherJITDylib != this || OtherSymbol != Name) {
        OtherMI.Dependants[this].insert(Name);
        DepsOnOtherJITDylib.insert(OtherSymbol);
      }
    }

    if (DepsOnOtherJITDylib.empty())
      MI.UnemittedDependencies.erase(&OtherJITDylib);
  }
}

void JITDylib::resolve(const SymbolMap &Resolved) {
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

void JITDylib::emit(const SymbolFlagsMap &Emitted) {
  auto FullyReadyQueries = ES.runSessionLocked([&, this]() {
    AsynchronousSymbolQuerySet ReadyQueries;

    for (const auto &KV : Emitted) {
      const auto &Name = KV.first;

      auto MII = MaterializingInfos.find(Name);
      assert(MII != MaterializingInfos.end() &&
             "Missing MaterializingInfo entry");

      auto &MI = MII->second;

      // For each dependant, transfer this node's emitted dependencies to
      // it. If the dependant node is ready (i.e. has no unemitted
      // dependencies) then notify any pending queries.
      for (auto &KV : MI.Dependants) {
        auto &DependantJD = *KV.first;
        for (auto &DependantName : KV.second) {
          auto DependantMII =
              DependantJD.MaterializingInfos.find(DependantName);
          assert(DependantMII != DependantJD.MaterializingInfos.end() &&
                 "Dependant should have MaterializingInfo");

          auto &DependantMI = DependantMII->second;

          // Remove the dependant's dependency on this node.
          assert(DependantMI.UnemittedDependencies[this].count(Name) &&
                 "Dependant does not count this symbol as a dependency?");
          DependantMI.UnemittedDependencies[this].erase(Name);
          if (DependantMI.UnemittedDependencies[this].empty())
            DependantMI.UnemittedDependencies.erase(this);

          // Transfer unemitted dependencies from this node to the dependant.
          DependantJD.transferEmittedNodeDependencies(DependantMI,
                                                      DependantName, MI);

          // If the dependant is emitted and this node was the last of its
          // unemitted dependencies then the dependant node is now ready, so
          // notify any pending queries on the dependant node.
          if (DependantMI.IsEmitted &&
              DependantMI.UnemittedDependencies.empty()) {
            assert(DependantMI.Dependants.empty() &&
                   "Dependants should be empty by now");
            for (auto &Q : DependantMI.PendingQueries) {
              Q->notifySymbolReady();
              if (Q->isFullyReady())
                ReadyQueries.insert(Q);
              Q->removeQueryDependence(DependantJD, DependantName);
            }

            // Since this dependant is now ready, we erase its MaterializingInfo
            // and update its materializing state.
            assert(DependantJD.Symbols.count(DependantName) &&
                   "Dependant has no entry in the Symbols table");
            auto &DependantSym = DependantJD.Symbols[DependantName];
            DependantSym.setFlags(DependantSym.getFlags() &
                                  ~JITSymbolFlags::Materializing);
            DependantJD.MaterializingInfos.erase(DependantMII);
          }
        }
      }
      MI.Dependants.clear();
      MI.IsEmitted = true;

      if (MI.UnemittedDependencies.empty()) {
        for (auto &Q : MI.PendingQueries) {
          Q->notifySymbolReady();
          if (Q->isFullyReady())
            ReadyQueries.insert(Q);
          Q->removeQueryDependence(*this, Name);
        }
        assert(Symbols.count(Name) &&
               "Symbol has no entry in the Symbols table");
        auto &Sym = Symbols[Name];
        Sym.setFlags(Sym.getFlags() & ~JITSymbolFlags::Materializing);
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

void JITDylib::notifyFailed(const SymbolNameSet &FailedSymbols) {

  // FIXME: This should fail any transitively dependant symbols too.

  auto FailedQueriesToNotify = ES.runSessionLocked([&, this]() {
    AsynchronousSymbolQuerySet FailedQueries;

    for (auto &Name : FailedSymbols) {
      auto I = Symbols.find(Name);
      assert(I != Symbols.end() && "Symbol not present in this JITDylib");
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

void JITDylib::setSearchOrder(JITDylibList NewSearchOrder,
                              bool SearchThisJITDylibFirst) {
  if (SearchThisJITDylibFirst && NewSearchOrder.front() != this)
    NewSearchOrder.insert(NewSearchOrder.begin(), this);

  ES.runSessionLocked([&]() { SearchOrder = std::move(NewSearchOrder); });
}

void JITDylib::addToSearchOrder(JITDylib &JD) {
  ES.runSessionLocked([&]() { SearchOrder.push_back(&JD); });
}

void JITDylib::replaceInSearchOrder(JITDylib &OldJD, JITDylib &NewJD) {
  ES.runSessionLocked([&]() {
    auto I = std::find(SearchOrder.begin(), SearchOrder.end(), &OldJD);

    if (I != SearchOrder.end())
      *I = &NewJD;
  });
}

void JITDylib::removeFromSearchOrder(JITDylib &JD) {
  ES.runSessionLocked([&]() {
    auto I = std::find(SearchOrder.begin(), SearchOrder.end(), &JD);
    if (I != SearchOrder.end())
      SearchOrder.erase(I);
  });
}

SymbolFlagsMap JITDylib::lookupFlags(const SymbolNameSet &Names) {
  return ES.runSessionLocked([&, this]() {
    SymbolFlagsMap Result;
    auto Unresolved = lookupFlagsImpl(Result, Names);
    if (FallbackDefinitionGenerator && !Unresolved.empty()) {
      auto FallbackDefs = FallbackDefinitionGenerator(*this, Unresolved);
      if (!FallbackDefs.empty()) {
        auto Unresolved2 = lookupFlagsImpl(Result, FallbackDefs);
        (void)Unresolved2;
        assert(Unresolved2.empty() &&
               "All fallback defs should have been found by lookupFlagsImpl");
      }
    };
    return Result;
  });
}

SymbolNameSet JITDylib::lookupFlagsImpl(SymbolFlagsMap &Flags,
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

void JITDylib::lodgeQuery(std::shared_ptr<AsynchronousSymbolQuery> &Q,
                          SymbolNameSet &Unresolved,
                          MaterializationUnitList &MUs) {
  assert(Q && "Query can not be null");

  lodgeQueryImpl(Q, Unresolved, MUs);
  if (FallbackDefinitionGenerator && !Unresolved.empty()) {
    auto FallbackDefs = FallbackDefinitionGenerator(*this, Unresolved);
    if (!FallbackDefs.empty()) {
      for (auto &D : FallbackDefs)
        Unresolved.erase(D);
      lodgeQueryImpl(Q, FallbackDefs, MUs);
      assert(FallbackDefs.empty() &&
             "All fallback defs should have been found by lookupImpl");
    }
  }
}

void JITDylib::lodgeQueryImpl(
    std::shared_ptr<AsynchronousSymbolQuery> &Q, SymbolNameSet &Unresolved,
    std::vector<std::unique_ptr<MaterializationUnit>> &MUs) {
  for (auto I = Unresolved.begin(), E = Unresolved.end(); I != E;) {
    auto TmpI = I++;
    auto Name = *TmpI;

    // Search for the name in Symbols. Skip it if not found.
    auto SymI = Symbols.find(Name);
    if (SymI == Symbols.end())
      continue;

    // If we found Name in JD, remove it frome the Unresolved set and add it
    // to the added set.
    Unresolved.erase(TmpI);

    // If the symbol has an address then resolve it.
    if (SymI->second.getAddress() != 0)
      Q->resolve(Name, SymI->second);

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

      // Move all symbols associated with this MaterializationUnit into
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
      // The symbol is neither lazy nor materializing, so it must be
      // ready. Notify the query and continue.
      Q->notifySymbolReady();
      continue;
    }

    // Add the query to the PendingQueries list.
    assert(SymI->second.getFlags().isMaterializing() &&
           "By this line the symbol should be materializing");
    auto &MI = MaterializingInfos[Name];
    MI.PendingQueries.push_back(Q);
    Q->addQueryDependence(*this, Name);
  }
}

SymbolNameSet JITDylib::legacyLookup(std::shared_ptr<AsynchronousSymbolQuery> Q,
                                     SymbolNameSet Names) {
  assert(Q && "Query can not be null");

  ES.runOutstandingMUs();

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

  // FIXME: Swap back to the old code below once RuntimeDyld works with
  //        callbacks from asynchronous queries.
  // Add MUs to the OutstandingMUs list.
  {
    std::lock_guard<std::recursive_mutex> Lock(ES.OutstandingMUsMutex);
    for (auto &MU : MUs)
      ES.OutstandingMUs.push_back(make_pair(this, std::move(MU)));
  }
  ES.runOutstandingMUs();

  // Dispatch any required MaterializationUnits for materialization.
  // for (auto &MU : MUs)
  //  ES.dispatchMaterialization(*this, std::move(MU));

  return Unresolved;
}

JITDylib::LookupImplActionFlags
JITDylib::lookupImpl(std::shared_ptr<AsynchronousSymbolQuery> &Q,
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

    // If we found Name, remove it frome the Unresolved set and add it
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
      // The symbol is neither lazy nor materializing, so it must be ready.
      // Notify the query and continue.
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

void JITDylib::dump(raw_ostream &OS) {
  ES.runSessionLocked([&, this]() {
    OS << "JITDylib \"" << JITDylibName
       << "\" (ES: " << format("0x%016x", reinterpret_cast<uintptr_t>(&ES))
       << "):\n"
       << "Symbol table:\n";

    for (auto &KV : Symbols) {
      OS << "    \"" << *KV.first << "\": ";
      if (auto Addr = KV.second.getAddress())
        OS << format("0x%016x", Addr);
      else
        OS << "<not resolved>";
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
         << "      IsEmitted = " << (KV.second.IsEmitted ? "true" : "false")
         << "\n"
         << "      " << KV.second.PendingQueries.size()
         << " pending queries: { ";
      for (auto &Q : KV.second.PendingQueries)
        OS << Q.get() << " ";
      OS << "}\n      Dependants:\n";
      for (auto &KV2 : KV.second.Dependants)
        OS << "        " << KV2.first->getName() << ": " << KV2.second << "\n";
      OS << "      Unemitted Dependencies:\n";
      for (auto &KV2 : KV.second.UnemittedDependencies)
        OS << "        " << KV2.first->getName() << ": " << KV2.second << "\n";
    }
  });
}

JITDylib::JITDylib(ExecutionSession &ES, std::string Name)
    : ES(ES), JITDylibName(std::move(Name)) {
  SearchOrder.push_back(this);
}

Error JITDylib::defineImpl(MaterializationUnit &MU) {
  SymbolNameSet Duplicates;
  SymbolNameSet MUDefsOverridden;

  struct ExistingDefOverriddenEntry {
    SymbolMap::iterator ExistingDefItr;
    JITSymbolFlags NewFlags;
  };
  std::vector<ExistingDefOverriddenEntry> ExistingDefsOverridden;

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
          ExistingDefsOverridden.push_back({EntryItr, NewFlags});
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
      for (const auto &EDO : ExistingDefsOverridden)
        if (EDO.ExistingDefItr->first == KV.first)
          Found = true;

      if (!Found)
        Symbols.erase(KV.first);
    }

    // FIXME: Return all duplicates.
    return make_error<DuplicateDefinition>(**Duplicates.begin());
  }

  // Update flags on existing defs and call discard on their materializers.
  for (auto &EDO : ExistingDefsOverridden) {
    assert(EDO.ExistingDefItr->second.getFlags().isLazy() &&
           !EDO.ExistingDefItr->second.getFlags().isMaterializing() &&
           "Overridden existing def should be in the Lazy state");

    EDO.ExistingDefItr->second.setFlags(EDO.NewFlags);

    auto UMII = UnmaterializedInfos.find(EDO.ExistingDefItr->first);
    assert(UMII != UnmaterializedInfos.end() &&
           "Overridden existing def should have an UnmaterializedInfo");

    UMII->second->MU->doDiscard(*this, EDO.ExistingDefItr->first);
  }

  // Discard overridden symbols povided by MU.
  for (auto &Sym : MUDefsOverridden)
    MU.doDiscard(*this, Sym);

  return Error::success();
}

void JITDylib::detachQueryHelper(AsynchronousSymbolQuery &Q,
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

void JITDylib::transferEmittedNodeDependencies(
    MaterializingInfo &DependantMI, const SymbolStringPtr &DependantName,
    MaterializingInfo &EmittedMI) {
  for (auto &KV : EmittedMI.UnemittedDependencies) {
    auto &DependencyJD = *KV.first;
    SymbolNameSet *UnemittedDependenciesOnDependencyJD = nullptr;

    for (auto &DependencyName : KV.second) {
      auto &DependencyMI = DependencyJD.MaterializingInfos[DependencyName];

      // Do not add self dependencies.
      if (&DependencyMI == &DependantMI)
        continue;

      // If we haven't looked up the dependencies for DependencyJD yet, do it
      // now and cache the result.
      if (!UnemittedDependenciesOnDependencyJD)
        UnemittedDependenciesOnDependencyJD =
            &DependantMI.UnemittedDependencies[&DependencyJD];

      DependencyMI.Dependants[this].insert(DependantName);
      UnemittedDependenciesOnDependencyJD->insert(DependencyName);
    }
  }
}

ExecutionSession::ExecutionSession(std::shared_ptr<SymbolStringPool> SSP)
    : SSP(SSP ? std::move(SSP) : std::make_shared<SymbolStringPool>()) {
  // Construct the main dylib.
  JDs.push_back(std::unique_ptr<JITDylib>(new JITDylib(*this, "<main>")));
}

JITDylib &ExecutionSession::getMainJITDylib() {
  return runSessionLocked([this]() -> JITDylib & { return *JDs.front(); });
}

JITDylib &ExecutionSession::createJITDylib(std::string Name,
                                           bool AddToMainDylibSearchOrder) {
  return runSessionLocked([&, this]() -> JITDylib & {
    JDs.push_back(
        std::unique_ptr<JITDylib>(new JITDylib(*this, std::move(Name))));
    if (AddToMainDylibSearchOrder)
      JDs.front()->addToSearchOrder(*JDs.back());
    return *JDs.back();
  });
}

void ExecutionSession::legacyFailQuery(AsynchronousSymbolQuery &Q, Error Err) {
  assert(!!Err && "Error should be in failure state");

  bool SendErrorToQuery;
  runSessionLocked([&]() {
    Q.detach();
    SendErrorToQuery = Q.canStillFail();
  });

  if (SendErrorToQuery)
    Q.handleFailed(std::move(Err));
  else
    reportError(std::move(Err));
}

Expected<SymbolMap> ExecutionSession::legacyLookup(
    LegacyAsyncLookupFunction AsyncLookup, SymbolNameSet Names,
    bool WaitUntilReady, RegisterDependenciesFunction RegisterDependencies) {
#if LLVM_ENABLE_THREADS
  // In the threaded case we use promises to return the results.
  std::promise<SymbolMap> PromisedResult;
  std::mutex ErrMutex;
  Error ResolutionError = Error::success();
  std::promise<void> PromisedReady;
  Error ReadyError = Error::success();
  auto OnResolve = [&](Expected<SymbolMap> R) {
    if (R)
      PromisedResult.set_value(std::move(*R));
    else {
      {
        ErrorAsOutParameter _(&ResolutionError);
        std::lock_guard<std::mutex> Lock(ErrMutex);
        ResolutionError = R.takeError();
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
        reportError(std::move(Err));
    };
  }

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
        reportError(std::move(Err));
    };
  }
#endif

  auto Query = std::make_shared<AsynchronousSymbolQuery>(
      Names, std::move(OnResolve), std::move(OnReady));
  // FIXME: This should be run session locked along with the registration code
  // and error reporting below.
  SymbolNameSet UnresolvedSymbols = AsyncLookup(Query, std::move(Names));

  // If the query was lodged successfully then register the dependencies,
  // otherwise fail it with an error.
  if (UnresolvedSymbols.empty())
    RegisterDependencies(Query->QueryRegistrations);
  else {
    bool DeliverError = runSessionLocked([&]() {
      Query->detach();
      return Query->canStillFail();
    });
    auto Err = make_error<SymbolsNotFound>(std::move(UnresolvedSymbols));
    if (DeliverError)
      Query->handleFailed(std::move(Err));
    else
      reportError(std::move(Err));
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

void ExecutionSession::lookup(
    const JITDylibList &JDs, const SymbolNameSet &Symbols,
    SymbolsResolvedCallback OnResolve, SymbolsReadyCallback OnReady,
    RegisterDependenciesFunction RegisterDependencies) {

  // lookup can be re-entered recursively if running on a single thread. Run any
  // outstanding MUs in case this query depends on them, otherwise the main
  // thread will starve waiting for a result from an MU that it failed to run.
  runOutstandingMUs();

  auto Unresolved = std::move(Symbols);
  std::map<JITDylib *, MaterializationUnitList> MUsMap;
  auto Q = std::make_shared<AsynchronousSymbolQuery>(
      Symbols, std::move(OnResolve), std::move(OnReady));
  bool QueryIsFullyResolved = false;
  bool QueryIsFullyReady = false;
  bool QueryFailed = false;

  runSessionLocked([&]() {
    for (auto *JD : JDs) {
      assert(JD && "JITDylibList entries must not be null");
      assert(!MUsMap.count(JD) &&
             "JITDylibList should not contain duplicate entries");
      JD->lodgeQuery(Q, Unresolved, MUsMap[JD]);
    }

    if (Unresolved.empty()) {
      // Query lodged successfully.

      // Record whether this query is fully ready / resolved. We will use
      // this to call handleFullyResolved/handleFullyReady outside the session
      // lock.
      QueryIsFullyResolved = Q->isFullyResolved();
      QueryIsFullyReady = Q->isFullyReady();

      // Call the register dependencies function.
      if (RegisterDependencies && !Q->QueryRegistrations.empty())
        RegisterDependencies(Q->QueryRegistrations);
    } else {
      // Query failed due to unresolved symbols.
      QueryFailed = true;

      // Disconnect the query from its dependencies.
      Q->detach();

      // Replace the MUs.
      for (auto &KV : MUsMap)
        for (auto &MU : KV.second)
          KV.first->replace(std::move(MU));
    }
  });

  if (QueryFailed) {
    Q->handleFailed(make_error<SymbolsNotFound>(std::move(Unresolved)));
    return;
  } else {
    if (QueryIsFullyResolved)
      Q->handleFullyResolved();
    if (QueryIsFullyReady)
      Q->handleFullyReady();
  }

  // Move the MUs to the OutstandingMUs list, then materialize.
  {
    std::lock_guard<std::recursive_mutex> Lock(OutstandingMUsMutex);

    for (auto &KV : MUsMap)
      for (auto &MU : KV.second)
        OutstandingMUs.push_back(std::make_pair(KV.first, std::move(MU)));
  }

  runOutstandingMUs();
}

Expected<SymbolMap>
ExecutionSession::lookup(const JITDylibList &JDs, const SymbolNameSet &Symbols,
                         RegisterDependenciesFunction RegisterDependencies,
                         bool WaitUntilReady) {
#if LLVM_ENABLE_THREADS
  // In the threaded case we use promises to return the results.
  std::promise<SymbolMap> PromisedResult;
  std::mutex ErrMutex;
  Error ResolutionError = Error::success();
  std::promise<void> PromisedReady;
  Error ReadyError = Error::success();
  auto OnResolve = [&](Expected<SymbolMap> R) {
    if (R)
      PromisedResult.set_value(std::move(*R));
    else {
      {
        ErrorAsOutParameter _(&ResolutionError);
        std::lock_guard<std::mutex> Lock(ErrMutex);
        ResolutionError = R.takeError();
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
        reportError(std::move(Err));
    };
  }

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
        reportError(std::move(Err));
    };
  }
#endif

  // Perform the asynchronous lookup.
  lookup(JDs, Symbols, OnResolve, OnReady, RegisterDependencies);

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

void ExecutionSession::runOutstandingMUs() {
  while (1) {
    std::pair<JITDylib *, std::unique_ptr<MaterializationUnit>> JITDylibAndMU;

    {
      std::lock_guard<std::recursive_mutex> Lock(OutstandingMUsMutex);
      if (!OutstandingMUs.empty()) {
        JITDylibAndMU = std::move(OutstandingMUs.back());
        OutstandingMUs.pop_back();
      }
    }

    if (JITDylibAndMU.first) {
      assert(JITDylibAndMU.second && "JITDylib, but no MU?");
      dispatchMaterialization(*JITDylibAndMU.first,
                              std::move(JITDylibAndMU.second));
    } else
      break;
  }
}

Expected<SymbolMap> lookup(const JITDylibList &JDs, SymbolNameSet Names) {

  if (JDs.empty())
    return SymbolMap();

  auto &ES = (*JDs.begin())->getExecutionSession();

  return ES.lookup(JDs, Names, NoDependenciesToRegister, true);
}

/// Look up a symbol by searching a list of JDs.
Expected<JITEvaluatedSymbol> lookup(const JITDylibList &JDs,
                                    SymbolStringPtr Name) {
  SymbolNameSet Names({Name});
  if (auto ResultMap = lookup(JDs, std::move(Names))) {
    assert(ResultMap->size() == 1 && "Unexpected number of results");
    assert(ResultMap->count(Name) && "Missing result for symbol");
    return std::move(ResultMap->begin()->second);
  } else
    return ResultMap.takeError();
}

MangleAndInterner::MangleAndInterner(ExecutionSession &ES, const DataLayout &DL)
    : ES(ES), DL(DL) {}

SymbolStringPtr MangleAndInterner::operator()(StringRef Name) {
  std::string MangledName;
  {
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
  }
  return ES.getSymbolStringPool().intern(MangledName);
}

} // End namespace orc.
} // End namespace llvm.
