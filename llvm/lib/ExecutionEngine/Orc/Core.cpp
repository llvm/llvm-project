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

namespace llvm {
namespace orc {

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

VSO::MaterializationInfo::MaterializationInfo(JITSymbolFlags Flags,
                                              AsynchronousSymbolQuery &Query)
    : Flags(std::move(Flags)), PendingResolution({&Query}) {}

JITSymbolFlags VSO::MaterializationInfo::getFlags() const { return Flags; }

JITTargetAddress VSO::MaterializationInfo::getAddress() const {
  return Address;
}

void VSO::MaterializationInfo::query(SymbolStringPtr Name,
                                     AsynchronousSymbolQuery &Query) {
  if (Address != 0) {
    Query.setDefinition(Name, JITEvaluatedSymbol(Address, Flags));
    PendingFinalization.push_back(&Query);
  } else
    PendingResolution.push_back(&Query);
}

void VSO::MaterializationInfo::resolve(SymbolStringPtr Name,
                                       JITEvaluatedSymbol Sym) {
  // FIXME: Sanity check flags?
  Flags = Sym.getFlags();
  Address = Sym.getAddress();
  for (auto *Query : PendingResolution) {
    Query->setDefinition(Name, std::move(Sym));
    PendingFinalization.push_back(Query);
  }
  PendingResolution = {};
}

void VSO::MaterializationInfo::finalize() {
  for (auto *Query : PendingFinalization)
    Query->notifySymbolFinalized();
  PendingFinalization = {};
}

VSO::SymbolTableEntry::SymbolTableEntry(JITSymbolFlags Flags,
                                        SymbolSource &Source)
    : Flags(JITSymbolFlags::FlagNames(Flags | JITSymbolFlags::NotMaterialized)),
      Source(&Source) {
  // FIXME: Assert flag sanity.
}

VSO::SymbolTableEntry::SymbolTableEntry(JITEvaluatedSymbol Sym)
    : Flags(Sym.getFlags()), Address(Sym.getAddress()) {
  // FIXME: Assert flag sanity.
}

VSO::SymbolTableEntry::SymbolTableEntry(SymbolTableEntry &&Other)
    : Flags(Other.Flags), Address(0) {
  if (Flags.isMaterializing())
    MatInfo = std::move(Other.MatInfo);
  else
    Source = Other.Source;
}

VSO::SymbolTableEntry::~SymbolTableEntry() {
  assert(!Flags.isMaterializing() &&
         "Symbol table entry destroyed while symbol was being materialized");
}

JITSymbolFlags VSO::SymbolTableEntry::getFlags() const { return Flags; }

void VSO::SymbolTableEntry::replaceWithSource(VSO &V, SymbolStringPtr Name,
                                              JITSymbolFlags Flags,
                                              SymbolSource &NewSource) {
  assert(!this->Flags.isMaterializing() &&
         "Attempted to replace symbol with lazy definition during "
         "materialization");
  if (!this->Flags.isMaterialized())
    Source->discard(V, Name);
  this->Flags = Flags;
  this->Source = &NewSource;
}

SymbolSource *VSO::SymbolTableEntry::query(SymbolStringPtr Name,
                                           AsynchronousSymbolQuery &Query) {
  if (Flags.isMaterializing()) {
    MatInfo->query(std::move(Name), Query);
    return nullptr;
  } else if (Flags.isMaterialized()) {
    Query.setDefinition(std::move(Name), JITEvaluatedSymbol(Address, Flags));
    Query.notifySymbolFinalized();
    return nullptr;
  }
  SymbolSource *S = Source;
  new (&MatInfo) std::unique_ptr<MaterializationInfo>(
      llvm::make_unique<MaterializationInfo>(Flags, Query));
  Flags |= JITSymbolFlags::Materializing;
  return S;
}

void VSO::SymbolTableEntry::resolve(VSO &V, SymbolStringPtr Name,
                                    JITEvaluatedSymbol Sym) {
  if (Flags.isMaterializing())
    MatInfo->resolve(std::move(Name), std::move(Sym));
  else {
    // If there's a layer for this symbol.
    if (!Flags.isMaterialized())
      Source->discard(V, Name);

    // FIXME: Should we assert flag state here (flags must match except for
    //        materialization state, overrides must be legal) or in the caller
    //        in VSO?
    Flags = Sym.getFlags();
    Address = Sym.getAddress();
  }
}

void VSO::SymbolTableEntry::finalize() {
  if (Flags.isMaterializing()) {
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

Error VSO::defineLazy(const SymbolFlagsMap &NewSymbols, SymbolSource &Source) {
  Error Err = Error::success();
  for (auto &KV : NewSymbols) {
    auto I = Symbols.find(KV.first);

    auto LinkageResult = compareLinkage(
        I == Symbols.end() ? None
                           : Optional<JITSymbolFlags>(I->second.getFlags()),
        KV.second);

    // Discard weaker definitions.
    if (LinkageResult == ExistingDefinitionIsStronger)
      Source.discard(*this, KV.first);

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

VSO::LookupFlagsResult VSO::lookupFlags(SymbolNameSet Names) {
  SymbolFlagsMap FlagsFound;

  for (SymbolNameSet::iterator I = Names.begin(), E = Names.end(); I != E;) {
    auto Tmp = I++;
    auto SymI = Symbols.find(*Tmp);

    // If the symbol isn't in this dylib then just continue.
    if (SymI == Symbols.end())
      continue;

    Names.erase(Tmp);

    FlagsFound[SymI->first] =
        JITSymbolFlags::stripTransientFlags(SymI->second.getFlags());
  }

  return {std::move(FlagsFound), std::move(Names)};
}

VSO::LookupResult VSO::lookup(AsynchronousSymbolQuery &Query,
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
    if (auto *Source = SymI->second.query(SymI->first, Query))
      MaterializationWork[Source].insert(SymI->first);
  }

  return {std::move(MaterializationWork), std::move(Names)};
}

VModuleKey ExecutionSession::allocateVModule() { return ++LastKey; }

void ExecutionSession::releaseVModule(VModuleKey VMod) {
  // FIXME: Recycle keys.
}

} // End namespace orc.
} // End namespace llvm.
