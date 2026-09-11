//===------------------ COFF.cpp - COFF format utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/COFF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LoadLinkableFile.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFFImportFile.h"

#define DEBUG_TYPE "orc"

namespace llvm::orc {

Expected<bool> COFFImportFileScanner::operator()(object::Archive &A,
                                                 MemoryBufferRef MemberBuf,
                                                 size_t Index) const {
  // Try to build a binary for the member.
  auto Bin = object::createBinary(MemberBuf);
  if (!Bin) {
    // If we can't then consume the error and return false (i.e. not loadable).
    consumeError(Bin.takeError());
    return false;
  }

  // If this is a COFF import file then handle it and return false (not
  // loadable).
  if ((*Bin)->isCOFFImportFile()) {
    ImportedDynamicLibraries.insert((*Bin)->getFileName().str());
    return false;
  }

  // Otherwise the member is loadable (at least as far as COFFImportFileScanner
  // is concerned), so return true;
  return true;
}

struct COFFStaticLibraryDefinitionGenerator::Impl {
  struct ImportRecord {
    std::string SymbolName;
    std::string ExportName;
    COFF::ImportType Type = COFF::IMPORT_DATA;
    SmallVector<SymbolStringPtr, 2> ProvidedSymbols;
  };

  Impl(ObjectLinkingLayer &L, jitlink::AnonymousPointerCreator CreatePointer,
       jitlink::PointerJumpStubCreator CreateStub)
      : L(L), ES(L.getExecutionSession()),
        CreatePointer(std::move(CreatePointer)),
        CreateStub(std::move(CreateStub)) {}

  Expected<bool> visitMember(std::set<std::string> &ImportedDynamicLibraries,
                             MemoryBufferRef MemberBuf) {
    auto Bin = object::createBinary(MemberBuf);
    if (!Bin) {
      consumeError(Bin.takeError());
      return false;
    }

    auto *ImportFile = dyn_cast<object::COFFImportFile>(Bin->get());
    if (!ImportFile)
      return true;

    if (ImportFile->getMachine() != COFF::IMAGE_FILE_MACHINE_AMD64)
      return make_error<StringError>(
          "COFFStaticLibraryDefinitionGenerator only supports x86-64 "
          "short-import members",
          inconvertibleErrorCode());

    if (ImportFile->getSymbolName().empty())
      return make_error<StringError>(
          "COFF import file has an empty symbol name",
          inconvertibleErrorCode());

    ImportedDynamicLibraries.insert(ImportFile->getFileName().str());

    size_t RecordIndex = ImportRecords.size();
    ImportRecord R;
    R.SymbolName = ImportFile->getSymbolName().str();
    R.ExportName = ImportFile->getExportName().str();
    R.Type = static_cast<COFF::ImportType>(
        ImportFile->getCOFFImportHeader()->getType());

    for (auto Sym : ImportFile->symbols()) {
      std::string Name;
      raw_string_ostream NameOS(Name);
      if (auto Err = Sym.printName(NameOS))
        return std::move(Err);
      auto InternedName = ES.intern(Name);
      R.ProvidedSymbols.push_back(InternedName);
      LazyImportSymbols.try_emplace(InternedName, RecordIndex);
    }

    ImportRecords.push_back(std::move(R));
    return false;
  }

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Symbols) {
    if (K != LookupKind::Static)
      return Error::success();

    DenseSet<size_t> ImportsToEmit;
    for (const auto &[Name, _] : Symbols)
      if (auto I = LazyImportSymbols.find(Name); I != LazyImportSymbols.end())
        ImportsToEmit.insert(I->second);

    if (auto Err =
            ObjectGenerator->tryToGenerate(LS, K, JD, JDLookupFlags, Symbols))
      return Err;

    if (ImportsToEmit.empty())
      return Error::success();

    JITDylibSearchOrder LinkOrder;
    JD.withLinkOrderDo([&](const JITDylibSearchOrder &LO) {
      LinkOrder.reserve(LO.size());
      for (auto &KV : LO)
        if (KV.first != &JD)
          LinkOrder.push_back(KV);
    });

    SymbolLookupSet LookupSet;
    DenseMap<size_t, SymbolStringPtr> ImportTargets;
    DenseSet<SymbolStringPtr> SeenExports;
    for (size_t RecordIndex : ImportsToEmit) {
      auto &R = ImportRecords[RecordIndex];
      if (R.ExportName.empty())
        return make_error<StringError>(
            "ordinal COFF imports are not supported for " + R.SymbolName,
            inconvertibleErrorCode());
      auto ExportName = ES.intern(R.ExportName);
      ImportTargets[RecordIndex] = ExportName;
      if (SeenExports.insert(ExportName).second)
        LookupSet.add(ExportName, SymbolLookupFlags::RequiredSymbol);
    }

    ES.lookup(
        LookupKind::Static, LinkOrder, std::move(LookupSet),
        SymbolState::Resolved,
        [this, JD = JITDylibSP(&JD), LS = std::move(LS),
         ImportsToEmit = std::move(ImportsToEmit),
         ImportTargets =
             std::move(ImportTargets)](Expected<SymbolMap> Resolved) mutable {
          if (!Resolved)
            return LS.continueLookup(Resolved.takeError());

          auto G = createImportGraph(ImportsToEmit, ImportTargets, *Resolved);
          if (!G)
            return LS.continueLookup(G.takeError());
          if (auto Err = L.add(*JD, std::move(*G)))
            return LS.continueLookup(std::move(Err));

          for (size_t RecordIndex : ImportsToEmit) {
            for (auto &Name : ImportRecords[RecordIndex].ProvidedSymbols)
              if (auto I = LazyImportSymbols.find(Name);
                  I != LazyImportSymbols.end() && I->second == RecordIndex)
                LazyImportSymbols.erase(I);
          }

          LS.continueLookup(Error::success());
        },
        NoDependenciesToRegister);

    return Error::success();
  }

  Expected<std::unique_ptr<jitlink::LinkGraph>>
  createImportGraph(const DenseSet<size_t> &ImportsToEmit,
                    const DenseMap<size_t, SymbolStringPtr> &ImportTargets,
                    const SymbolMap &Resolved) {
    Triple TT = ES.getTargetTriple();

    auto G = std::make_unique<jitlink::LinkGraph>(
        "<COFF_IMPORTS>", ES.getSymbolStringPool(), TT, SubtargetFeatures(),
        jitlink::getGenericEdgeKindName);
    auto &Sec =
        G->createSection("$__COFF_IMPORTS", MemProt::Read | MemProt::Exec);

    for (size_t RecordIndex : ImportsToEmit) {
      auto &R = ImportRecords[RecordIndex];
      auto TargetName = ImportTargets.lookup(RecordIndex);
      auto TargetDef = Resolved.find(TargetName);
      if (TargetDef == Resolved.end())
        return make_error<StringError>("resolved COFF import target " +
                                           *TargetName + " is missing for " +
                                           R.SymbolName,
                                       inconvertibleErrorCode());

      auto &Target = G->addAbsoluteSymbol(
          TargetName, TargetDef->second.getAddress(), G->getPointerSize(),
          jitlink::Linkage::Strong, jitlink::Scope::Local, false);
      auto &Ptr = CreatePointer(*G, Sec, &Target, 0);

      SymbolStringPtr ImpName;
      SymbolStringPtr PlainName;
      for (auto &Name : R.ProvidedSymbols) {
        if (StringRef(*Name).starts_with(getImpPrefix()))
          ImpName = Name;
        else
          PlainName = Name;
      }

      if (!ImpName)
        return make_error<StringError>(
            "COFF import does not provide an __imp_ symbol for " + R.SymbolName,
            inconvertibleErrorCode());

      Ptr.setName(ImpName);
      Ptr.setLinkage(jitlink::Linkage::Strong);
      Ptr.setScope(jitlink::Scope::Default);

      if (!PlainName || R.Type == COFF::IMPORT_DATA)
        continue;

      if (R.Type == COFF::IMPORT_CODE) {
        auto &Stub = CreateStub(*G, Sec, Ptr);
        Stub.setName(PlainName);
        Stub.setLinkage(jitlink::Linkage::Strong);
        Stub.setScope(jitlink::Scope::Default);
      } else if (R.Type == COFF::IMPORT_CONST)
        G->addDefinedSymbol(Ptr.getBlock(), Ptr.getOffset(), PlainName,
                            Ptr.getSize(), jitlink::Linkage::Strong,
                            jitlink::Scope::Default, false, false);
    }

    return std::move(G);
  }

  static constexpr StringLiteral getImpPrefix() { return "__imp_"; }

  ObjectLinkingLayer &L;
  ExecutionSession &ES;
  jitlink::AnonymousPointerCreator CreatePointer;
  jitlink::PointerJumpStubCreator CreateStub;
  std::unique_ptr<StaticLibraryDefinitionGenerator> ObjectGenerator;
  DenseMap<SymbolStringPtr, size_t> LazyImportSymbols;
  std::vector<ImportRecord> ImportRecords;
};

Expected<std::unique_ptr<COFFStaticLibraryDefinitionGenerator>>
COFFStaticLibraryDefinitionGenerator::Load(
    ObjectLinkingLayer &L, const char *FileName,
    std::set<std::string> &ImportedDynamicLibraries) {
  auto Linkable =
      loadLinkableFile(FileName, L.getExecutionSession().getTargetTriple(),
                       LoadArchives::Required);
  if (!Linkable)
    return Linkable.takeError();
  auto Archive = object::Archive::create(Linkable->first->getMemBufferRef());
  if (!Archive)
    return Archive.takeError();
  return Create(L, std::move(Linkable->first), std::move(*Archive),
                ImportedDynamicLibraries);
}

Expected<std::unique_ptr<COFFStaticLibraryDefinitionGenerator>>
COFFStaticLibraryDefinitionGenerator::Create(
    ObjectLinkingLayer &L, std::unique_ptr<MemoryBuffer> ArchiveBuffer,
    std::unique_ptr<object::Archive> Archive,
    std::set<std::string> &ImportedDynamicLibraries) {
  Triple TT = L.getExecutionSession().getTargetTriple();
  if (!TT.isOSBinFormatCOFF() || TT.getArch() != Triple::x86_64)
    return make_error<StringError>(
        "COFFStaticLibraryDefinitionGenerator only supports x86-64 COFF "
        "targets: " +
            TT.str(),
        inconvertibleErrorCode());

  auto CreatePointer = jitlink::getAnonymousPointerCreator(TT);
  if (!CreatePointer)
    return make_error<StringError>(
        "COFFStaticLibraryDefinitionGenerator: no pointer creator for " +
            TT.str(),
        inconvertibleErrorCode());
  auto CreateStub = jitlink::getPointerJumpStubCreator(TT);
  if (!CreateStub)
    return make_error<StringError>(
        "COFFStaticLibraryDefinitionGenerator: no stub creator for " + TT.str(),
        inconvertibleErrorCode());

  auto P = std::make_unique<Impl>(L, std::move(CreatePointer),
                                  std::move(CreateStub));
  auto VisitMembers = [P = P.get(), &ImportedDynamicLibraries](
                          object::Archive &, MemoryBufferRef MemberBuf,
                          size_t) -> Expected<bool> {
    return P->visitMember(ImportedDynamicLibraries, MemberBuf);
  };
  auto ObjectGenerator = StaticLibraryDefinitionGenerator::Create(
      L, std::move(ArchiveBuffer), std::move(Archive), std::move(VisitMembers));
  if (!ObjectGenerator)
    return ObjectGenerator.takeError();
  P->ObjectGenerator = std::move(*ObjectGenerator);
  return std::unique_ptr<COFFStaticLibraryDefinitionGenerator>(
      new COFFStaticLibraryDefinitionGenerator(std::move(P)));
}

COFFStaticLibraryDefinitionGenerator::~COFFStaticLibraryDefinitionGenerator() =
    default;

COFFStaticLibraryDefinitionGenerator::COFFStaticLibraryDefinitionGenerator(
    std::unique_ptr<Impl> P)
    : P(std::move(P)) {}

Error COFFStaticLibraryDefinitionGenerator::tryToGenerate(
    LookupState &LS, LookupKind K, JITDylib &JD,
    JITDylibLookupFlags JDLookupFlags, const SymbolLookupSet &Symbols) {
  return P->tryToGenerate(LS, K, JD, JDLookupFlags, Symbols);
}

} // namespace llvm::orc
