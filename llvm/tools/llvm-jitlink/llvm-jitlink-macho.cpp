//===-- llvm-jitlink-macho.cpp -- MachO parsing support for llvm-jitlink --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MachO parsing support for llvm-jitlink.
//
//===----------------------------------------------------------------------===//

#include "llvm-jitlink.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "llvm_jitlink"

using namespace llvm;
using namespace llvm::jitlink;

static bool isMachOGOTSection(Section &S) { return S.getName() == "$__GOT"; }

static bool isMachOStubsSection(Section &S) {
  return S.getName() == "$__STUBS";
}

static Expected<Edge &> getFirstRelocationEdge(LinkGraph &G, Block &B) {
  auto EItr = std::find_if(B.edges().begin(), B.edges().end(),
                           [](Edge &E) { return E.isRelocation(); });
  if (EItr == B.edges().end())
    return make_error<StringError>("GOT entry in " + G.getName() + ", \"" +
                                       B.getSection().getName() +
                                       "\" has no relocations",
                                   inconvertibleErrorCode());
  return *EItr;
}

static Expected<Symbol &> getMachOGOTTarget(LinkGraph &G, Block &B) {
  auto E = getFirstRelocationEdge(G, B);
  if (!E)
    return E.takeError();
  auto &TargetSym = E->getTarget();
  if (!TargetSym.hasName())
    return make_error<StringError>(
        "GOT entry in " + G.getName() + ", \"" +
            TargetSym.getBlock().getSection().getName() +
            "\" points to anonymous "
            "symbol",
        inconvertibleErrorCode());
  if (TargetSym.isDefined() || TargetSym.isAbsolute())
    return make_error<StringError>(
        "GOT entry \"" + TargetSym.getName() + "\" in " + G.getName() + ", \"" +
            TargetSym.getBlock().getSection().getName() +
            "\" does not point to an external symbol",
        inconvertibleErrorCode());
  return TargetSym;
}

static Expected<Symbol &> getMachOStubTarget(LinkGraph &G, Block &B) {
  auto E = getFirstRelocationEdge(G, B);
  if (!E)
    return E.takeError();
  auto &GOTSym = E->getTarget();
  if (!GOTSym.isDefined() || !isMachOGOTSection(GOTSym.getBlock().getSection()))
    return make_error<StringError>(
        "Stubs entry in " + G.getName() + ", \"" +
            GOTSym.getBlock().getSection().getName() +
            "\" does not point to GOT entry",
        inconvertibleErrorCode());
  return getMachOGOTTarget(G, GOTSym.getBlock());
}

namespace llvm {

Error registerMachOStubsAndGOT(Session &S, LinkGraph &G) {
  auto FileName = sys::path::filename(G.getName());
  if (S.FileInfos.count(FileName)) {
    return make_error<StringError>("When -check is passed, file names must be "
                                   "distinct (duplicate: \"" +
                                       FileName + "\")",
                                   inconvertibleErrorCode());
  }

  auto &FileInfo = S.FileInfos[FileName];
  LLVM_DEBUG({
    dbgs() << "Registering MachO file info for \"" << FileName << "\"\n";
  });
  for (auto &Sec : G.sections()) {
    LLVM_DEBUG({
      dbgs() << "  Section \"" << Sec.getName() << "\": "
             << (llvm::empty(Sec.symbols()) ? "empty. skipping."
                                            : "processing...")
             << "\n";
    });

    // Skip empty sections.
    if (llvm::empty(Sec.symbols()))
      continue;

    if (FileInfo.SectionInfos.count(Sec.getName()))
      return make_error<StringError>("Encountered duplicate section name \"" +
                                         Sec.getName() + "\" in \"" + FileName +
                                         "\"",
                                     inconvertibleErrorCode());

    bool isGOTSection = isMachOGOTSection(Sec);
    bool isStubsSection = isMachOStubsSection(Sec);

    bool SectionContainsContent = false;
    bool SectionContainsZeroFill = false;

    auto *FirstSym = *Sec.symbols().begin();
    auto *LastSym = FirstSym;
    for (auto *Sym : Sec.symbols()) {
      if (Sym->getAddress() < FirstSym->getAddress())
        FirstSym = Sym;
      if (Sym->getAddress() > LastSym->getAddress())
        LastSym = Sym;
      if (isGOTSection) {
        if (Sym->isSymbolZeroFill())
          return make_error<StringError>("zero-fill atom in GOT section",
                                         inconvertibleErrorCode());

        if (auto TS = getMachOGOTTarget(G, Sym->getBlock()))
          FileInfo.GOTEntryInfos[TS->getName()] = {Sym->getSymbolContent(),
                                                   Sym->getAddress()};
        else
          return TS.takeError();
        SectionContainsContent = true;
      } else if (isStubsSection) {
        if (Sym->isSymbolZeroFill())
          return make_error<StringError>("zero-fill atom in Stub section",
                                         inconvertibleErrorCode());

        if (auto TS = getMachOStubTarget(G, Sym->getBlock()))
          FileInfo.StubInfos[TS->getName()] = {Sym->getSymbolContent(),
                                               Sym->getAddress()};
        else
          return TS.takeError();
        SectionContainsContent = true;
      } else if (Sym->hasName()) {
        if (Sym->isSymbolZeroFill()) {
          S.SymbolInfos[Sym->getName()] = {Sym->getSize(), Sym->getAddress()};
          SectionContainsZeroFill = true;
        } else {
          S.SymbolInfos[Sym->getName()] = {Sym->getSymbolContent(),
                                           Sym->getAddress()};
          SectionContainsContent = true;
        }
      }
    }

    JITTargetAddress SecAddr = FirstSym->getAddress();
    uint64_t SecSize =
        (LastSym->getBlock().getAddress() + LastSym->getBlock().getSize()) -
        SecAddr;

    if (SectionContainsZeroFill && SectionContainsContent)
      return make_error<StringError>("Mixed zero-fill and content sections not "
                                     "supported yet",
                                     inconvertibleErrorCode());
    if (SectionContainsZeroFill)
      FileInfo.SectionInfos[Sec.getName()] = {SecSize, SecAddr};
    else
      FileInfo.SectionInfos[Sec.getName()] = {
          StringRef(FirstSym->getBlock().getContent().data(), SecSize),
          SecAddr};
  }

  return Error::success();
}

} // end namespace llvm
