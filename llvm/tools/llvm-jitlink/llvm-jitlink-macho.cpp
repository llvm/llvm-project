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

#define DEBUG_TYPE "llvm-jitlink"

using namespace llvm;
using namespace llvm::jitlink;

static bool isMachOGOTSection(Section &S) { return S.getName() == "$__GOT"; }

static bool isMachOStubsSection(Section &S) {
  return S.getName() == "$__STUBS";
}

static Expected<Edge &> getFirstRelocationEdge(AtomGraph &G, DefinedAtom &DA) {
  auto EItr = std::find_if(DA.edges().begin(), DA.edges().end(),
                           [](Edge &E) { return E.isRelocation(); });
  if (EItr == DA.edges().end())
    return make_error<StringError>("GOT entry in " + G.getName() + ", \"" +
                                       DA.getSection().getName() +
                                       "\" has no relocations",
                                   inconvertibleErrorCode());
  return *EItr;
}

static Expected<Atom &> getMachOGOTTarget(AtomGraph &G, DefinedAtom &DA) {
  auto E = getFirstRelocationEdge(G, DA);
  if (!E)
    return E.takeError();
  auto &TA = E->getTarget();
  if (!TA.hasName())
    return make_error<StringError>("GOT entry in " + G.getName() + ", \"" +
                                       DA.getSection().getName() +
                                       "\" points to anonymous "
                                       "atom",
                                   inconvertibleErrorCode());
  if (TA.isDefined() || TA.isAbsolute())
    return make_error<StringError>(
        "GOT entry \"" + TA.getName() + "\" in " + G.getName() + ", \"" +
            DA.getSection().getName() + "\" does not point to an external atom",
        inconvertibleErrorCode());
  return TA;
}

static Expected<Atom &> getMachOStubTarget(AtomGraph &G, DefinedAtom &DA) {
  auto E = getFirstRelocationEdge(G, DA);
  if (!E)
    return E.takeError();
  auto &GOTA = E->getTarget();
  if (!GOTA.isDefined() ||
      !isMachOGOTSection(static_cast<DefinedAtom &>(GOTA).getSection()))
    return make_error<StringError>("Stubs entry in " + G.getName() + ", \"" +
                                       DA.getSection().getName() +
                                       "\" does not point to GOT entry",
                                   inconvertibleErrorCode());
  return getMachOGOTTarget(G, static_cast<DefinedAtom &>(GOTA));
}

namespace llvm {

Error registerMachOStubsAndGOT(Session &S, AtomGraph &G) {
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
             << (Sec.atoms_empty() ? "empty. skipping." : "processing...")
             << "\n";
    });

    // Skip empty sections.
    if (Sec.atoms_empty())
      continue;

    if (FileInfo.SectionInfos.count(Sec.getName()))
      return make_error<StringError>("Encountered duplicate section name \"" +
                                         Sec.getName() + "\" in \"" + FileName +
                                         "\"",
                                     inconvertibleErrorCode());

    bool isGOTSection = isMachOGOTSection(Sec);
    bool isStubsSection = isMachOStubsSection(Sec);

    auto *FirstAtom = *Sec.atoms().begin();
    auto *LastAtom = FirstAtom;
    for (auto *DA : Sec.atoms()) {
      if (DA->getAddress() < FirstAtom->getAddress())
        FirstAtom = DA;
      if (DA->getAddress() > LastAtom->getAddress())
        LastAtom = DA;
      if (isGOTSection) {
        if (Sec.isZeroFill())
          return make_error<StringError>("Content atom in zero-fill section",
                                         inconvertibleErrorCode());

        if (auto TA = getMachOGOTTarget(G, *DA)) {
          FileInfo.GOTEntryInfos[TA->getName()] = {DA->getContent(),
                                                   DA->getAddress()};
        } else
          return TA.takeError();
      } else if (isStubsSection) {
        if (Sec.isZeroFill())
          return make_error<StringError>("Content atom in zero-fill section",
                                         inconvertibleErrorCode());

        if (auto TA = getMachOStubTarget(G, *DA))
          FileInfo.StubInfos[TA->getName()] = {DA->getContent(),
                                               DA->getAddress()};
        else
          return TA.takeError();
      } else if (DA->hasName() && DA->isGlobal()) {
        if (DA->isZeroFill())
          S.SymbolInfos[DA->getName()] = {DA->getSize(), DA->getAddress()};
        else {
          if (Sec.isZeroFill())
            return make_error<StringError>("Content atom in zero-fill section",
                                           inconvertibleErrorCode());
          S.SymbolInfos[DA->getName()] = {DA->getContent(), DA->getAddress()};
        }
      }
    }

    JITTargetAddress SecAddr = FirstAtom->getAddress();
    uint64_t SecSize = (LastAtom->getAddress() + LastAtom->getSize()) -
                       FirstAtom->getAddress();

    if (Sec.isZeroFill())
      FileInfo.SectionInfos[Sec.getName()] = {SecSize, SecAddr};
    else
      FileInfo.SectionInfos[Sec.getName()] = {
          StringRef(FirstAtom->getContent().data(), SecSize), SecAddr};
  }

  return Error::success();
}

} // end namespace llvm
