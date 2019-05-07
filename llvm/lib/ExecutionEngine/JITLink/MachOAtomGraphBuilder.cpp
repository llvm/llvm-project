//=--------- MachOAtomGraphBuilder.cpp - MachO AtomGraph builder ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic MachO AtomGraph buliding code.
//
//===----------------------------------------------------------------------===//

#include "MachOAtomGraphBuilder.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

MachOAtomGraphBuilder::~MachOAtomGraphBuilder() {}

Expected<std::unique_ptr<AtomGraph>> MachOAtomGraphBuilder::buildGraph() {
  if (auto Err = parseSections())
    return std::move(Err);

  if (auto Err = addAtoms())
    return std::move(Err);

  if (auto Err = addRelocations())
    return std::move(Err);

  return std::move(G);
}

MachOAtomGraphBuilder::MachOAtomGraphBuilder(const object::MachOObjectFile &Obj)
    : Obj(Obj),
      G(llvm::make_unique<AtomGraph>(Obj.getFileName(), getPointerSize(Obj),
                                     getEndianness(Obj))) {}

void MachOAtomGraphBuilder::addCustomAtomizer(StringRef SectionName,
                                              CustomAtomizeFunction Atomizer) {
  assert(!CustomAtomizeFunctions.count(SectionName) &&
         "Custom atomizer for this section already exists");
  CustomAtomizeFunctions[SectionName] = std::move(Atomizer);
}

unsigned
MachOAtomGraphBuilder::getPointerSize(const object::MachOObjectFile &Obj) {
  return Obj.is64Bit() ? 8 : 4;
}

support::endianness
MachOAtomGraphBuilder::getEndianness(const object::MachOObjectFile &Obj) {
  return Obj.isLittleEndian() ? support::little : support::big;
}

MachOAtomGraphBuilder::MachOSection &MachOAtomGraphBuilder::getCommonSection() {
  if (!CommonSymbolsSection) {
    auto Prot = static_cast<sys::Memory::ProtectionFlags>(
        sys::Memory::MF_READ | sys::Memory::MF_WRITE);
    auto &GenericSection = G->createSection("<common>", Prot, true);
    CommonSymbolsSection = MachOSection(GenericSection);
  }
  return *CommonSymbolsSection;
}

Error MachOAtomGraphBuilder::parseSections() {
  for (auto &SecRef : Obj.sections()) {
    assert((SecRef.getAlignment() <= std::numeric_limits<uint32_t>::max()) &&
           "Section alignment does not fit in 32 bits");

    StringRef Name;
    if (auto EC = SecRef.getName(Name))
      return errorCodeToError(EC);

    StringRef Content;

    // If this is a virtual section, leave its content empty.
    if (!SecRef.isVirtual()) {
      if (auto EC = SecRef.getContents(Content))
        return errorCodeToError(EC);
      if (Content.size() != SecRef.getSize())
        return make_error<JITLinkError>("Section content size does not match "
                                        "declared size for " +
                                        Name);
    }

    unsigned SectionIndex = SecRef.getIndex() + 1;

    LLVM_DEBUG({
      dbgs() << "Adding section " << Name << ": "
             << format("0x%016" PRIx64, SecRef.getAddress())
             << ", size: " << Content.size()
             << ", align: " << SecRef.getAlignment() << "\n";
    });

    // FIXME: Get real section permissions
    // How, exactly, on MachO?
    sys::Memory::ProtectionFlags Prot;
    if (SecRef.isText())
      Prot = static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                       sys::Memory::MF_EXEC);
    else
      Prot = static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                       sys::Memory::MF_WRITE);

    auto &GenericSection = G->createSection(Name, Prot, SecRef.isBSS());
    if (SecRef.isVirtual())
      Sections[SectionIndex] =
          MachOSection(GenericSection, SecRef.getAddress(),
                       SecRef.getAlignment(), SecRef.getSize());
    Sections[SectionIndex] = MachOSection(GenericSection, SecRef.getAddress(),
                                          SecRef.getAlignment(), Content);
  }

  return Error::success();
}

// Adds atoms with identified start addresses (but not lengths) for all named
// atoms.
// Also, for every section that contains named atoms, but does not have an
// atom at offset zero of that section, constructs an anonymous atom covering
// that range.
Error MachOAtomGraphBuilder::addNonCustomAtoms() {
  using AddrToAtomMap = std::map<JITTargetAddress, DefinedAtom *>;
  DenseMap<MachOSection *, AddrToAtomMap> SecToAtoms;

  DenseMap<MachOSection *, unsigned> FirstOrdinal;

  for (auto SymI = Obj.symbol_begin(), SymE = Obj.symbol_end(); SymI != SymE;
       ++SymI) {
    object::SymbolRef Sym(SymI->getRawDataRefImpl(), &Obj);

    auto Name = Sym.getName();
    if (!Name)
      return Name.takeError();

    auto Addr = Sym.getAddress();
    if (!Addr)
      return Addr.takeError();

    auto SymType = Sym.getType();
    if (!SymType)
      return SymType.takeError();

    auto Flags = Sym.getFlags();

    if (Flags & object::SymbolRef::SF_Undefined) {
      LLVM_DEBUG(dbgs() << "Adding undef atom \"" << *Name << "\"\n");
      G->addExternalAtom(*Name);
      continue;
    } else if (Flags & object::SymbolRef::SF_Absolute) {
      LLVM_DEBUG(dbgs() << "Adding absolute \"" << *Name << "\" addr: "
                        << format("0x%016" PRIx64, *Addr) << "\n");
      auto &A = G->addAbsoluteAtom(*Name, *Addr);
      A.setGlobal(Flags & object::SymbolRef::SF_Global);
      A.setExported(Flags & object::SymbolRef::SF_Exported);
      A.setWeak(Flags & object::SymbolRef::SF_Weak);
      continue;
    } else if (Flags & object::SymbolRef::SF_Common) {
      LLVM_DEBUG({
        dbgs() << "Adding common \"" << *Name
               << "\" addr: " << format("0x%016" PRIx64, *Addr) << "\n";
      });
      auto &A =
          G->addCommonAtom(getCommonSection().getGenericSection(), *Name, *Addr,
                           std::max(Sym.getAlignment(), 1U),
                           Obj.getCommonSymbolSize(Sym.getRawDataRefImpl()));
      A.setGlobal(Flags & object::SymbolRef::SF_Global);
      A.setExported(Flags & object::SymbolRef::SF_Exported);
      continue;
    }

    LLVM_DEBUG(dbgs() << "Adding defined atom \"" << *Name << "\"\n");

    // This atom is neither undefined nor absolute, so it must be defined in
    // this object. Get its section index.
    auto SecItr = Sym.getSection();
    if (!SecItr)
      return SecItr.takeError();

    uint64_t SectionIndex = (*SecItr)->getIndex() + 1;

    LLVM_DEBUG(dbgs() << "  to section index " << SectionIndex << "\n");

    auto SecByIndexItr = Sections.find(SectionIndex);
    if (SecByIndexItr == Sections.end())
      return make_error<JITLinkError>("Unrecognized section index in macho");

    auto &Sec = SecByIndexItr->second;

    auto &A = G->addDefinedAtom(Sec.getGenericSection(), *Name, *Addr,
                                std::max(Sym.getAlignment(), 1U));

    A.setGlobal(Flags & object::SymbolRef::SF_Global);
    A.setExported(Flags & object::SymbolRef::SF_Exported);
    A.setWeak(Flags & object::SymbolRef::SF_Weak);

    A.setCallable(*SymType & object::SymbolRef::ST_Function);

    LLVM_DEBUG({
      dbgs() << "  Added " << *Name
             << " addr: " << format("0x%016" PRIx64, *Addr)
             << ", align: " << A.getAlignment()
             << ", section: " << Sec.getGenericSection().getName() << "\n";
    });

    auto &SecAtoms = SecToAtoms[&Sec];
    SecAtoms[A.getAddress() - Sec.getAddress()] = &A;
  }

  // Add anonymous atoms.
  for (auto &KV : Sections) {
    auto &S = KV.second;

    // Skip empty sections.
    if (S.empty())
      continue;

    // Skip sections with custom handling.
    if (CustomAtomizeFunctions.count(S.getName()))
      continue;

    auto SAI = SecToAtoms.find(&S);

    // If S is not in the SecToAtoms map then it contained no named atom. Add
    // one anonymous atom to cover the whole section.
    if (SAI == SecToAtoms.end()) {
      SecToAtoms[&S][0] = &G->addAnonymousAtom(
          S.getGenericSection(), S.getAddress(), S.getAlignment());
      continue;
    }

    // Otherwise, check whether this section had an atom covering offset zero.
    // If not, add one.
    auto &SecAtoms = SAI->second;
    if (!SecAtoms.count(0))
      SecAtoms[0] = &G->addAnonymousAtom(S.getGenericSection(), S.getAddress(),
                                         S.getAlignment());
  }

  LLVM_DEBUG(dbgs() << "MachOGraphBuilder setting atom content\n");

  // Set atom contents.
  for (auto &KV : SecToAtoms) {
    auto &S = *KV.first;
    auto &SecAtoms = KV.second;

    // Iterate the atoms in reverse order and set up their contents.
    JITTargetAddress LastAtomAddr = S.getSize();
    for (auto I = SecAtoms.rbegin(), E = SecAtoms.rend(); I != E; ++I) {
      auto Offset = I->first;
      auto &A = *I->second;
      LLVM_DEBUG({
        dbgs() << "  " << A << " to [ " << S.getAddress() + Offset << " .. "
               << S.getAddress() + LastAtomAddr << " ]\n";
      });
      if (S.isZeroFill())
        A.setZeroFill(LastAtomAddr - Offset);
      else
        A.setContent(S.getContent().substr(Offset, LastAtomAddr - Offset));
      LastAtomAddr = Offset;
    }
  }

  return Error::success();
}

Error MachOAtomGraphBuilder::addAtoms() {
  // Add all named atoms.
  if (auto Err = addNonCustomAtoms())
    return Err;

  // Process special sections.
  for (auto &KV : Sections) {
    auto &S = KV.second;
    auto HI = CustomAtomizeFunctions.find(S.getGenericSection().getName());
    if (HI != CustomAtomizeFunctions.end()) {
      auto &Atomize = HI->second;
      if (auto Err = Atomize(S))
        return Err;
    }
  }

  return Error::success();
}

} // end namespace jitlink
} // end namespace llvm
