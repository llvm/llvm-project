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

bool MachOAtomGraphBuilder::areLayoutLocked(const Atom &A, const Atom &B) {
  // If these atoms are the same then they're trivially "locked".
  if (&A == &B)
    return true;

  // If A and B are different, check whether either is undefined. (in which
  // case they are not locked).
  if (!A.isDefined() || !B.isDefined())
    return false;

  // A and B are different, but they're both defined atoms. We need to check
  // whether they're part of the same alt_entry chain.
  auto &DA = static_cast<const DefinedAtom &>(A);
  auto &DB = static_cast<const DefinedAtom &>(B);

  auto AStartItr = AltEntryStarts.find(&DA);
  if (AStartItr == AltEntryStarts.end()) // If A is not in a chain bail out.
    return false;

  auto BStartItr = AltEntryStarts.find(&DB);
  if (BStartItr == AltEntryStarts.end()) // If B is not in a chain bail out.
    return false;

  // A and B are layout locked if they're in the same chain.
  return AStartItr->second == BStartItr->second;
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
    auto &GenericSection = G->createSection("<common>", 1, Prot, true);
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

    unsigned SectionIndex = SecRef.getIndex() + 1;

    uint32_t Align = SecRef.getAlignment();
    if (!isPowerOf2_32(Align))
      return make_error<JITLinkError>("Section " + Name +
                                      " has non-power-of-2 "
                                      "alignment");

    // FIXME: Get real section permissions
    // How, exactly, on MachO?
    sys::Memory::ProtectionFlags Prot;
    if (SecRef.isText())
      Prot = static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                       sys::Memory::MF_EXEC);
    else
      Prot = static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                       sys::Memory::MF_WRITE);

    auto &GenericSection = G->createSection(Name, Align, Prot, SecRef.isBSS());

    LLVM_DEBUG({
      dbgs() << "Adding section " << Name << ": "
             << format("0x%016" PRIx64, SecRef.getAddress())
             << ", align: " << SecRef.getAlignment() << "\n";
    });

    assert(!Sections.count(SectionIndex) && "Section index already in use");

    auto &MachOSec =
        Sections
            .try_emplace(SectionIndex, GenericSection, SecRef.getAddress(),
                         SecRef.getAlignment())
            .first->second;

    if (!SecRef.isVirtual()) {
      // If this section has content then record it.
      StringRef Content;
      if (auto EC = SecRef.getContents(Content))
        return errorCodeToError(EC);
      if (Content.size() != SecRef.getSize())
        return make_error<JITLinkError>("Section content size does not match "
                                        "declared size for " +
                                        Name);
      MachOSec.setContent(Content);
    } else {
      // If this is a zero-fill section then just record the size.
      MachOSec.setZeroFill(SecRef.getSize());
    }

    uint32_t SectionFlags =
        Obj.is64Bit() ? Obj.getSection64(SecRef.getRawDataRefImpl()).flags
                      : Obj.getSection(SecRef.getRawDataRefImpl()).flags;

    MachOSec.setNoDeadStrip(SectionFlags & MachO::S_ATTR_NO_DEAD_STRIP);
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
  std::vector<DefinedAtom *> AltEntryAtoms;

  DenseSet<StringRef> ProcessedSymbols; // Used to check for duplicate defs.

  for (auto SymI = Obj.symbol_begin(), SymE = Obj.symbol_end(); SymI != SymE;
       ++SymI) {
    object::SymbolRef Sym(SymI->getRawDataRefImpl(), &Obj);

    auto Name = Sym.getName();
    if (!Name)
      return Name.takeError();

    // Bail out on duplicate definitions: There should never be more than one
    // definition for a symbol in a given object file.
    if (ProcessedSymbols.count(*Name))
      return make_error<JITLinkError>("Duplicate definition within object: " +
                                      *Name);
    else
      ProcessedSymbols.insert(*Name);

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

    auto &DA = G->addDefinedAtom(Sec.getGenericSection(), *Name, *Addr,
                                 std::max(Sym.getAlignment(), 1U));

    DA.setGlobal(Flags & object::SymbolRef::SF_Global);
    DA.setExported(Flags & object::SymbolRef::SF_Exported);
    DA.setWeak(Flags & object::SymbolRef::SF_Weak);

    DA.setCallable(*SymType & object::SymbolRef::ST_Function);

    // Check NDesc flags.
    {
      uint16_t NDesc = 0;
      if (Obj.is64Bit())
        NDesc = Obj.getSymbol64TableEntry(SymI->getRawDataRefImpl()).n_desc;
      else
        NDesc = Obj.getSymbolTableEntry(SymI->getRawDataRefImpl()).n_desc;

      // Record atom for alt-entry post-processing (where the layout-next
      // constraints will be added).
      if (NDesc & MachO::N_ALT_ENTRY)
        AltEntryAtoms.push_back(&DA);

      // If this atom has a no-dead-strip attr attached then mark it live.
      if (NDesc & MachO::N_NO_DEAD_STRIP)
        DA.setLive(true);
    }

    LLVM_DEBUG({
      dbgs() << "  Added " << *Name
             << " addr: " << format("0x%016" PRIx64, *Addr)
             << ", align: " << DA.getAlignment()
             << ", section: " << Sec.getGenericSection().getName() << "\n";
    });

    auto &SecAtoms = SecToAtoms[&Sec];
    SecAtoms[DA.getAddress() - Sec.getAddress()] = &DA;
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

  // Set atom contents and any section-based flags.
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

      // If the section has no-dead-strip set then mark the atom as live.
      if (S.isNoDeadStrip())
        A.setLive(true);

      LastAtomAddr = Offset;
    }
  }

  LLVM_DEBUG(dbgs() << "Adding alt-entry starts\n");

  // Sort alt-entry atoms by address in ascending order.
  llvm::sort(AltEntryAtoms.begin(), AltEntryAtoms.end(),
             [](const DefinedAtom *LHS, const DefinedAtom *RHS) {
               return LHS->getAddress() < RHS->getAddress();
             });

  // Process alt-entry atoms in address order to build the table of alt-entry
  // atoms to alt-entry chain starts.
  for (auto *DA : AltEntryAtoms) {
    assert(!AltEntryStarts.count(DA) && "Duplicate entry in AltEntryStarts");

    // DA is an alt-entry atom. Look for the predecessor atom that it is locked
    // to, bailing out if we do not find one.
    auto AltEntryPred = G->findAtomByAddress(DA->getAddress() - 1);
    if (!AltEntryPred)
      return AltEntryPred.takeError();

    // Add a LayoutNext edge from the predecessor to this atom.
    AltEntryPred->setLayoutNext(*DA);

    // Check to see whether the predecessor itself is an alt-entry atom.
    auto AltEntryStartItr = AltEntryStarts.find(&*AltEntryPred);
    if (AltEntryStartItr != AltEntryStarts.end()) {
      // If the predecessor was an alt-entry atom then re-use its value.
      LLVM_DEBUG({
        dbgs() << "  " << *DA << " -> " << *AltEntryStartItr->second
               << " (based on existing entry for " << *AltEntryPred << ")\n";
      });
      AltEntryStarts[DA] = AltEntryStartItr->second;
    } else {
      // If the predecessor does not have an entry then add an entry for this
      // atom (i.e. the alt_entry atom) and a self-reference entry for the
      /// predecessory atom that is the start of this chain.
      LLVM_DEBUG({
        dbgs() << "  " << *AltEntryPred << " -> " << *AltEntryPred << "\n"
               << "  " << *DA << " -> " << *AltEntryPred << "\n";
      });
      AltEntryStarts[&*AltEntryPred] = &*AltEntryPred;
      AltEntryStarts[DA] = &*AltEntryPred;
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
