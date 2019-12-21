//=--------- MachOLinkGraphBuilder.cpp - MachO LinkGraph builder ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic MachO LinkGraph buliding code.
//
//===----------------------------------------------------------------------===//

#include "MachOLinkGraphBuilder.h"

#define DEBUG_TYPE "jitlink"

static const char *CommonSectionName = "__common";

namespace llvm {
namespace jitlink {

MachOLinkGraphBuilder::~MachOLinkGraphBuilder() {}

Expected<std::unique_ptr<LinkGraph>> MachOLinkGraphBuilder::buildGraph() {

  // Sanity check: we only operate on relocatable objects.
  if (!Obj.isRelocatableObject())
    return make_error<JITLinkError>("Object is not a relocatable MachO");

  if (auto Err = createNormalizedSections())
    return std::move(Err);

  if (auto Err = createNormalizedSymbols())
    return std::move(Err);

  if (auto Err = graphifyRegularSymbols())
    return std::move(Err);

  if (auto Err = graphifySectionsWithCustomParsers())
    return std::move(Err);

  if (auto Err = addRelocations())
    return std::move(Err);

  return std::move(G);
}

MachOLinkGraphBuilder::MachOLinkGraphBuilder(const object::MachOObjectFile &Obj)
    : Obj(Obj),
      G(std::make_unique<LinkGraph>(Obj.getFileName(), getPointerSize(Obj),
                                    getEndianness(Obj))) {}

void MachOLinkGraphBuilder::addCustomSectionParser(
    StringRef SectionName, SectionParserFunction Parser) {
  assert(!CustomSectionParserFunctions.count(SectionName) &&
         "Custom parser for this section already exists");
  CustomSectionParserFunctions[SectionName] = std::move(Parser);
}

Linkage MachOLinkGraphBuilder::getLinkage(uint16_t Desc) {
  if ((Desc & MachO::N_WEAK_DEF) || (Desc & MachO::N_WEAK_REF))
    return Linkage::Weak;
  return Linkage::Strong;
}

Scope MachOLinkGraphBuilder::getScope(StringRef Name, uint8_t Type) {
  if (Name.startswith("l"))
    return Scope::Local;
  if (Type & MachO::N_PEXT)
    return Scope::Hidden;
  if (Type & MachO::N_EXT)
    return Scope::Default;
  return Scope::Local;
}

bool MachOLinkGraphBuilder::isAltEntry(const NormalizedSymbol &NSym) {
  return NSym.Desc & MachO::N_ALT_ENTRY;
}

unsigned
MachOLinkGraphBuilder::getPointerSize(const object::MachOObjectFile &Obj) {
  return Obj.is64Bit() ? 8 : 4;
}

support::endianness
MachOLinkGraphBuilder::getEndianness(const object::MachOObjectFile &Obj) {
  return Obj.isLittleEndian() ? support::little : support::big;
}

Section &MachOLinkGraphBuilder::getCommonSection() {
  if (!CommonSection) {
    auto Prot = static_cast<sys::Memory::ProtectionFlags>(
        sys::Memory::MF_READ | sys::Memory::MF_WRITE);
    CommonSection = &G->createSection(CommonSectionName, Prot);
  }
  return *CommonSection;
}

Error MachOLinkGraphBuilder::createNormalizedSections() {
  // Build normalized sections. Verifies that section data is in-range (for
  // sections with content) and that address ranges are non-overlapping.

  LLVM_DEBUG(dbgs() << "Creating normalized sections...\n");

  for (auto &SecRef : Obj.sections()) {
    NormalizedSection NSec;
    uint32_t DataOffset = 0;

    auto SecIndex = Obj.getSectionIndex(SecRef.getRawDataRefImpl());

    auto Name = SecRef.getName();
    if (!Name)
      return Name.takeError();

    if (Obj.is64Bit()) {
      const MachO::section_64 &Sec64 =
          Obj.getSection64(SecRef.getRawDataRefImpl());

      NSec.Address = Sec64.addr;
      NSec.Size = Sec64.size;
      NSec.Alignment = 1ULL << Sec64.align;
      NSec.Flags = Sec64.flags;
      DataOffset = Sec64.offset;
    } else {
      const MachO::section &Sec32 = Obj.getSection(SecRef.getRawDataRefImpl());
      NSec.Address = Sec32.addr;
      NSec.Size = Sec32.size;
      NSec.Alignment = 1ULL << Sec32.align;
      NSec.Flags = Sec32.flags;
      DataOffset = Sec32.offset;
    }

    LLVM_DEBUG({
      dbgs() << "  " << *Name << ": " << formatv("{0:x16}", NSec.Address)
             << " -- " << formatv("{0:x16}", NSec.Address + NSec.Size)
             << ", align: " << NSec.Alignment << ", index: " << SecIndex
             << "\n";
    });

    // Get the section data if any.
    {
      unsigned SectionType = NSec.Flags & MachO::SECTION_TYPE;
      if (SectionType != MachO::S_ZEROFILL &&
          SectionType != MachO::S_GB_ZEROFILL) {

        if (DataOffset + NSec.Size > Obj.getData().size())
          return make_error<JITLinkError>(
              "Section data extends past end of file");

        NSec.Data = Obj.getData().data() + DataOffset;
      }
    }

    // Get prot flags.
    // FIXME: Make sure this test is correct (it's probably missing cases
    // as-is).
    sys::Memory::ProtectionFlags Prot;
    if (NSec.Flags & MachO::S_ATTR_PURE_INSTRUCTIONS)
      Prot = static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                       sys::Memory::MF_EXEC);
    else
      Prot = static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                       sys::Memory::MF_WRITE);

    NSec.GraphSection = &G->createSection(*Name, Prot);
    IndexToSection.insert(std::make_pair(SecIndex, std::move(NSec)));
  }

  std::vector<NormalizedSection *> Sections;
  Sections.reserve(IndexToSection.size());
  for (auto &KV : IndexToSection)
    Sections.push_back(&KV.second);

  // If we didn't end up creating any sections then bail out. The code below
  // assumes that we have at least one section.
  if (Sections.empty())
    return Error::success();

  llvm::sort(Sections,
             [](const NormalizedSection *LHS, const NormalizedSection *RHS) {
               assert(LHS && RHS && "Null section?");
               if (LHS->Address != RHS->Address)
                 return LHS->Address < RHS->Address;
               return LHS->Size < RHS->Size;
             });

  for (unsigned I = 0, E = Sections.size() - 1; I != E; ++I) {
    auto &Cur = *Sections[I];
    auto &Next = *Sections[I + 1];
    if (Next.Address < Cur.Address + Cur.Size)
      return make_error<JITLinkError>(
          "Address range for section " + Cur.GraphSection->getName() +
          formatv(" [ {0:x16} -- {1:x16} ] ", Cur.Address,
                  Cur.Address + Cur.Size) +
          "overlaps " +
          formatv(" [ {0:x16} -- {1:x16} ] ", Next.Address,
                  Next.Address + Next.Size));
  }

  return Error::success();
}

Error MachOLinkGraphBuilder::createNormalizedSymbols() {
  LLVM_DEBUG(dbgs() << "Creating normalized symbols...\n");

  for (auto &SymRef : Obj.symbols()) {

    unsigned SymbolIndex = Obj.getSymbolIndex(SymRef.getRawDataRefImpl());
    uint64_t Value;
    uint32_t NStrX;
    uint8_t Type;
    uint8_t Sect;
    uint16_t Desc;

    if (Obj.is64Bit()) {
      const MachO::nlist_64 &NL64 =
          Obj.getSymbol64TableEntry(SymRef.getRawDataRefImpl());
      Value = NL64.n_value;
      NStrX = NL64.n_strx;
      Type = NL64.n_type;
      Sect = NL64.n_sect;
      Desc = NL64.n_desc;
    } else {
      const MachO::nlist &NL32 =
          Obj.getSymbolTableEntry(SymRef.getRawDataRefImpl());
      Value = NL32.n_value;
      NStrX = NL32.n_strx;
      Type = NL32.n_type;
      Sect = NL32.n_sect;
      Desc = NL32.n_desc;
    }

    // Skip stabs.
    // FIXME: Are there other symbols we should be skipping?
    if (Type & MachO::N_STAB)
      continue;

    Optional<StringRef> Name;
    if (NStrX) {
      if (auto NameOrErr = SymRef.getName())
        Name = *NameOrErr;
      else
        return NameOrErr.takeError();
    }

    LLVM_DEBUG({
      dbgs() << "  ";
      if (!Name)
        dbgs() << "<anonymous symbol>";
      else
        dbgs() << *Name;
      dbgs() << ": value = " << formatv("{0:x16}", Value)
             << ", type = " << formatv("{0:x2}", Type)
             << ", desc = " << formatv("{0:x4}", Desc) << ", sect = ";
      if (Sect)
        dbgs() << static_cast<unsigned>(Sect - 1);
      else
        dbgs() << "none";
      dbgs() << "\n";
    });

    // If this symbol has a section, sanity check that the addresses line up.
    NormalizedSection *NSec = nullptr;
    if (Sect != 0) {
      if (auto NSecOrErr = findSectionByIndex(Sect - 1))
        NSec = &*NSecOrErr;
      else
        return NSecOrErr.takeError();

      if (Value < NSec->Address || Value > NSec->Address + NSec->Size)
        return make_error<JITLinkError>("Symbol address does not fall within "
                                        "section");
    }

    IndexToSymbol[SymbolIndex] =
        &createNormalizedSymbol(*Name, Value, Type, Sect, Desc,
                                getLinkage(Type), getScope(*Name, Type));
  }

  return Error::success();
}

void MachOLinkGraphBuilder::addSectionStartSymAndBlock(
    Section &GraphSec, uint64_t Address, const char *Data, uint64_t Size,
    uint32_t Alignment, bool IsLive) {
  Block &B =
      Data ? G->createContentBlock(GraphSec, StringRef(Data, Size), Address,
                                   Alignment, 0)
           : G->createZeroFillBlock(GraphSec, Size, Address, Alignment, 0);
  auto &Sym = G->addAnonymousSymbol(B, 0, Size, false, IsLive);
  assert(!AddrToCanonicalSymbol.count(Sym.getAddress()) &&
         "Anonymous block start symbol clashes with existing symbol address");
  AddrToCanonicalSymbol[Sym.getAddress()] = &Sym;
}

Error MachOLinkGraphBuilder::graphifyRegularSymbols() {

  LLVM_DEBUG(dbgs() << "Creating graph symbols...\n");

  /// We only have 256 section indexes: Use a vector rather than a map.
  std::vector<std::vector<NormalizedSymbol *>> SecIndexToSymbols;
  SecIndexToSymbols.resize(256);

  // Create commons, externs, and absolutes, and partition all other symbols by
  // section.
  for (auto &KV : IndexToSymbol) {
    auto &NSym = *KV.second;

    switch (NSym.Type & MachO::N_TYPE) {
    case MachO::N_UNDF:
      if (NSym.Value) {
        if (!NSym.Name)
          return make_error<JITLinkError>("Anonymous common symbol at index " +
                                          Twine(KV.first));
        NSym.GraphSymbol = &G->addCommonSymbol(
            *NSym.Name, NSym.S, getCommonSection(), 0, NSym.Value,
            1ull << MachO::GET_COMM_ALIGN(NSym.Desc),
            NSym.Desc & MachO::N_NO_DEAD_STRIP);
      } else {
        if (!NSym.Name)
          return make_error<JITLinkError>("Anonymous external symbol at "
                                          "index " +
                                          Twine(KV.first));
        NSym.GraphSymbol = &G->addExternalSymbol(
            *NSym.Name, 0,
            NSym.Desc & MachO::N_WEAK_REF ? Linkage::Weak : Linkage::Strong);
      }
      break;
    case MachO::N_ABS:
      if (!NSym.Name)
        return make_error<JITLinkError>("Anonymous absolute symbol at index " +
                                        Twine(KV.first));
      NSym.GraphSymbol = &G->addAbsoluteSymbol(
          *NSym.Name, NSym.Value, 0, Linkage::Strong, Scope::Default,
          NSym.Desc & MachO::N_NO_DEAD_STRIP);
      break;
    case MachO::N_SECT:
      SecIndexToSymbols[NSym.Sect - 1].push_back(&NSym);
      break;
    case MachO::N_PBUD:
      return make_error<JITLinkError>(
          "Unupported N_PBUD symbol " +
          (NSym.Name ? ("\"" + *NSym.Name + "\"") : Twine("<anon>")) +
          " at index " + Twine(KV.first));
    case MachO::N_INDR:
      return make_error<JITLinkError>(
          "Unupported N_INDR symbol " +
          (NSym.Name ? ("\"" + *NSym.Name + "\"") : Twine("<anon>")) +
          " at index " + Twine(KV.first));
    default:
      return make_error<JITLinkError>(
          "Unrecognized symbol type " + Twine(NSym.Type & MachO::N_TYPE) +
          " for symbol " +
          (NSym.Name ? ("\"" + *NSym.Name + "\"") : Twine("<anon>")) +
          " at index " + Twine(KV.first));
    }
  }

  // Loop over sections performing regular graphification for those that
  // don't have custom parsers.
  for (auto &KV : IndexToSection) {
    auto SecIndex = KV.first;
    auto &NSec = KV.second;

    // Skip sections with custom parsers.
    if (CustomSectionParserFunctions.count(NSec.GraphSection->getName())) {
      LLVM_DEBUG({
        dbgs() << "  Skipping section " << NSec.GraphSection->getName()
               << " as it has a custom parser.\n";
      });
      continue;
    } else
      LLVM_DEBUG({
        dbgs() << "  Processing section " << NSec.GraphSection->getName()
               << "...\n";
      });

    bool SectionIsNoDeadStrip = NSec.Flags & MachO::S_ATTR_NO_DEAD_STRIP;
    bool SectionIsText = NSec.Flags & MachO::S_ATTR_PURE_INSTRUCTIONS;

    auto &SecNSymStack = SecIndexToSymbols[SecIndex];

    // If this section is non-empty but there are no symbols covering it then
    // create one block and anonymous symbol to cover the entire section.
    if (SecNSymStack.empty()) {
      if (NSec.Size > 0) {
        LLVM_DEBUG({
          dbgs() << "    Section non-empty, but contains no symbols. "
                    "Creating anonymous block to cover "
                 << formatv("{0:x16}", NSec.Address) << " -- "
                 << formatv("{0:x16}", NSec.Address + NSec.Size) << "\n";
        });
        addSectionStartSymAndBlock(*NSec.GraphSection, NSec.Address, NSec.Data,
                                   NSec.Size, NSec.Alignment,
                                   SectionIsNoDeadStrip);
      } else
        LLVM_DEBUG({
          dbgs() << "    Section empty and contains no symbols. Skipping.\n";
        });
      continue;
    }

    // Sort the symbol stack in by address, alt-entry status, scope, and name.
    // We sort in reverse order so that symbols will be visited in the right
    // order when we pop off the stack below.
    llvm::sort(SecNSymStack, [](const NormalizedSymbol *LHS,
                                const NormalizedSymbol *RHS) {
      if (LHS->Value != RHS->Value)
        return LHS->Value > RHS->Value;
      if (isAltEntry(*LHS) != isAltEntry(*RHS))
        return isAltEntry(*RHS);
      if (LHS->S != RHS->S)
        return static_cast<uint8_t>(LHS->S) < static_cast<uint8_t>(RHS->S);
      return LHS->Name < RHS->Name;
    });

    // The first symbol in a section can not be an alt-entry symbol.
    if (!SecNSymStack.empty() && isAltEntry(*SecNSymStack.back()))
      return make_error<JITLinkError>(
          "First symbol in " + NSec.GraphSection->getName() + " is alt-entry");

    // If the section is non-empty but there is no symbol covering the start
    // address then add an anonymous one.
    if (SecNSymStack.back()->Value != NSec.Address) {
      auto AnonBlockSize = SecNSymStack.back()->Value - NSec.Address;
      LLVM_DEBUG({
        dbgs() << "    Section start not covered by symbol. "
               << "Creating anonymous block to cover [ "
               << formatv("{0:x16}", NSec.Address) << " -- "
               << formatv("{0:x16}", NSec.Address + AnonBlockSize) << " ]\n";
      });
      addSectionStartSymAndBlock(*NSec.GraphSection, NSec.Address, NSec.Data,
                                 AnonBlockSize, NSec.Alignment,
                                 SectionIsNoDeadStrip);
    }

    // Visit section symbols in order by popping off the reverse-sorted stack,
    // building blocks for each alt-entry chain and creating symbols as we go.
    while (!SecNSymStack.empty()) {
      SmallVector<NormalizedSymbol *, 8> BlockSyms;

      BlockSyms.push_back(SecNSymStack.back());
      SecNSymStack.pop_back();
      while (!SecNSymStack.empty() &&
             (isAltEntry(*SecNSymStack.back()) ||
              SecNSymStack.back()->Value == BlockSyms.back()->Value)) {
        BlockSyms.push_back(SecNSymStack.back());
        SecNSymStack.pop_back();
      }

      // BlockNSyms now contains the block symbols in reverse canonical order.
      JITTargetAddress BlockStart = BlockSyms.front()->Value;
      JITTargetAddress BlockEnd = SecNSymStack.empty()
                                      ? NSec.Address + NSec.Size
                                      : SecNSymStack.back()->Value;
      JITTargetAddress BlockOffset = BlockStart - NSec.Address;
      JITTargetAddress BlockSize = BlockEnd - BlockStart;

      LLVM_DEBUG({
        dbgs() << "    Creating block for " << formatv("{0:x16}", BlockStart)
               << " -- " << formatv("{0:x16}", BlockEnd) << ": "
               << NSec.GraphSection->getName() << " + "
               << formatv("{0:x16}", BlockOffset) << " with "
               << BlockSyms.size() << " symbol(s)...\n";
      });

      Block &B =
          NSec.Data
              ? G->createContentBlock(
                    *NSec.GraphSection,
                    StringRef(NSec.Data + BlockOffset, BlockSize), BlockStart,
                    NSec.Alignment, BlockStart % NSec.Alignment)
              : G->createZeroFillBlock(*NSec.GraphSection, BlockSize,
                                       BlockStart, NSec.Alignment,
                                       BlockStart % NSec.Alignment);

      Optional<JITTargetAddress> LastCanonicalAddr;
      JITTargetAddress SymEnd = BlockEnd;
      while (!BlockSyms.empty()) {
        auto &NSym = *BlockSyms.back();
        BlockSyms.pop_back();

        bool SymLive =
            (NSym.Desc & MachO::N_NO_DEAD_STRIP) || SectionIsNoDeadStrip;

        LLVM_DEBUG({
          dbgs() << "      " << formatv("{0:x16}", NSym.Value) << " -- "
                 << formatv("{0:x16}", SymEnd) << ": ";
          if (!NSym.Name)
            dbgs() << "<anonymous symbol>";
          else
            dbgs() << NSym.Name;
          if (SymLive)
            dbgs() << " [no-dead-strip]";
          if (LastCanonicalAddr == NSym.Value)
            dbgs() << " [non-canonical]";
          dbgs() << "\n";
        });

        auto &Sym =
            NSym.Name
                ? G->addDefinedSymbol(B, NSym.Value - BlockStart, *NSym.Name,
                                      SymEnd - NSym.Value, NSym.L, NSym.S,
                                      SectionIsText, SymLive)
                : G->addAnonymousSymbol(B, NSym.Value - BlockStart,
                                        SymEnd - NSym.Value, SectionIsText,
                                        SymLive);
        NSym.GraphSymbol = &Sym;
        if (LastCanonicalAddr != Sym.getAddress()) {
          if (LastCanonicalAddr)
            SymEnd = *LastCanonicalAddr;
          LastCanonicalAddr = Sym.getAddress();
          setCanonicalSymbol(Sym);
        }
      }
    }
  }

  return Error::success();
}

Error MachOLinkGraphBuilder::graphifySectionsWithCustomParsers() {
  // Graphify special sections.
  for (auto &KV : IndexToSection) {
    auto &NSec = KV.second;

    auto HI = CustomSectionParserFunctions.find(NSec.GraphSection->getName());
    if (HI != CustomSectionParserFunctions.end()) {
      auto &Parse = HI->second;
      if (auto Err = Parse(NSec))
        return Err;
    }
  }

  return Error::success();
}

} // end namespace jitlink
} // end namespace llvm
