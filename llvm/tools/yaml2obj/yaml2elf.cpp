//===- yaml2elf - Convert YAML to a ELF object file -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The ELF component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "yaml2obj.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// This class is used to build up a contiguous binary blob while keeping
// track of an offset in the output (which notionally begins at
// `InitialOffset`).
namespace {
class ContiguousBlobAccumulator {
  const uint64_t InitialOffset;
  SmallVector<char, 128> Buf;
  raw_svector_ostream OS;

  /// \returns The new offset.
  uint64_t padToAlignment(unsigned Align) {
    if (Align == 0)
      Align = 1;
    uint64_t CurrentOffset = InitialOffset + OS.tell();
    uint64_t AlignedOffset = alignTo(CurrentOffset, Align);
    OS.write_zeros(AlignedOffset - CurrentOffset);
    return AlignedOffset; // == CurrentOffset;
  }

public:
  ContiguousBlobAccumulator(uint64_t InitialOffset_)
      : InitialOffset(InitialOffset_), Buf(), OS(Buf) {}
  template <class Integer>
  raw_ostream &getOSAndAlignedOffset(Integer &Offset, unsigned Align) {
    Offset = padToAlignment(Align);
    return OS;
  }
  void writeBlobToStream(raw_ostream &Out) { Out << OS.str(); }
};
} // end anonymous namespace

// Used to keep track of section and symbol names, so that in the YAML file
// sections and symbols can be referenced by name instead of by index.
namespace {
class NameToIdxMap {
  StringMap<int> Map;
public:
  /// \returns true if name is already present in the map.
  bool addName(StringRef Name, unsigned i) {
    return !Map.insert(std::make_pair(Name, (int)i)).second;
  }
  /// \returns true if name is not present in the map
  bool lookup(StringRef Name, unsigned &Idx) const {
    StringMap<int>::const_iterator I = Map.find(Name);
    if (I == Map.end())
      return true;
    Idx = I->getValue();
    return false;
  }
  /// asserts if name is not present in the map
  unsigned get(StringRef Name) const {
    unsigned Idx = 0;
    auto missing = lookup(Name, Idx);
    (void)missing;
    assert(!missing && "Expected section not found in index");
    return Idx;
  }
  unsigned size() const { return Map.size(); }
};
} // end anonymous namespace

template <class T>
static size_t arrayDataSize(ArrayRef<T> A) {
  return A.size() * sizeof(T);
}

template <class T>
static void writeArrayData(raw_ostream &OS, ArrayRef<T> A) {
  OS.write((const char *)A.data(), arrayDataSize(A));
}

template <class T>
static void zero(T &Obj) {
  memset(&Obj, 0, sizeof(Obj));
}

namespace {
/// "Single point of truth" for the ELF file construction.
/// TODO: This class still has a ways to go before it is truly a "single
/// point of truth".
template <class ELFT>
class ELFState {
  typedef typename ELFT::Ehdr Elf_Ehdr;
  typedef typename ELFT::Phdr Elf_Phdr;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::Relr Elf_Relr;
  typedef typename ELFT::Dyn Elf_Dyn;

  enum class SymtabType { Static, Dynamic };

  /// The future ".strtab" section.
  StringTableBuilder DotStrtab{StringTableBuilder::ELF};

  /// The future ".shstrtab" section.
  StringTableBuilder DotShStrtab{StringTableBuilder::ELF};

  /// The future ".dynstr" section.
  StringTableBuilder DotDynstr{StringTableBuilder::ELF};

  NameToIdxMap SN2I;
  NameToIdxMap SymN2I;
  const ELFYAML::Object &Doc;

  bool buildSectionIndex();
  bool buildSymbolIndex(ArrayRef<ELFYAML::Symbol> Symbols);
  void initELFHeader(Elf_Ehdr &Header);
  void initProgramHeaders(std::vector<Elf_Phdr> &PHeaders);
  bool initSectionHeaders(std::vector<Elf_Shdr> &SHeaders,
                          ContiguousBlobAccumulator &CBA);
  void initSymtabSectionHeader(Elf_Shdr &SHeader, SymtabType STType,
                               ContiguousBlobAccumulator &CBA);
  void initStrtabSectionHeader(Elf_Shdr &SHeader, StringRef Name,
                               StringTableBuilder &STB,
                               ContiguousBlobAccumulator &CBA);
  void setProgramHeaderLayout(std::vector<Elf_Phdr> &PHeaders,
                              std::vector<Elf_Shdr> &SHeaders);
  void addSymbols(ArrayRef<ELFYAML::Symbol> Symbols, std::vector<Elf_Sym> &Syms,
                  const StringTableBuilder &Strtab);
  bool writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::RawContentSection &Section,
                           ContiguousBlobAccumulator &CBA);
  bool writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::RelocationSection &Section,
                           ContiguousBlobAccumulator &CBA);
  bool writeSectionContent(Elf_Shdr &SHeader, const ELFYAML::Group &Group,
                           ContiguousBlobAccumulator &CBA);
  bool writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::SymverSection &Section,
                           ContiguousBlobAccumulator &CBA);
  bool writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::VerneedSection &Section,
                           ContiguousBlobAccumulator &CBA);
  bool writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::VerdefSection &Section,
                           ContiguousBlobAccumulator &CBA);
  bool writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::MipsABIFlags &Section,
                           ContiguousBlobAccumulator &CBA);
  bool writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::DynamicSection &Section,
                           ContiguousBlobAccumulator &CBA);
  std::vector<StringRef> implicitSectionNames() const;

  // - SHT_NULL entry (placed first, i.e. 0'th entry)
  // - symbol table (.symtab) (defaults to after last yaml section)
  // - string table (.strtab) (defaults to after .symtab)
  // - section header string table (.shstrtab) (defaults to after .strtab)
  // - dynamic symbol table (.dynsym) (defaults to after .shstrtab)
  // - dynamic string table (.dynstr) (defaults to after .dynsym)
  unsigned getDotSymTabSecNo() const { return SN2I.get(".symtab"); }
  unsigned getDotStrTabSecNo() const { return SN2I.get(".strtab"); }
  unsigned getDotShStrTabSecNo() const { return SN2I.get(".shstrtab"); }
  unsigned getDotDynSymSecNo() const { return SN2I.get(".dynsym"); }
  unsigned getDotDynStrSecNo() const { return SN2I.get(".dynstr"); }
  unsigned getSectionCount() const { return SN2I.size() + 1; }

  ELFState(const ELFYAML::Object &D) : Doc(D) {}

public:
  static int writeELF(raw_ostream &OS, const ELFYAML::Object &Doc);

private:
  void finalizeStrings();
};
} // end anonymous namespace

template <class ELFT>
void ELFState<ELFT>::initELFHeader(Elf_Ehdr &Header) {
  using namespace llvm::ELF;
  zero(Header);
  Header.e_ident[EI_MAG0] = 0x7f;
  Header.e_ident[EI_MAG1] = 'E';
  Header.e_ident[EI_MAG2] = 'L';
  Header.e_ident[EI_MAG3] = 'F';
  Header.e_ident[EI_CLASS] = ELFT::Is64Bits ? ELFCLASS64 : ELFCLASS32;
  Header.e_ident[EI_DATA] = Doc.Header.Data;
  Header.e_ident[EI_VERSION] = EV_CURRENT;
  Header.e_ident[EI_OSABI] = Doc.Header.OSABI;
  Header.e_ident[EI_ABIVERSION] = Doc.Header.ABIVersion;
  Header.e_type = Doc.Header.Type;
  Header.e_machine = Doc.Header.Machine;
  Header.e_version = EV_CURRENT;
  Header.e_entry = Doc.Header.Entry;
  Header.e_phoff = sizeof(Header);
  Header.e_flags = Doc.Header.Flags;
  Header.e_ehsize = sizeof(Elf_Ehdr);
  Header.e_phentsize = sizeof(Elf_Phdr);
  Header.e_phnum = Doc.ProgramHeaders.size();
  Header.e_shentsize = sizeof(Elf_Shdr);
  // Immediately following the ELF header and program headers.
  Header.e_shoff =
      sizeof(Header) + sizeof(Elf_Phdr) * Doc.ProgramHeaders.size();
  Header.e_shnum = getSectionCount();
  Header.e_shstrndx = getDotShStrTabSecNo();
}

template <class ELFT>
void ELFState<ELFT>::initProgramHeaders(std::vector<Elf_Phdr> &PHeaders) {
  for (const auto &YamlPhdr : Doc.ProgramHeaders) {
    Elf_Phdr Phdr;
    Phdr.p_type = YamlPhdr.Type;
    Phdr.p_flags = YamlPhdr.Flags;
    Phdr.p_vaddr = YamlPhdr.VAddr;
    Phdr.p_paddr = YamlPhdr.PAddr;
    PHeaders.push_back(Phdr);
  }
}

static bool convertSectionIndex(NameToIdxMap &SN2I, StringRef SecName,
                                StringRef IndexSrc, unsigned &IndexDest) {
  if (SN2I.lookup(IndexSrc, IndexDest) && !to_integer(IndexSrc, IndexDest)) {
    WithColor::error() << "Unknown section referenced: '" << IndexSrc
                       << "' at YAML section '" << SecName << "'.\n";
    return false;
  }
  return true;
}

template <class ELFT>
bool ELFState<ELFT>::initSectionHeaders(std::vector<Elf_Shdr> &SHeaders,
                                        ContiguousBlobAccumulator &CBA) {
  // Ensure SHN_UNDEF entry is present. An all-zero section header is a
  // valid SHN_UNDEF entry since SHT_NULL == 0.
  Elf_Shdr SHeader;
  zero(SHeader);
  SHeaders.push_back(SHeader);

  for (const auto &Sec : Doc.Sections) {
    zero(SHeader);
    SHeader.sh_name = DotShStrtab.getOffset(Sec->Name);
    SHeader.sh_type = Sec->Type;
    SHeader.sh_flags = Sec->Flags;
    SHeader.sh_addr = Sec->Address;
    SHeader.sh_addralign = Sec->AddressAlign;

    if (!Sec->Link.empty()) {
      unsigned Index;
      if (!convertSectionIndex(SN2I, Sec->Name, Sec->Link, Index))
        return false;
      SHeader.sh_link = Index;
    }

    if (auto S = dyn_cast<ELFYAML::RawContentSection>(Sec.get())) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::RelocationSection>(Sec.get())) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::Group>(Sec.get())) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::MipsABIFlags>(Sec.get())) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::NoBitsSection>(Sec.get())) {
      SHeader.sh_entsize = 0;
      SHeader.sh_size = S->Size;
      // SHT_NOBITS section does not have content
      // so just to setup the section offset.
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
    } else if (auto S = dyn_cast<ELFYAML::DynamicSection>(Sec.get())) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::SymverSection>(Sec.get())) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::VerneedSection>(Sec.get())) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::VerdefSection>(Sec.get())) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else
      llvm_unreachable("Unknown section type");

    SHeaders.push_back(SHeader);
  }
  return true;
}

static size_t findFirstNonGlobal(ArrayRef<ELFYAML::Symbol> Symbols) {
  for (size_t I = 0; I < Symbols.size(); ++I)
    if (Symbols[I].Binding.value != ELF::STB_LOCAL)
      return I;
  return Symbols.size();
}

template <class ELFT>
void ELFState<ELFT>::initSymtabSectionHeader(Elf_Shdr &SHeader,
                                             SymtabType STType,
                                             ContiguousBlobAccumulator &CBA) {
  zero(SHeader);
  bool IsStatic = STType == SymtabType::Static;
  SHeader.sh_name = DotShStrtab.getOffset(IsStatic ? ".symtab" : ".dynsym");
  SHeader.sh_type = IsStatic ? ELF::SHT_SYMTAB : ELF::SHT_DYNSYM;
  SHeader.sh_link = IsStatic ? getDotStrTabSecNo() : getDotDynStrSecNo();
  if (!IsStatic)
    SHeader.sh_flags |= ELF::SHF_ALLOC;

  // One greater than symbol table index of the last local symbol.
  const auto &Symbols = IsStatic ? Doc.Symbols : Doc.DynamicSymbols;
  SHeader.sh_info = findFirstNonGlobal(Symbols) + 1;
  SHeader.sh_entsize = sizeof(Elf_Sym);
  SHeader.sh_addralign = 8;

  // Get the section index ignoring the SHT_NULL section.
  unsigned SecNdx =
      IsStatic ? getDotSymTabSecNo() - 1 : getDotDynSymSecNo() - 1;
  // If the symbol table section is explicitly described in the YAML
  // then we should set the fields requested.
  if (SecNdx < Doc.Sections.size()) {
    ELFYAML::Section *Sec = Doc.Sections[SecNdx].get();
    SHeader.sh_addr = Sec->Address;
    if (auto S = dyn_cast<ELFYAML::RawContentSection>(Sec))
      SHeader.sh_info = S->Info;
  }

  std::vector<Elf_Sym> Syms;
  {
    // Ensure STN_UNDEF is present
    Elf_Sym Sym;
    zero(Sym);
    Syms.push_back(Sym);
  }

  addSymbols(Symbols, Syms, IsStatic ? DotStrtab : DotDynstr);

  writeArrayData(
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign),
      makeArrayRef(Syms));
  SHeader.sh_size = arrayDataSize(makeArrayRef(Syms));
}

template <class ELFT>
void ELFState<ELFT>::initStrtabSectionHeader(Elf_Shdr &SHeader, StringRef Name,
                                             StringTableBuilder &STB,
                                             ContiguousBlobAccumulator &CBA) {
  zero(SHeader);
  SHeader.sh_name = DotShStrtab.getOffset(Name);
  SHeader.sh_type = ELF::SHT_STRTAB;
  STB.write(CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign));
  SHeader.sh_size = STB.getSize();
  SHeader.sh_addralign = 1;

  // If .dynstr section is explicitly described in the YAML
  // then we want to use its section address.
  if (Name == ".dynstr") {
    // Take section index and ignore the SHT_NULL section.
    unsigned SecNdx = getDotDynStrSecNo() - 1;
    if (SecNdx < Doc.Sections.size())
      SHeader.sh_addr = Doc.Sections[SecNdx]->Address;

    // We assume that .dynstr is always allocatable.
    SHeader.sh_flags |= ELF::SHF_ALLOC;
  }
}

template <class ELFT>
void ELFState<ELFT>::setProgramHeaderLayout(std::vector<Elf_Phdr> &PHeaders,
                                            std::vector<Elf_Shdr> &SHeaders) {
  uint32_t PhdrIdx = 0;
  for (auto &YamlPhdr : Doc.ProgramHeaders) {
    auto &PHeader = PHeaders[PhdrIdx++];

    if (YamlPhdr.Offset) {
      PHeader.p_offset = *YamlPhdr.Offset;
    } else {
      if (YamlPhdr.Sections.size())
        PHeader.p_offset = UINT32_MAX;
      else
        PHeader.p_offset = 0;

      // Find the minimum offset for the program header.
      for (auto SecName : YamlPhdr.Sections) {
        uint32_t Index = 0;
        SN2I.lookup(SecName.Section, Index);
        const auto &SHeader = SHeaders[Index];
        PHeader.p_offset = std::min(PHeader.p_offset, SHeader.sh_offset);
      }
    }

    // Find the maximum offset of the end of a section in order to set p_filesz,
    // if not set explicitly.
    if (YamlPhdr.FileSize) {
      PHeader.p_filesz = *YamlPhdr.FileSize;
    } else {
      PHeader.p_filesz = 0;
      for (auto SecName : YamlPhdr.Sections) {
        uint32_t Index = 0;
        SN2I.lookup(SecName.Section, Index);
        const auto &SHeader = SHeaders[Index];
        uint64_t EndOfSection;
        if (SHeader.sh_type == llvm::ELF::SHT_NOBITS)
          EndOfSection = SHeader.sh_offset;
        else
          EndOfSection = SHeader.sh_offset + SHeader.sh_size;
        uint64_t EndOfSegment = PHeader.p_offset + PHeader.p_filesz;
        EndOfSegment = std::max(EndOfSegment, EndOfSection);
        PHeader.p_filesz = EndOfSegment - PHeader.p_offset;
      }
    }

    // If not set explicitly, find the memory size by adding the size of
    // sections at the end of the segment. These should be empty (size of zero)
    // and NOBITS sections.
    if (YamlPhdr.MemSize) {
      PHeader.p_memsz = *YamlPhdr.MemSize;
    } else {
      PHeader.p_memsz = PHeader.p_filesz;
      for (auto SecName : YamlPhdr.Sections) {
        uint32_t Index = 0;
        SN2I.lookup(SecName.Section, Index);
        const auto &SHeader = SHeaders[Index];
        if (SHeader.sh_offset == PHeader.p_offset + PHeader.p_filesz)
          PHeader.p_memsz += SHeader.sh_size;
      }
    }

    // Set the alignment of the segment to be the same as the maximum alignment
    // of the sections with the same offset so that by default the segment
    // has a valid and sensible alignment.
    if (YamlPhdr.Align) {
      PHeader.p_align = *YamlPhdr.Align;
    } else {
      PHeader.p_align = 1;
      for (auto SecName : YamlPhdr.Sections) {
        uint32_t Index = 0;
        SN2I.lookup(SecName.Section, Index);
        const auto &SHeader = SHeaders[Index];
        if (SHeader.sh_offset == PHeader.p_offset)
          PHeader.p_align = std::max(PHeader.p_align, SHeader.sh_addralign);
      }
    }
  }
}

template <class ELFT>
void ELFState<ELFT>::addSymbols(ArrayRef<ELFYAML::Symbol> Symbols,
                                std::vector<Elf_Sym> &Syms,
                                const StringTableBuilder &Strtab) {
  for (const auto &Sym : Symbols) {
    Elf_Sym Symbol;
    zero(Symbol);
    if (!Sym.Name.empty())
      Symbol.st_name = Strtab.getOffset(Sym.Name);
    Symbol.setBindingAndType(Sym.Binding, Sym.Type);
    if (!Sym.Section.empty()) {
      unsigned Index;
      if (SN2I.lookup(Sym.Section, Index)) {
        WithColor::error() << "Unknown section referenced: '" << Sym.Section
                           << "' by YAML symbol " << Sym.Name << ".\n";
        exit(1);
      }
      Symbol.st_shndx = Index;
    } else if (Sym.Index) {
      Symbol.st_shndx = *Sym.Index;
    }
    // else Symbol.st_shndex == SHN_UNDEF (== 0), since it was zero'd earlier.
    Symbol.st_value = Sym.Value;
    Symbol.st_other = Sym.Other;
    Symbol.st_size = Sym.Size;
    Syms.push_back(Symbol);
  }
}

template <class ELFT>
bool ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::RawContentSection &Section,
    ContiguousBlobAccumulator &CBA) {
  assert(Section.Size >= Section.Content.binary_size() &&
         "Section size and section content are inconsistent");
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  Section.Content.writeAsBinary(OS);
  OS.write_zeros(Section.Size - Section.Content.binary_size());

  if (Section.EntSize)
    SHeader.sh_entsize = *Section.EntSize;
  else if (Section.Type == llvm::ELF::SHT_RELR)
    SHeader.sh_entsize = sizeof(Elf_Relr);
  else
    SHeader.sh_entsize = 0;
  SHeader.sh_size = Section.Size;
  SHeader.sh_info = Section.Info;
  return true;
}

static bool isMips64EL(const ELFYAML::Object &Doc) {
  return Doc.Header.Machine == ELFYAML::ELF_EM(llvm::ELF::EM_MIPS) &&
         Doc.Header.Class == ELFYAML::ELF_ELFCLASS(ELF::ELFCLASS64) &&
         Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB);
}

template <class ELFT>
bool
ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                    const ELFYAML::RelocationSection &Section,
                                    ContiguousBlobAccumulator &CBA) {
  assert((Section.Type == llvm::ELF::SHT_REL ||
          Section.Type == llvm::ELF::SHT_RELA) &&
         "Section type is not SHT_REL nor SHT_RELA");

  bool IsRela = Section.Type == llvm::ELF::SHT_RELA;
  SHeader.sh_entsize = IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);
  SHeader.sh_size = SHeader.sh_entsize * Section.Relocations.size();

  // For relocation section set link to .symtab by default.
  if (Section.Link.empty())
    SHeader.sh_link = getDotSymTabSecNo();

  unsigned Index = 0;
  if (!Section.RelocatableSec.empty() &&
      !convertSectionIndex(SN2I, Section.Name, Section.RelocatableSec, Index))
    return false;
  SHeader.sh_info = Index;

  auto &OS = CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  for (const auto &Rel : Section.Relocations) {
    unsigned SymIdx = 0;
    // If a relocation references a symbol, try to look one up in the symbol
    // table. If it is not there, treat the value as a symbol index.
    if (Rel.Symbol && SymN2I.lookup(*Rel.Symbol, SymIdx) &&
        !to_integer(*Rel.Symbol, SymIdx)) {
      WithColor::error() << "Unknown symbol referenced: '" << *Rel.Symbol
                         << "' at YAML section '" << Section.Name << "'.\n";
      return false;
    }

    if (IsRela) {
      Elf_Rela REntry;
      zero(REntry);
      REntry.r_offset = Rel.Offset;
      REntry.r_addend = Rel.Addend;
      REntry.setSymbolAndType(SymIdx, Rel.Type, isMips64EL(Doc));
      OS.write((const char *)&REntry, sizeof(REntry));
    } else {
      Elf_Rel REntry;
      zero(REntry);
      REntry.r_offset = Rel.Offset;
      REntry.setSymbolAndType(SymIdx, Rel.Type, isMips64EL(Doc));
      OS.write((const char *)&REntry, sizeof(REntry));
    }
  }
  return true;
}

template <class ELFT>
bool ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::Group &Section,
                                         ContiguousBlobAccumulator &CBA) {
  assert(Section.Type == llvm::ELF::SHT_GROUP &&
         "Section type is not SHT_GROUP");

  SHeader.sh_entsize = 4;
  SHeader.sh_size = SHeader.sh_entsize * Section.Members.size();

  unsigned SymIdx;
  if (SymN2I.lookup(Section.Signature, SymIdx) &&
      !to_integer(Section.Signature, SymIdx)) {
    WithColor::error() << "Unknown symbol referenced: '" << Section.Signature
                       << "' at YAML section '" << Section.Name << "'.\n";
    return false;
  }
  SHeader.sh_info = SymIdx;

  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  for (const ELFYAML::SectionOrType &Member : Section.Members) {
    unsigned int SectionIndex = 0;
    if (Member.sectionNameOrType == "GRP_COMDAT")
      SectionIndex = llvm::ELF::GRP_COMDAT;
    else if (!convertSectionIndex(SN2I, Section.Name, Member.sectionNameOrType,
                                  SectionIndex))
      return false;
    support::endian::write<uint32_t>(OS, SectionIndex, ELFT::TargetEndianness);
  }
  return true;
}

template <class ELFT>
bool ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::SymverSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  for (uint16_t Version : Section.Entries)
    support::endian::write<uint16_t>(OS, Version, ELFT::TargetEndianness);

  SHeader.sh_entsize = 2;
  SHeader.sh_size = Section.Entries.size() * SHeader.sh_entsize;
  return true;
}

template <class ELFT>
bool ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::VerdefSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  typedef typename ELFT::Verdef Elf_Verdef;
  typedef typename ELFT::Verdaux Elf_Verdaux;
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  uint64_t AuxCnt = 0;
  for (size_t I = 0; I < Section.Entries.size(); ++I) {
    const ELFYAML::VerdefEntry &E = Section.Entries[I];

    Elf_Verdef VerDef;
    VerDef.vd_version = E.Version;
    VerDef.vd_flags = E.Flags;
    VerDef.vd_ndx = E.VersionNdx;
    VerDef.vd_hash = E.Hash;
    VerDef.vd_aux = sizeof(Elf_Verdef);
    VerDef.vd_cnt = E.VerNames.size();
    if (I == Section.Entries.size() - 1)
      VerDef.vd_next = 0;
    else
      VerDef.vd_next =
          sizeof(Elf_Verdef) + E.VerNames.size() * sizeof(Elf_Verdaux);
    OS.write((const char *)&VerDef, sizeof(Elf_Verdef));

    for (size_t J = 0; J < E.VerNames.size(); ++J, ++AuxCnt) {
      Elf_Verdaux VernAux;
      VernAux.vda_name = DotDynstr.getOffset(E.VerNames[J]);
      if (J == E.VerNames.size() - 1)
        VernAux.vda_next = 0;
      else
        VernAux.vda_next = sizeof(Elf_Verdaux);
      OS.write((const char *)&VernAux, sizeof(Elf_Verdaux));
    }
  }

  SHeader.sh_size = Section.Entries.size() * sizeof(Elf_Verdef) +
                    AuxCnt * sizeof(Elf_Verdaux);
  SHeader.sh_info = Section.Info;

  return true;
}

template <class ELFT>
bool ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::VerneedSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  typedef typename ELFT::Verneed Elf_Verneed;
  typedef typename ELFT::Vernaux Elf_Vernaux;

  auto &OS = CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  uint64_t AuxCnt = 0;
  for (size_t I = 0; I < Section.VerneedV.size(); ++I) {
    const ELFYAML::VerneedEntry &VE = Section.VerneedV[I];

    Elf_Verneed VerNeed;
    VerNeed.vn_version = VE.Version;
    VerNeed.vn_file = DotDynstr.getOffset(VE.File);
    if (I == Section.VerneedV.size() - 1)
      VerNeed.vn_next = 0;
    else
      VerNeed.vn_next =
          sizeof(Elf_Verneed) + VE.AuxV.size() * sizeof(Elf_Vernaux);
    VerNeed.vn_cnt = VE.AuxV.size();
    VerNeed.vn_aux = sizeof(Elf_Verneed);
    OS.write((const char *)&VerNeed, sizeof(Elf_Verneed));

    for (size_t J = 0; J < VE.AuxV.size(); ++J, ++AuxCnt) {
      const ELFYAML::VernauxEntry &VAuxE = VE.AuxV[J];

      Elf_Vernaux VernAux;
      VernAux.vna_hash = VAuxE.Hash;
      VernAux.vna_flags = VAuxE.Flags;
      VernAux.vna_other = VAuxE.Other;
      VernAux.vna_name = DotDynstr.getOffset(VAuxE.Name);
      if (J == VE.AuxV.size() - 1)
        VernAux.vna_next = 0;
      else
        VernAux.vna_next = sizeof(Elf_Vernaux);
      OS.write((const char *)&VernAux, sizeof(Elf_Vernaux));
    }
  }

  SHeader.sh_size = Section.VerneedV.size() * sizeof(Elf_Verneed) +
                    AuxCnt * sizeof(Elf_Vernaux);
  SHeader.sh_info = Section.Info;

  return true;
}

template <class ELFT>
bool ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::MipsABIFlags &Section,
                                         ContiguousBlobAccumulator &CBA) {
  assert(Section.Type == llvm::ELF::SHT_MIPS_ABIFLAGS &&
         "Section type is not SHT_MIPS_ABIFLAGS");

  object::Elf_Mips_ABIFlags<ELFT> Flags;
  zero(Flags);
  SHeader.sh_entsize = sizeof(Flags);
  SHeader.sh_size = SHeader.sh_entsize;

  auto &OS = CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  Flags.version = Section.Version;
  Flags.isa_level = Section.ISALevel;
  Flags.isa_rev = Section.ISARevision;
  Flags.gpr_size = Section.GPRSize;
  Flags.cpr1_size = Section.CPR1Size;
  Flags.cpr2_size = Section.CPR2Size;
  Flags.fp_abi = Section.FpABI;
  Flags.isa_ext = Section.ISAExtension;
  Flags.ases = Section.ASEs;
  Flags.flags1 = Section.Flags1;
  Flags.flags2 = Section.Flags2;
  OS.write((const char *)&Flags, sizeof(Flags));

  return true;
}

template <class ELFT>
bool ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::DynamicSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  typedef typename ELFT::uint uintX_t;

  assert(Section.Type == llvm::ELF::SHT_DYNAMIC &&
         "Section type is not SHT_DYNAMIC");

  if (!Section.Entries.empty() && Section.Content) {
    WithColor::error()
        << "Cannot specify both raw content and explicit entries "
           "for dynamic section '"
        << Section.Name << "'.\n";
    return false;
  }

  if (Section.Content)
    SHeader.sh_size = Section.Content->binary_size();
  else
    SHeader.sh_size = 2 * sizeof(uintX_t) * Section.Entries.size();
  if (Section.EntSize)
    SHeader.sh_entsize = *Section.EntSize;
  else
    SHeader.sh_entsize = sizeof(Elf_Dyn);

  raw_ostream &OS = CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  for (const ELFYAML::DynamicEntry &DE : Section.Entries) {
    support::endian::write<uintX_t>(OS, DE.Tag, ELFT::TargetEndianness);
    support::endian::write<uintX_t>(OS, DE.Val, ELFT::TargetEndianness);
  }
  if (Section.Content)
    Section.Content->writeAsBinary(OS);

  return true;
}

template <class ELFT> bool ELFState<ELFT>::buildSectionIndex() {
  for (unsigned i = 0, e = Doc.Sections.size(); i != e; ++i) {
    StringRef Name = Doc.Sections[i]->Name;
    DotShStrtab.add(Name);
    // "+ 1" to take into account the SHT_NULL entry.
    if (SN2I.addName(Name, i + 1)) {
      WithColor::error() << "Repeated section name: '" << Name
                         << "' at YAML section number " << i << ".\n";
      return false;
    }
  }

  auto SecNo = 1 + Doc.Sections.size();
  // Add special sections after input sections, if necessary.
  for (StringRef Name : implicitSectionNames())
    if (!SN2I.addName(Name, SecNo)) {
      // Account for this section, since it wasn't in the Doc
      ++SecNo;
      DotShStrtab.add(Name);
    }

  DotShStrtab.finalize();
  return true;
}

template <class ELFT>
bool ELFState<ELFT>::buildSymbolIndex(ArrayRef<ELFYAML::Symbol> Symbols) {
  bool GlobalSymbolSeen = false;
  std::size_t I = 0;
  for (const auto &Sym : Symbols) {
    ++I;

    StringRef Name = Sym.Name;
    if (Sym.Binding.value == ELF::STB_LOCAL && GlobalSymbolSeen) {
      WithColor::error() << "Local symbol '" + Name +
                                "' after global in Symbols list.\n";
      return false;
    }
    if (Sym.Binding.value != ELF::STB_LOCAL)
      GlobalSymbolSeen = true;

    if (!Name.empty() && SymN2I.addName(Name, I)) {
      WithColor::error() << "Repeated symbol name: '" << Name << "'.\n";
      return false;
    }
  }
  return true;
}

template <class ELFT> void ELFState<ELFT>::finalizeStrings() {
  // Add the regular symbol names to .strtab section.
  for (const ELFYAML::Symbol &Sym : Doc.Symbols)
    DotStrtab.add(Sym.Name);
  DotStrtab.finalize();

  if (Doc.DynamicSymbols.empty())
    return;

  // Add the dynamic symbol names to .dynstr section.
  for (const ELFYAML::Symbol &Sym : Doc.DynamicSymbols)
    DotDynstr.add(Sym.Name);

  // SHT_GNU_verdef and SHT_GNU_verneed sections might also
  // add strings to .dynstr section.
  for (const std::unique_ptr<ELFYAML::Section> &Sec : Doc.Sections) {
    if (auto VerNeed = dyn_cast<ELFYAML::VerneedSection>(Sec.get())) {
      for (const ELFYAML::VerneedEntry &VE : VerNeed->VerneedV) {
        DotDynstr.add(VE.File);
        for (const ELFYAML::VernauxEntry &Aux : VE.AuxV)
          DotDynstr.add(Aux.Name);
      }
    } else if (auto VerDef = dyn_cast<ELFYAML::VerdefSection>(Sec.get())) {
      for (const ELFYAML::VerdefEntry &E : VerDef->Entries)
        for (StringRef Name : E.VerNames)
          DotDynstr.add(Name);
    }
  }

  DotDynstr.finalize();
}

template <class ELFT>
int ELFState<ELFT>::writeELF(raw_ostream &OS, const ELFYAML::Object &Doc) {
  ELFState<ELFT> State(Doc);

  // Finalize .strtab and .dynstr sections. We do that early because want to
  // finalize the string table builders before writing the content of the
  // sections that might want to use them.
  State.finalizeStrings();

  if (!State.buildSectionIndex())
    return 1;

  if (!State.buildSymbolIndex(Doc.Symbols))
    return 1;

  Elf_Ehdr Header;
  State.initELFHeader(Header);

  // TODO: Flesh out section header support.

  std::vector<Elf_Phdr> PHeaders;
  State.initProgramHeaders(PHeaders);

  // XXX: This offset is tightly coupled with the order that we write
  // things to `OS`.
  const size_t SectionContentBeginOffset = Header.e_ehsize +
                                           Header.e_phentsize * Header.e_phnum +
                                           Header.e_shentsize * Header.e_shnum;
  ContiguousBlobAccumulator CBA(SectionContentBeginOffset);

  std::vector<Elf_Shdr> SHeaders;
  if (!State.initSectionHeaders(SHeaders, CBA))
    return 1;

  // Populate SHeaders with implicit sections not present in the Doc
  for (StringRef Name : State.implicitSectionNames())
    if (State.SN2I.get(Name) >= SHeaders.size())
      SHeaders.push_back({});

  // Initialize the implicit sections
  State.initSymtabSectionHeader(SHeaders[State.SN2I.get(".symtab")],
                                SymtabType::Static, CBA);
  State.initStrtabSectionHeader(SHeaders[State.SN2I.get(".strtab")], ".strtab",
                                State.DotStrtab, CBA);
  State.initStrtabSectionHeader(SHeaders[State.SN2I.get(".shstrtab")],
                                ".shstrtab", State.DotShStrtab, CBA);
  if (!Doc.DynamicSymbols.empty()) {
    State.initSymtabSectionHeader(SHeaders[State.SN2I.get(".dynsym")],
                                  SymtabType::Dynamic, CBA);
    State.initStrtabSectionHeader(SHeaders[State.SN2I.get(".dynstr")],
                                  ".dynstr", State.DotDynstr, CBA);
  }

  // Now we can decide segment offsets
  State.setProgramHeaderLayout(PHeaders, SHeaders);

  OS.write((const char *)&Header, sizeof(Header));
  writeArrayData(OS, makeArrayRef(PHeaders));
  writeArrayData(OS, makeArrayRef(SHeaders));
  CBA.writeBlobToStream(OS);
  return 0;
}

template <class ELFT>
std::vector<StringRef> ELFState<ELFT>::implicitSectionNames() const {
  if (Doc.DynamicSymbols.empty())
    return {".symtab", ".strtab", ".shstrtab"};
  return {".symtab", ".strtab", ".shstrtab", ".dynsym", ".dynstr"};
}

int yaml2elf(llvm::ELFYAML::Object &Doc, raw_ostream &Out) {
  bool IsLE = Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB);
  bool Is64Bit = Doc.Header.Class == ELFYAML::ELF_ELFCLASS(ELF::ELFCLASS64);
  if (Is64Bit) {
    if (IsLE)
      return ELFState<object::ELF64LE>::writeELF(Out, Doc);
    return ELFState<object::ELF64BE>::writeELF(Out, Doc);
  }
  if (IsLE)
    return ELFState<object::ELF32LE>::writeELF(Out, Doc);
  return ELFState<object::ELF32BE>::writeELF(Out, Doc);
}
