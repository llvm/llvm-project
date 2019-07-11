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
  StringMap<unsigned> Map;

public:
  /// \Returns false if name is already present in the map.
  bool addName(StringRef Name, unsigned Ndx) {
    return Map.insert({Name, Ndx}).second;
  }
  /// \Returns false if name is not present in the map.
  bool lookup(StringRef Name, unsigned &Idx) const {
    auto I = Map.find(Name);
    if (I == Map.end())
      return false;
    Idx = I->getValue();
    return true;
  }
  /// Asserts if name is not present in the map.
  unsigned get(StringRef Name) const {
    unsigned Idx;
    if (lookup(Name, Idx))
      return Idx;
    assert(false && "Expected section not found in index");
    return 0;
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
  bool initImplicitHeader(ELFState<ELFT> &State, ContiguousBlobAccumulator &CBA,
                          Elf_Shdr &Header, StringRef SecName,
                          ELFYAML::Section *YAMLSec);
  bool initSectionHeaders(ELFState<ELFT> &State,
                          std::vector<Elf_Shdr> &SHeaders,
                          ContiguousBlobAccumulator &CBA);
  void initSymtabSectionHeader(Elf_Shdr &SHeader, SymtabType STType,
                               ContiguousBlobAccumulator &CBA,
                               ELFYAML::Section *YAMLSec);
  void initStrtabSectionHeader(Elf_Shdr &SHeader, StringRef Name,
                               StringTableBuilder &STB,
                               ContiguousBlobAccumulator &CBA,
                               ELFYAML::Section *YAMLSec);
  void setProgramHeaderLayout(std::vector<Elf_Phdr> &PHeaders,
                              std::vector<Elf_Shdr> &SHeaders);
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

  Header.e_shentsize =
      Doc.Header.SHEntSize ? (uint16_t)*Doc.Header.SHEntSize : sizeof(Elf_Shdr);
  // Immediately following the ELF header and program headers.
  Header.e_shoff =
      Doc.Header.SHOffset
          ? (uint16_t)*Doc.Header.SHOffset
          : sizeof(Header) + sizeof(Elf_Phdr) * Doc.ProgramHeaders.size();
  Header.e_shnum =
      Doc.Header.SHNum ? (uint16_t)*Doc.Header.SHNum : SN2I.size() + 1;
  Header.e_shstrndx = Doc.Header.SHStrNdx ? (uint16_t)*Doc.Header.SHStrNdx
                                          : SN2I.get(".shstrtab");
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
  if (!SN2I.lookup(IndexSrc, IndexDest) && !to_integer(IndexSrc, IndexDest)) {
    WithColor::error() << "Unknown section referenced: '" << IndexSrc
                       << "' at YAML section '" << SecName << "'.\n";
    return false;
  }
  return true;
}

template <class ELFT>
bool ELFState<ELFT>::initImplicitHeader(ELFState<ELFT> &State,
                                        ContiguousBlobAccumulator &CBA,
                                        Elf_Shdr &Header, StringRef SecName,
                                        ELFYAML::Section *YAMLSec) {
  // Check if the header was already initialized.
  if (Header.sh_offset)
    return false;

  if (SecName == ".symtab")
    State.initSymtabSectionHeader(Header, SymtabType::Static, CBA, YAMLSec);
  else if (SecName == ".strtab")
    State.initStrtabSectionHeader(Header, SecName, State.DotStrtab, CBA,
                                  YAMLSec);
  else if (SecName == ".shstrtab")
    State.initStrtabSectionHeader(Header, SecName, State.DotShStrtab, CBA,
                                  YAMLSec);

  else if (SecName == ".dynsym")
    State.initSymtabSectionHeader(Header, SymtabType::Dynamic, CBA, YAMLSec);
  else if (SecName == ".dynstr")
    State.initStrtabSectionHeader(Header, SecName, State.DotDynstr, CBA,
                                  YAMLSec);
  else
    return false;

  // Override the sh_offset/sh_size fields if requested.
  if (YAMLSec) {
    if (YAMLSec->ShOffset)
      Header.sh_offset = *YAMLSec->ShOffset;
    if (YAMLSec->ShSize)
      Header.sh_size = *YAMLSec->ShSize;
  }

  return true;
}

static StringRef dropUniqueSuffix(StringRef S) {
  size_t SuffixPos = S.rfind(" [");
  if (SuffixPos == StringRef::npos)
    return S;
  return S.substr(0, SuffixPos);
}

template <class ELFT>
bool ELFState<ELFT>::initSectionHeaders(ELFState<ELFT> &State,
                                        std::vector<Elf_Shdr> &SHeaders,
                                        ContiguousBlobAccumulator &CBA) {
  // Build a list of sections we are going to add implicitly.
  std::vector<StringRef> ImplicitSections;
  for (StringRef Name : State.implicitSectionNames())
    if (State.SN2I.get(Name) > Doc.Sections.size())
      ImplicitSections.push_back(Name);

  // Ensure SHN_UNDEF entry is present. An all-zero section header is a
  // valid SHN_UNDEF entry since SHT_NULL == 0.
  SHeaders.resize(Doc.Sections.size() + ImplicitSections.size() + 1);
  zero(SHeaders[0]);

  for (size_t I = 1; I < Doc.Sections.size() + ImplicitSections.size() + 1; ++I) {
    Elf_Shdr &SHeader = SHeaders[I];
    zero(SHeader);
    ELFYAML::Section *Sec =
        I > Doc.Sections.size() ? nullptr : Doc.Sections[I - 1].get();

    // We have a few sections like string or symbol tables that are usually
    // added implicitly to the end. However, if they are explicitly specified
    // in the YAML, we need to write them here. This ensures the file offset
    // remains correct.
    StringRef SecName =
        Sec ? Sec->Name : ImplicitSections[I - Doc.Sections.size() - 1];
    if (initImplicitHeader(State, CBA, SHeader, SecName, Sec))
      continue;

    assert(Sec && "It can't be null unless it is an implicit section. But all "
                  "implicit sections should already have been handled above.");

    SHeader.sh_name = DotShStrtab.getOffset(dropUniqueSuffix(SecName));
    SHeader.sh_type = Sec->Type;
    if (Sec->Flags)
      SHeader.sh_flags = *Sec->Flags;
    SHeader.sh_addr = Sec->Address;
    SHeader.sh_addralign = Sec->AddressAlign;

    if (!Sec->Link.empty()) {
      unsigned Index;
      if (!convertSectionIndex(SN2I, Sec->Name, Sec->Link, Index))
        return false;
      SHeader.sh_link = Index;
    }

    if (auto S = dyn_cast<ELFYAML::RawContentSection>(Sec)) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::RelocationSection>(Sec)) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::Group>(Sec)) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::MipsABIFlags>(Sec)) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::NoBitsSection>(Sec)) {
      SHeader.sh_entsize = 0;
      SHeader.sh_size = S->Size;
      // SHT_NOBITS section does not have content
      // so just to setup the section offset.
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
    } else if (auto S = dyn_cast<ELFYAML::DynamicSection>(Sec)) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::SymverSection>(Sec)) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::VerneedSection>(Sec)) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::VerdefSection>(Sec)) {
      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else
      llvm_unreachable("Unknown section type");

    // Override the sh_offset/sh_size fields if requested.
    if (Sec) {
      if (Sec->ShOffset)
        SHeader.sh_offset = *Sec->ShOffset;
      if (Sec->ShSize)
        SHeader.sh_size = *Sec->ShSize;
    }
  }

  return true;
}

static size_t findFirstNonGlobal(ArrayRef<ELFYAML::Symbol> Symbols) {
  for (size_t I = 0; I < Symbols.size(); ++I)
    if (Symbols[I].Binding.value != ELF::STB_LOCAL)
      return I;
  return Symbols.size();
}

static uint64_t writeRawSectionData(raw_ostream &OS,
                                    const ELFYAML::RawContentSection &RawSec) {
  size_t ContentSize = 0;
  if (RawSec.Content) {
    RawSec.Content->writeAsBinary(OS);
    ContentSize = RawSec.Content->binary_size();
  }

  if (!RawSec.Size)
    return ContentSize;

  OS.write_zeros(*RawSec.Size - ContentSize);
  return *RawSec.Size;
}

template <class ELFT>
static std::vector<typename ELFT::Sym>
toELFSymbols(NameToIdxMap &SN2I, ArrayRef<ELFYAML::Symbol> Symbols,
             const StringTableBuilder &Strtab) {
  using Elf_Sym = typename ELFT::Sym;

  std::vector<Elf_Sym> Ret;
  Ret.resize(Symbols.size() + 1);

  size_t I = 0;
  for (const auto &Sym : Symbols) {
    Elf_Sym &Symbol = Ret[++I];

    // If NameIndex, which contains the name offset, is explicitly specified, we
    // use it. This is useful for preparing broken objects. Otherwise, we add
    // the specified Name to the string table builder to get its offset.
    if (Sym.NameIndex)
      Symbol.st_name = *Sym.NameIndex;
    else if (!Sym.Name.empty())
      Symbol.st_name = Strtab.getOffset(dropUniqueSuffix(Sym.Name));

    Symbol.setBindingAndType(Sym.Binding, Sym.Type);
    if (!Sym.Section.empty()) {
      unsigned Index;
      if (!SN2I.lookup(Sym.Section, Index)) {
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
  }

  return Ret;
}

template <class ELFT>
void ELFState<ELFT>::initSymtabSectionHeader(Elf_Shdr &SHeader,
                                             SymtabType STType,
                                             ContiguousBlobAccumulator &CBA,
                                             ELFYAML::Section *YAMLSec) {

  bool IsStatic = STType == SymtabType::Static;
  const auto &Symbols = IsStatic ? Doc.Symbols : Doc.DynamicSymbols;

  ELFYAML::RawContentSection *RawSec =
      dyn_cast_or_null<ELFYAML::RawContentSection>(YAMLSec);
  if (RawSec && !Symbols.empty() && (RawSec->Content || RawSec->Size)) {
    if (RawSec->Content)
      WithColor::error() << "Cannot specify both `Content` and " +
                                (IsStatic ? Twine("`Symbols`")
                                          : Twine("`DynamicSymbols`")) +
                                " for symbol table section '"
                         << RawSec->Name << "'.\n";
    if (RawSec->Size)
      WithColor::error() << "Cannot specify both `Size` and " +
                                (IsStatic ? Twine("`Symbols`")
                                          : Twine("`DynamicSymbols`")) +
                                " for symbol table section '"
                         << RawSec->Name << "'.\n";
    exit(1);
  }

  zero(SHeader);
  SHeader.sh_name = DotShStrtab.getOffset(IsStatic ? ".symtab" : ".dynsym");

  if (YAMLSec)
    SHeader.sh_type = YAMLSec->Type;
  else
    SHeader.sh_type = IsStatic ? ELF::SHT_SYMTAB : ELF::SHT_DYNSYM;

  if (RawSec && !RawSec->Link.empty()) {
    // If the Link field is explicitly defined in the document,
    // we should use it.
    unsigned Index;
    if (!convertSectionIndex(SN2I, RawSec->Name, RawSec->Link, Index))
      return;
    SHeader.sh_link = Index;
  } else {
    // When we describe the .dynsym section in the document explicitly, it is
    // allowed to omit the "DynamicSymbols" tag. In this case .dynstr is not
    // added implicitly and we should be able to leave the Link zeroed if
    // .dynstr is not defined.
    unsigned Link = 0;
    if (IsStatic)
      Link = SN2I.get(".strtab");
    else
      SN2I.lookup(".dynstr", Link);
    SHeader.sh_link = Link;
  }

  if (YAMLSec && YAMLSec->Flags)
    SHeader.sh_flags = *YAMLSec->Flags;
  else if (!IsStatic)
    SHeader.sh_flags = ELF::SHF_ALLOC;

  // If the symbol table section is explicitly described in the YAML
  // then we should set the fields requested.
  SHeader.sh_info = (RawSec && RawSec->Info) ? (unsigned)(*RawSec->Info)
                                             : findFirstNonGlobal(Symbols) + 1;
  SHeader.sh_entsize = (YAMLSec && YAMLSec->EntSize)
                           ? (uint64_t)(*YAMLSec->EntSize)
                           : sizeof(Elf_Sym);
  SHeader.sh_addralign = YAMLSec ? (uint64_t)YAMLSec->AddressAlign : 8;
  SHeader.sh_addr = YAMLSec ? (uint64_t)YAMLSec->Address : 0;

  auto &OS = CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  if (RawSec && (RawSec->Content || RawSec->Size)) {
    assert(Symbols.empty());
    SHeader.sh_size = writeRawSectionData(OS, *RawSec);
    return;
  }

  std::vector<Elf_Sym> Syms =
      toELFSymbols<ELFT>(SN2I, Symbols, IsStatic ? DotStrtab : DotDynstr);
  writeArrayData(OS, makeArrayRef(Syms));
  SHeader.sh_size = arrayDataSize(makeArrayRef(Syms));
}

template <class ELFT>
void ELFState<ELFT>::initStrtabSectionHeader(Elf_Shdr &SHeader, StringRef Name,
                                             StringTableBuilder &STB,
                                             ContiguousBlobAccumulator &CBA,
                                             ELFYAML::Section *YAMLSec) {
  zero(SHeader);
  SHeader.sh_name = DotShStrtab.getOffset(Name);
  SHeader.sh_type = YAMLSec ? YAMLSec->Type : ELF::SHT_STRTAB;
  SHeader.sh_addralign = YAMLSec ? (uint64_t)YAMLSec->AddressAlign : 1;

  ELFYAML::RawContentSection *RawSec =
      dyn_cast_or_null<ELFYAML::RawContentSection>(YAMLSec);

  auto &OS = CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  if (RawSec && (RawSec->Content || RawSec->Size)) {
    SHeader.sh_size = writeRawSectionData(OS, *RawSec);
  } else {
    STB.write(OS);
    SHeader.sh_size = STB.getSize();
  }

  if (YAMLSec && YAMLSec->EntSize)
    SHeader.sh_entsize = *YAMLSec->EntSize;

  if (RawSec && RawSec->Info)
    SHeader.sh_info = *RawSec->Info;

  if (YAMLSec && YAMLSec->Flags)
    SHeader.sh_flags = *YAMLSec->Flags;
  else if (Name == ".dynstr")
    SHeader.sh_flags = ELF::SHF_ALLOC;

  // If the section is explicitly described in the YAML
  // then we want to use its section address.
  if (YAMLSec)
    SHeader.sh_addr = YAMLSec->Address;
}

template <class ELFT>
void ELFState<ELFT>::setProgramHeaderLayout(std::vector<Elf_Phdr> &PHeaders,
                                            std::vector<Elf_Shdr> &SHeaders) {
  uint32_t PhdrIdx = 0;
  for (auto &YamlPhdr : Doc.ProgramHeaders) {
    Elf_Phdr &PHeader = PHeaders[PhdrIdx++];

    std::vector<Elf_Shdr *> Sections;
    for (const ELFYAML::SectionName &SecName : YamlPhdr.Sections) {
      unsigned Index;
      if (!SN2I.lookup(SecName.Section, Index)) {
        WithColor::error() << "Unknown section referenced: '" << SecName.Section
                           << "' by program header.\n";
        exit(1);
      }
      Sections.push_back(&SHeaders[Index]);
    }

    if (YamlPhdr.Offset) {
      PHeader.p_offset = *YamlPhdr.Offset;
    } else {
      if (YamlPhdr.Sections.size())
        PHeader.p_offset = UINT32_MAX;
      else
        PHeader.p_offset = 0;

      // Find the minimum offset for the program header.
      for (Elf_Shdr *SHeader : Sections)
        PHeader.p_offset = std::min(PHeader.p_offset, SHeader->sh_offset);
    }

    // Find the maximum offset of the end of a section in order to set p_filesz,
    // if not set explicitly.
    if (YamlPhdr.FileSize) {
      PHeader.p_filesz = *YamlPhdr.FileSize;
    } else {
      PHeader.p_filesz = 0;
      for (Elf_Shdr *SHeader : Sections) {
        uint64_t EndOfSection;
        if (SHeader->sh_type == llvm::ELF::SHT_NOBITS)
          EndOfSection = SHeader->sh_offset;
        else
          EndOfSection = SHeader->sh_offset + SHeader->sh_size;
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
      for (Elf_Shdr *SHeader : Sections)
        if (SHeader->sh_offset == PHeader.p_offset + PHeader.p_filesz)
          PHeader.p_memsz += SHeader->sh_size;
    }

    // Set the alignment of the segment to be the same as the maximum alignment
    // of the sections with the same offset so that by default the segment
    // has a valid and sensible alignment.
    if (YamlPhdr.Align) {
      PHeader.p_align = *YamlPhdr.Align;
    } else {
      PHeader.p_align = 1;
      for (Elf_Shdr *SHeader : Sections)
        if (SHeader->sh_offset == PHeader.p_offset)
          PHeader.p_align = std::max(PHeader.p_align, SHeader->sh_addralign);
    }
  }
}

template <class ELFT>
bool ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::RawContentSection &Section,
    ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  SHeader.sh_size = writeRawSectionData(OS, Section);

  if (Section.EntSize)
    SHeader.sh_entsize = *Section.EntSize;
  else if (Section.Type == llvm::ELF::SHT_RELR)
    SHeader.sh_entsize = sizeof(Elf_Relr);
  else
    SHeader.sh_entsize = 0;

  if (Section.Info)
    SHeader.sh_info = *Section.Info;

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
    SHeader.sh_link = SN2I.get(".symtab");

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
    if (Rel.Symbol && !SymN2I.lookup(*Rel.Symbol, SymIdx) &&
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
  if (!SymN2I.lookup(Section.Signature, SymIdx) &&
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
    DotShStrtab.add(dropUniqueSuffix(Name));
    // "+ 1" to take into account the SHT_NULL entry.
    if (!SN2I.addName(Name, i + 1)) {
      WithColor::error() << "Repeated section name: '" << Name
                         << "' at YAML section number " << i << ".\n";
      return false;
    }
  }

  auto SecNo = 1 + Doc.Sections.size();
  // Add special sections after input sections, if necessary.
  for (StringRef Name : implicitSectionNames())
    if (SN2I.addName(Name, SecNo)) {
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

    if (!Name.empty() && !SymN2I.addName(Name, I)) {
      WithColor::error() << "Repeated symbol name: '" << Name << "'.\n";
      return false;
    }
  }
  return true;
}

template <class ELFT> void ELFState<ELFT>::finalizeStrings() {
  // Add the regular symbol names to .strtab section.
  for (const ELFYAML::Symbol &Sym : Doc.Symbols)
    DotStrtab.add(dropUniqueSuffix(Sym.Name));
  DotStrtab.finalize();

  // Add the dynamic symbol names to .dynstr section.
  for (const ELFYAML::Symbol &Sym : Doc.DynamicSymbols)
    DotDynstr.add(dropUniqueSuffix(Sym.Name));

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
  if (!State.initSectionHeaders(State, SHeaders, CBA))
    return 1;

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
