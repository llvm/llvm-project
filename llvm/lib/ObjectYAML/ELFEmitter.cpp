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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/LEB128.h"
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

public:
  ContiguousBlobAccumulator(uint64_t InitialOffset_)
      : InitialOffset(InitialOffset_), Buf(), OS(Buf) {}

  template <class Integer>
  raw_ostream &getOSAndAlignedOffset(Integer &Offset, unsigned Align) {
    Offset = padToAlignment(Align);
    return OS;
  }

  /// \returns The new offset.
  uint64_t padToAlignment(unsigned Align) {
    if (Align == 0)
      Align = 1;
    uint64_t CurrentOffset = InitialOffset + OS.tell();
    uint64_t AlignedOffset = alignTo(CurrentOffset, Align);
    OS.write_zeros(AlignedOffset - CurrentOffset);
    return AlignedOffset; // == CurrentOffset;
  }

  void writeBlobToStream(raw_ostream &Out) { Out << OS.str(); }
};

// Used to keep track of section and symbol names, so that in the YAML file
// sections and symbols can be referenced by name instead of by index.
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

namespace {
struct Fragment {
  uint64_t Offset;
  uint64_t Size;
  uint32_t Type;
  uint64_t AddrAlign;
};
} // namespace

/// "Single point of truth" for the ELF file construction.
/// TODO: This class still has a ways to go before it is truly a "single
/// point of truth".
template <class ELFT> class ELFState {
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
  NameToIdxMap DynSymN2I;
  ELFYAML::Object &Doc;

  bool HasError = false;
  yaml::ErrorHandler ErrHandler;
  void reportError(const Twine &Msg);

  std::vector<Elf_Sym> toELFSymbols(ArrayRef<ELFYAML::Symbol> Symbols,
                                    const StringTableBuilder &Strtab);
  unsigned toSectionIndex(StringRef S, StringRef LocSec, StringRef LocSym = "");
  unsigned toSymbolIndex(StringRef S, StringRef LocSec, bool IsDynamic);

  void buildSectionIndex();
  void buildSymbolIndexes();
  void initProgramHeaders(std::vector<Elf_Phdr> &PHeaders);
  bool initImplicitHeader(ContiguousBlobAccumulator &CBA, Elf_Shdr &Header,
                          StringRef SecName, ELFYAML::Section *YAMLSec);
  void initSectionHeaders(std::vector<Elf_Shdr> &SHeaders,
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

  std::vector<Fragment>
  getPhdrFragments(const ELFYAML::ProgramHeader &Phdr,
                   ArrayRef<typename ELFT::Shdr> SHeaders);

  void finalizeStrings();
  void writeELFHeader(ContiguousBlobAccumulator &CBA, raw_ostream &OS);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::RawContentSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::RelocationSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader, const ELFYAML::Group &Group,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::SymtabShndxSection &Shndx,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::SymverSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::VerneedSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::VerdefSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::MipsABIFlags &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::DynamicSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::StackSizesSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::HashSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::AddrsigSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::NoteSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::GnuHashSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::LinkerOptionsSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::DependentLibrariesSection &Section,
                           ContiguousBlobAccumulator &CBA);

  void writeFill(ELFYAML::Fill &Fill, ContiguousBlobAccumulator &CBA);

  ELFState(ELFYAML::Object &D, yaml::ErrorHandler EH);

public:
  static bool writeELF(raw_ostream &OS, ELFYAML::Object &Doc,
                       yaml::ErrorHandler EH);
};
} // end anonymous namespace

template <class T> static size_t arrayDataSize(ArrayRef<T> A) {
  return A.size() * sizeof(T);
}

template <class T> static void writeArrayData(raw_ostream &OS, ArrayRef<T> A) {
  OS.write((const char *)A.data(), arrayDataSize(A));
}

template <class T> static void zero(T &Obj) { memset(&Obj, 0, sizeof(Obj)); }

template <class ELFT>
ELFState<ELFT>::ELFState(ELFYAML::Object &D, yaml::ErrorHandler EH)
    : Doc(D), ErrHandler(EH) {
  std::vector<ELFYAML::Section *> Sections = Doc.getSections();
  StringSet<> DocSections;
  for (const ELFYAML::Section *Sec : Sections)
    if (!Sec->Name.empty())
      DocSections.insert(Sec->Name);

  // Insert SHT_NULL section implicitly when it is not defined in YAML.
  if (Sections.empty() || Sections.front()->Type != ELF::SHT_NULL)
    Doc.Chunks.insert(
        Doc.Chunks.begin(),
        std::make_unique<ELFYAML::Section>(
            ELFYAML::Chunk::ChunkKind::RawContent, /*IsImplicit=*/true));

  std::vector<StringRef> ImplicitSections;
  if (Doc.Symbols)
    ImplicitSections.push_back(".symtab");
  ImplicitSections.insert(ImplicitSections.end(), {".strtab", ".shstrtab"});

  if (!Doc.DynamicSymbols.empty())
    ImplicitSections.insert(ImplicitSections.end(), {".dynsym", ".dynstr"});

  // Insert placeholders for implicit sections that are not
  // defined explicitly in YAML.
  for (StringRef SecName : ImplicitSections) {
    if (DocSections.count(SecName))
      continue;

    std::unique_ptr<ELFYAML::Chunk> Sec = std::make_unique<ELFYAML::Section>(
        ELFYAML::Chunk::ChunkKind::RawContent, true /*IsImplicit*/);
    Sec->Name = SecName;
    Doc.Chunks.push_back(std::move(Sec));
  }
}

template <class ELFT>
void ELFState<ELFT>::writeELFHeader(ContiguousBlobAccumulator &CBA, raw_ostream &OS) {
  using namespace llvm::ELF;

  Elf_Ehdr Header;
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
  Header.e_phoff = Doc.ProgramHeaders.size() ? sizeof(Header) : 0;
  Header.e_flags = Doc.Header.Flags;
  Header.e_ehsize = sizeof(Elf_Ehdr);
  Header.e_phentsize = Doc.ProgramHeaders.size() ? sizeof(Elf_Phdr) : 0;
  Header.e_phnum = Doc.ProgramHeaders.size();

  Header.e_shentsize =
      Doc.Header.SHEntSize ? (uint16_t)*Doc.Header.SHEntSize : sizeof(Elf_Shdr);
  // Immediately following the ELF header and program headers.
  // Align the start of the section header and write the ELF header.
  uint64_t SHOff;
  CBA.getOSAndAlignedOffset(SHOff, sizeof(typename ELFT::uint));
  Header.e_shoff =
      Doc.Header.SHOff ? typename ELFT::uint(*Doc.Header.SHOff) : SHOff;
  Header.e_shnum =
      Doc.Header.SHNum ? (uint16_t)*Doc.Header.SHNum : Doc.getSections().size();
  Header.e_shstrndx = Doc.Header.SHStrNdx ? (uint16_t)*Doc.Header.SHStrNdx
                                          : SN2I.get(".shstrtab");

  OS.write((const char *)&Header, sizeof(Header));
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

template <class ELFT>
unsigned ELFState<ELFT>::toSectionIndex(StringRef S, StringRef LocSec,
                                        StringRef LocSym) {
  unsigned Index;
  if (SN2I.lookup(S, Index) || to_integer(S, Index))
    return Index;

  assert(LocSec.empty() || LocSym.empty());
  if (!LocSym.empty())
    reportError("unknown section referenced: '" + S + "' by YAML symbol '" +
                LocSym + "'");
  else
    reportError("unknown section referenced: '" + S + "' by YAML section '" +
                LocSec + "'");
  return 0;
}

template <class ELFT>
unsigned ELFState<ELFT>::toSymbolIndex(StringRef S, StringRef LocSec,
                                       bool IsDynamic) {
  const NameToIdxMap &SymMap = IsDynamic ? DynSymN2I : SymN2I;
  unsigned Index;
  // Here we try to look up S in the symbol table. If it is not there,
  // treat its value as a symbol index.
  if (!SymMap.lookup(S, Index) && !to_integer(S, Index)) {
    reportError("unknown symbol referenced: '" + S + "' by YAML section '" +
                LocSec + "'");
    return 0;
  }
  return Index;
}

template <class ELFT>
bool ELFState<ELFT>::initImplicitHeader(ContiguousBlobAccumulator &CBA,
                                        Elf_Shdr &Header, StringRef SecName,
                                        ELFYAML::Section *YAMLSec) {
  // Check if the header was already initialized.
  if (Header.sh_offset)
    return false;

  if (SecName == ".symtab")
    initSymtabSectionHeader(Header, SymtabType::Static, CBA, YAMLSec);
  else if (SecName == ".strtab")
    initStrtabSectionHeader(Header, SecName, DotStrtab, CBA, YAMLSec);
  else if (SecName == ".shstrtab")
    initStrtabSectionHeader(Header, SecName, DotShStrtab, CBA, YAMLSec);
  else if (SecName == ".dynsym")
    initSymtabSectionHeader(Header, SymtabType::Dynamic, CBA, YAMLSec);
  else if (SecName == ".dynstr")
    initStrtabSectionHeader(Header, SecName, DotDynstr, CBA, YAMLSec);
  else
    return false;

  // Override the fields if requested.
  if (YAMLSec) {
    if (YAMLSec->ShName)
      Header.sh_name = *YAMLSec->ShName;
    if (YAMLSec->ShOffset)
      Header.sh_offset = *YAMLSec->ShOffset;
    if (YAMLSec->ShSize)
      Header.sh_size = *YAMLSec->ShSize;
  }

  return true;
}

StringRef llvm::ELFYAML::dropUniqueSuffix(StringRef S) {
  size_t SuffixPos = S.rfind(" [");
  if (SuffixPos == StringRef::npos)
    return S;
  return S.substr(0, SuffixPos);
}

template <class ELFT>
void ELFState<ELFT>::initSectionHeaders(std::vector<Elf_Shdr> &SHeaders,
                                        ContiguousBlobAccumulator &CBA) {
  // Ensure SHN_UNDEF entry is present. An all-zero section header is a
  // valid SHN_UNDEF entry since SHT_NULL == 0.
  SHeaders.resize(Doc.getSections().size());

  size_t SecNdx = -1;
  for (const std::unique_ptr<ELFYAML::Chunk> &D : Doc.Chunks) {
    if (auto S = dyn_cast<ELFYAML::Fill>(D.get())) {
      writeFill(*S, CBA);
      continue;
    }

    ++SecNdx;
    ELFYAML::Section *Sec = cast<ELFYAML::Section>(D.get());
    if (SecNdx == 0 && Sec->IsImplicit)
      continue;

    // We have a few sections like string or symbol tables that are usually
    // added implicitly to the end. However, if they are explicitly specified
    // in the YAML, we need to write them here. This ensures the file offset
    // remains correct.
    Elf_Shdr &SHeader = SHeaders[SecNdx];
    if (initImplicitHeader(CBA, SHeader, Sec->Name,
                           Sec->IsImplicit ? nullptr : Sec))
      continue;

    assert(Sec && "It can't be null unless it is an implicit section. But all "
                  "implicit sections should already have been handled above.");

    SHeader.sh_name =
        DotShStrtab.getOffset(ELFYAML::dropUniqueSuffix(Sec->Name));
    SHeader.sh_type = Sec->Type;
    if (Sec->Flags)
      SHeader.sh_flags = *Sec->Flags;
    SHeader.sh_addr = Sec->Address;
    SHeader.sh_addralign = Sec->AddressAlign;

    if (!Sec->Link.empty())
      SHeader.sh_link = toSectionIndex(Sec->Link, Sec->Name);

    if (SecNdx == 0) {
      if (auto RawSec = dyn_cast<ELFYAML::RawContentSection>(Sec)) {
        // We do not write any content for special SHN_UNDEF section.
        if (RawSec->Size)
          SHeader.sh_size = *RawSec->Size;
        if (RawSec->Info)
          SHeader.sh_info = *RawSec->Info;
      }
      if (Sec->EntSize)
        SHeader.sh_entsize = *Sec->EntSize;
    } else if (auto S = dyn_cast<ELFYAML::RawContentSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::SymtabShndxSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::RelocationSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::Group>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::MipsABIFlags>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::NoBitsSection>(Sec)) {
      SHeader.sh_entsize = 0;
      SHeader.sh_size = S->Size;
      // SHT_NOBITS section does not have content
      // so just to setup the section offset.
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
    } else if (auto S = dyn_cast<ELFYAML::DynamicSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::SymverSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::VerneedSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::VerdefSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::StackSizesSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::HashSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::AddrsigSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::LinkerOptionsSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::NoteSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::GnuHashSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::DependentLibrariesSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else {
      llvm_unreachable("Unknown section type");
    }

    // Override the fields if requested.
    if (Sec) {
      if (Sec->ShName)
        SHeader.sh_name = *Sec->ShName;
      if (Sec->ShOffset)
        SHeader.sh_offset = *Sec->ShOffset;
      if (Sec->ShSize)
        SHeader.sh_size = *Sec->ShSize;
    }
  }
}

static size_t findFirstNonGlobal(ArrayRef<ELFYAML::Symbol> Symbols) {
  for (size_t I = 0; I < Symbols.size(); ++I)
    if (Symbols[I].Binding.value != ELF::STB_LOCAL)
      return I;
  return Symbols.size();
}

static uint64_t writeContent(raw_ostream &OS,
                             const Optional<yaml::BinaryRef> &Content,
                             const Optional<llvm::yaml::Hex64> &Size) {
  size_t ContentSize = 0;
  if (Content) {
    Content->writeAsBinary(OS);
    ContentSize = Content->binary_size();
  }

  if (!Size)
    return ContentSize;

  OS.write_zeros(*Size - ContentSize);
  return *Size;
}

template <class ELFT>
std::vector<typename ELFT::Sym>
ELFState<ELFT>::toELFSymbols(ArrayRef<ELFYAML::Symbol> Symbols,
                             const StringTableBuilder &Strtab) {
  std::vector<Elf_Sym> Ret;
  Ret.resize(Symbols.size() + 1);

  size_t I = 0;
  for (const ELFYAML::Symbol &Sym : Symbols) {
    Elf_Sym &Symbol = Ret[++I];

    // If NameIndex, which contains the name offset, is explicitly specified, we
    // use it. This is useful for preparing broken objects. Otherwise, we add
    // the specified Name to the string table builder to get its offset.
    if (Sym.NameIndex)
      Symbol.st_name = *Sym.NameIndex;
    else if (!Sym.Name.empty())
      Symbol.st_name = Strtab.getOffset(ELFYAML::dropUniqueSuffix(Sym.Name));

    Symbol.setBindingAndType(Sym.Binding, Sym.Type);
    if (!Sym.Section.empty())
      Symbol.st_shndx = toSectionIndex(Sym.Section, "", Sym.Name);
    else if (Sym.Index)
      Symbol.st_shndx = *Sym.Index;

    Symbol.st_value = Sym.Value;
    Symbol.st_other = Sym.Other ? *Sym.Other : 0;
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
  ArrayRef<ELFYAML::Symbol> Symbols;
  if (IsStatic && Doc.Symbols)
    Symbols = *Doc.Symbols;
  else if (!IsStatic)
    Symbols = Doc.DynamicSymbols;

  ELFYAML::RawContentSection *RawSec =
      dyn_cast_or_null<ELFYAML::RawContentSection>(YAMLSec);
  if (RawSec && !Symbols.empty() && (RawSec->Content || RawSec->Size)) {
    if (RawSec->Content)
      reportError("cannot specify both `Content` and " +
                  (IsStatic ? Twine("`Symbols`") : Twine("`DynamicSymbols`")) +
                  " for symbol table section '" + RawSec->Name + "'");
    if (RawSec->Size)
      reportError("cannot specify both `Size` and " +
                  (IsStatic ? Twine("`Symbols`") : Twine("`DynamicSymbols`")) +
                  " for symbol table section '" + RawSec->Name + "'");
    return;
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
    SHeader.sh_link = toSectionIndex(RawSec->Link, RawSec->Name);
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
    SHeader.sh_size = writeContent(OS, RawSec->Content, RawSec->Size);
    return;
  }

  std::vector<Elf_Sym> Syms =
      toELFSymbols(Symbols, IsStatic ? DotStrtab : DotDynstr);
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
    SHeader.sh_size = writeContent(OS, RawSec->Content, RawSec->Size);
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

template <class ELFT> void ELFState<ELFT>::reportError(const Twine &Msg) {
  ErrHandler(Msg);
  HasError = true;
}

template <class ELFT>
std::vector<Fragment>
ELFState<ELFT>::getPhdrFragments(const ELFYAML::ProgramHeader &Phdr,
                                 ArrayRef<typename ELFT::Shdr> SHeaders) {
  DenseMap<StringRef, ELFYAML::Fill *> NameToFill;
  for (const std::unique_ptr<ELFYAML::Chunk> &D : Doc.Chunks)
    if (auto S = dyn_cast<ELFYAML::Fill>(D.get()))
      NameToFill[S->Name] = S;

  std::vector<Fragment> Ret;
  for (const ELFYAML::SectionName &SecName : Phdr.Sections) {
    unsigned Index;
    if (SN2I.lookup(SecName.Section, Index)) {
      const typename ELFT::Shdr &H = SHeaders[Index];
      Ret.push_back({H.sh_offset, H.sh_size, H.sh_type, H.sh_addralign});
      continue;
    }

    if (ELFYAML::Fill *Fill = NameToFill.lookup(SecName.Section)) {
      Ret.push_back({Fill->ShOffset, Fill->Size, llvm::ELF::SHT_PROGBITS,
                     /*ShAddrAlign=*/1});
      continue;
    }

    reportError("unknown section or fill referenced: '" + SecName.Section +
                "' by program header");
  }

  return Ret;
}

template <class ELFT>
void ELFState<ELFT>::setProgramHeaderLayout(std::vector<Elf_Phdr> &PHeaders,
                                            std::vector<Elf_Shdr> &SHeaders) {
  uint32_t PhdrIdx = 0;
  for (auto &YamlPhdr : Doc.ProgramHeaders) {
    Elf_Phdr &PHeader = PHeaders[PhdrIdx++];
    std::vector<Fragment> Fragments = getPhdrFragments(YamlPhdr, SHeaders);

    if (YamlPhdr.Offset) {
      PHeader.p_offset = *YamlPhdr.Offset;
    } else {
      if (YamlPhdr.Sections.size())
        PHeader.p_offset = UINT32_MAX;
      else
        PHeader.p_offset = 0;

      // Find the minimum offset for the program header.
      for (const Fragment &F : Fragments)
        PHeader.p_offset = std::min((uint64_t)PHeader.p_offset, F.Offset);
    }

    // Find the maximum offset of the end of a section in order to set p_filesz
    // and p_memsz. When setting p_filesz, trailing SHT_NOBITS sections are not
    // counted.
    uint64_t FileOffset = PHeader.p_offset, MemOffset = PHeader.p_offset;
    for (const Fragment &F : Fragments) {
      uint64_t End = F.Offset + F.Size;
      MemOffset = std::max(MemOffset, End);

      if (F.Type != llvm::ELF::SHT_NOBITS)
        FileOffset = std::max(FileOffset, End);
    }

    // Set the file size and the memory size if not set explicitly.
    PHeader.p_filesz = YamlPhdr.FileSize ? uint64_t(*YamlPhdr.FileSize)
                                         : FileOffset - PHeader.p_offset;
    PHeader.p_memsz = YamlPhdr.MemSize ? uint64_t(*YamlPhdr.MemSize)
                                       : MemOffset - PHeader.p_offset;

    if (YamlPhdr.Align) {
      PHeader.p_align = *YamlPhdr.Align;
    } else {
      // Set the alignment of the segment to be the maximum alignment of the
      // sections so that by default the segment has a valid and sensible
      // alignment.
      PHeader.p_align = 1;
      for (const Fragment &F : Fragments)
        PHeader.p_align = std::max((uint64_t)PHeader.p_align, F.AddrAlign);
    }
  }
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::RawContentSection &Section,
    ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  SHeader.sh_size = writeContent(OS, Section.Content, Section.Size);

  if (Section.EntSize)
    SHeader.sh_entsize = *Section.EntSize;
  else if (Section.Type == llvm::ELF::SHT_RELR)
    SHeader.sh_entsize = sizeof(Elf_Relr);
  else
    SHeader.sh_entsize = 0;

  if (Section.Info)
    SHeader.sh_info = *Section.Info;
}

static bool isMips64EL(const ELFYAML::Object &Doc) {
  return Doc.Header.Machine == ELFYAML::ELF_EM(llvm::ELF::EM_MIPS) &&
         Doc.Header.Class == ELFYAML::ELF_ELFCLASS(ELF::ELFCLASS64) &&
         Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB);
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::RelocationSection &Section,
    ContiguousBlobAccumulator &CBA) {
  assert((Section.Type == llvm::ELF::SHT_REL ||
          Section.Type == llvm::ELF::SHT_RELA) &&
         "Section type is not SHT_REL nor SHT_RELA");

  bool IsRela = Section.Type == llvm::ELF::SHT_RELA;
  SHeader.sh_entsize = IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);
  SHeader.sh_size = SHeader.sh_entsize * Section.Relocations.size();

  // For relocation section set link to .symtab by default.
  unsigned Link = 0;
  if (Section.Link.empty() && SN2I.lookup(".symtab", Link))
    SHeader.sh_link = Link;

  if (!Section.RelocatableSec.empty())
    SHeader.sh_info = toSectionIndex(Section.RelocatableSec, Section.Name);

  auto &OS = CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  for (const auto &Rel : Section.Relocations) {
    unsigned SymIdx = Rel.Symbol ? toSymbolIndex(*Rel.Symbol, Section.Name,
                                                 Section.Link == ".dynsym")
                                 : 0;
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
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::SymtabShndxSection &Shndx,
    ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  for (uint32_t E : Shndx.Entries)
    support::endian::write<uint32_t>(OS, E, ELFT::TargetEndianness);

  SHeader.sh_entsize = Shndx.EntSize ? (uint64_t)*Shndx.EntSize : 4;
  SHeader.sh_size = Shndx.Entries.size() * SHeader.sh_entsize;
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::Group &Section,
                                         ContiguousBlobAccumulator &CBA) {
  assert(Section.Type == llvm::ELF::SHT_GROUP &&
         "Section type is not SHT_GROUP");

  unsigned Link = 0;
  if (Section.Link.empty() && SN2I.lookup(".symtab", Link))
    SHeader.sh_link = Link;

  SHeader.sh_entsize = 4;
  SHeader.sh_size = SHeader.sh_entsize * Section.Members.size();

  if (Section.Signature)
    SHeader.sh_info =
        toSymbolIndex(*Section.Signature, Section.Name, /*IsDynamic=*/false);

  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  for (const ELFYAML::SectionOrType &Member : Section.Members) {
    unsigned int SectionIndex = 0;
    if (Member.sectionNameOrType == "GRP_COMDAT")
      SectionIndex = llvm::ELF::GRP_COMDAT;
    else
      SectionIndex = toSectionIndex(Member.sectionNameOrType, Section.Name);
    support::endian::write<uint32_t>(OS, SectionIndex, ELFT::TargetEndianness);
  }
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::SymverSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  for (uint16_t Version : Section.Entries)
    support::endian::write<uint16_t>(OS, Version, ELFT::TargetEndianness);

  SHeader.sh_entsize = Section.EntSize ? (uint64_t)*Section.EntSize : 2;
  SHeader.sh_size = Section.Entries.size() * SHeader.sh_entsize;
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::StackSizesSection &Section,
    ContiguousBlobAccumulator &CBA) {
  using uintX_t = typename ELFT::uint;
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  if (Section.Content || Section.Size) {
    SHeader.sh_size = writeContent(OS, Section.Content, Section.Size);
    return;
  }

  for (const ELFYAML::StackSizeEntry &E : *Section.Entries) {
    support::endian::write<uintX_t>(OS, E.Address, ELFT::TargetEndianness);
    SHeader.sh_size += sizeof(uintX_t) + encodeULEB128(E.Size, OS);
  }
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::LinkerOptionsSection &Section,
    ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  if (Section.Content) {
    SHeader.sh_size = writeContent(OS, Section.Content, None);
    return;
  }

  if (!Section.Options)
    return;

  for (const ELFYAML::LinkerOption &LO : *Section.Options) {
    OS.write(LO.Key.data(), LO.Key.size());
    OS.write('\0');
    OS.write(LO.Value.data(), LO.Value.size());
    OS.write('\0');
    SHeader.sh_size += (LO.Key.size() + LO.Value.size() + 2);
  }
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::DependentLibrariesSection &Section,
    ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  if (Section.Content) {
    SHeader.sh_size = writeContent(OS, Section.Content, None);
    return;
  }

  if (!Section.Libs)
    return;

  for (StringRef Lib : *Section.Libs) {
    OS.write(Lib.data(), Lib.size());
    OS.write('\0');
    SHeader.sh_size += Lib.size() + 1;
  }
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::HashSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  unsigned Link = 0;
  if (Section.Link.empty() && SN2I.lookup(".dynsym", Link))
    SHeader.sh_link = Link;

  if (Section.Content || Section.Size) {
    SHeader.sh_size = writeContent(OS, Section.Content, Section.Size);
    return;
  }

  support::endian::write<uint32_t>(OS, Section.Bucket->size(),
                                   ELFT::TargetEndianness);
  support::endian::write<uint32_t>(OS, Section.Chain->size(),
                                   ELFT::TargetEndianness);
  for (uint32_t Val : *Section.Bucket)
    support::endian::write<uint32_t>(OS, Val, ELFT::TargetEndianness);
  for (uint32_t Val : *Section.Chain)
    support::endian::write<uint32_t>(OS, Val, ELFT::TargetEndianness);

  SHeader.sh_size = (2 + Section.Bucket->size() + Section.Chain->size()) * 4;
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
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
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
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
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
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
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::DynamicSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  typedef typename ELFT::uint uintX_t;

  assert(Section.Type == llvm::ELF::SHT_DYNAMIC &&
         "Section type is not SHT_DYNAMIC");

  if (!Section.Entries.empty() && Section.Content)
    reportError("cannot specify both raw content and explicit entries "
                "for dynamic section '" +
                Section.Name + "'");

  if (Section.Content)
    SHeader.sh_size = Section.Content->binary_size();
  else
    SHeader.sh_size = 2 * sizeof(uintX_t) * Section.Entries.size();
  if (Section.EntSize)
    SHeader.sh_entsize = *Section.EntSize;
  else
    SHeader.sh_entsize = sizeof(Elf_Dyn);

  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  for (const ELFYAML::DynamicEntry &DE : Section.Entries) {
    support::endian::write<uintX_t>(OS, DE.Tag, ELFT::TargetEndianness);
    support::endian::write<uintX_t>(OS, DE.Val, ELFT::TargetEndianness);
  }
  if (Section.Content)
    Section.Content->writeAsBinary(OS);
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::AddrsigSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  unsigned Link = 0;
  if (Section.Link.empty() && SN2I.lookup(".symtab", Link))
    SHeader.sh_link = Link;

  if (Section.Content || Section.Size) {
    SHeader.sh_size = writeContent(OS, Section.Content, Section.Size);
    return;
  }

  for (const ELFYAML::AddrsigSymbol &Sym : *Section.Symbols) {
    uint64_t Val =
        Sym.Name ? toSymbolIndex(*Sym.Name, Section.Name, /*IsDynamic=*/false)
                 : (uint32_t)*Sym.Index;
    SHeader.sh_size += encodeULEB128(Val, OS);
  }
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::NoteSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  uint64_t Offset = OS.tell();

  if (Section.Content || Section.Size) {
    SHeader.sh_size = writeContent(OS, Section.Content, Section.Size);
    return;
  }

  for (const ELFYAML::NoteEntry &NE : *Section.Notes) {
    // Write name size.
    if (NE.Name.empty())
      support::endian::write<uint32_t>(OS, 0, ELFT::TargetEndianness);
    else
      support::endian::write<uint32_t>(OS, NE.Name.size() + 1,
                                       ELFT::TargetEndianness);

    // Write description size.
    if (NE.Desc.binary_size() == 0)
      support::endian::write<uint32_t>(OS, 0, ELFT::TargetEndianness);
    else
      support::endian::write<uint32_t>(OS, NE.Desc.binary_size(),
                                       ELFT::TargetEndianness);

    // Write type.
    support::endian::write<uint32_t>(OS, NE.Type, ELFT::TargetEndianness);

    // Write name, null terminator and padding.
    if (!NE.Name.empty()) {
      support::endian::write<uint8_t>(OS, arrayRefFromStringRef(NE.Name),
                                      ELFT::TargetEndianness);
      support::endian::write<uint8_t>(OS, 0, ELFT::TargetEndianness);
      CBA.padToAlignment(4);
    }

    // Write description and padding.
    if (NE.Desc.binary_size() != 0) {
      NE.Desc.writeAsBinary(OS);
      CBA.padToAlignment(4);
    }
  }

  SHeader.sh_size = OS.tell() - Offset;
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::GnuHashSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  unsigned Link = 0;
  if (Section.Link.empty() && SN2I.lookup(".dynsym", Link))
    SHeader.sh_link = Link;

  if (Section.Content) {
    SHeader.sh_size = writeContent(OS, Section.Content, None);
    return;
  }

  // We write the header first, starting with the hash buckets count. Normally
  // it is the number of entries in HashBuckets, but the "NBuckets" property can
  // be used to override this field, which is useful for producing broken
  // objects.
  if (Section.Header->NBuckets)
    support::endian::write<uint32_t>(OS, *Section.Header->NBuckets,
                                     ELFT::TargetEndianness);
  else
    support::endian::write<uint32_t>(OS, Section.HashBuckets->size(),
                                     ELFT::TargetEndianness);

  // Write the index of the first symbol in the dynamic symbol table accessible
  // via the hash table.
  support::endian::write<uint32_t>(OS, Section.Header->SymNdx,
                                   ELFT::TargetEndianness);

  // Write the number of words in the Bloom filter. As above, the "MaskWords"
  // property can be used to set this field to any value.
  if (Section.Header->MaskWords)
    support::endian::write<uint32_t>(OS, *Section.Header->MaskWords,
                                     ELFT::TargetEndianness);
  else
    support::endian::write<uint32_t>(OS, Section.BloomFilter->size(),
                                     ELFT::TargetEndianness);

  // Write the shift constant used by the Bloom filter.
  support::endian::write<uint32_t>(OS, Section.Header->Shift2,
                                   ELFT::TargetEndianness);

  // We've finished writing the header. Now write the Bloom filter.
  for (llvm::yaml::Hex64 Val : *Section.BloomFilter)
    support::endian::write<typename ELFT::uint>(OS, Val,
                                                ELFT::TargetEndianness);

  // Write an array of hash buckets.
  for (llvm::yaml::Hex32 Val : *Section.HashBuckets)
    support::endian::write<uint32_t>(OS, Val, ELFT::TargetEndianness);

  // Write an array of hash values.
  for (llvm::yaml::Hex32 Val : *Section.HashValues)
    support::endian::write<uint32_t>(OS, Val, ELFT::TargetEndianness);

  SHeader.sh_size = 16 /*Header size*/ +
                    Section.BloomFilter->size() * sizeof(typename ELFT::uint) +
                    Section.HashBuckets->size() * 4 +
                    Section.HashValues->size() * 4;
}

template <class ELFT>
void ELFState<ELFT>::writeFill(ELFYAML::Fill &Fill,
                               ContiguousBlobAccumulator &CBA) {
  raw_ostream &OS = CBA.getOSAndAlignedOffset(Fill.ShOffset, /*Align=*/1);

  size_t PatternSize = Fill.Pattern ? Fill.Pattern->binary_size() : 0;
  if (!PatternSize) {
    OS.write_zeros(Fill.Size);
    return;
  }

  // Fill the content with the specified pattern.
  uint64_t Written = 0;
  for (; Written + PatternSize <= Fill.Size; Written += PatternSize)
    Fill.Pattern->writeAsBinary(OS);
  Fill.Pattern->writeAsBinary(OS, Fill.Size - Written);
}

template <class ELFT> void ELFState<ELFT>::buildSectionIndex() {
  size_t SecNdx = -1;
  StringSet<> Seen;
  for (size_t I = 0; I < Doc.Chunks.size(); ++I) {
    const std::unique_ptr<ELFYAML::Chunk> &C = Doc.Chunks[I];
    bool IsSection = isa<ELFYAML::Section>(C.get());
    if (IsSection)
      ++SecNdx;

    if (C->Name.empty())
      continue;

    if (!Seen.insert(C->Name).second)
      reportError("repeated section/fill name: '" + C->Name +
                  "' at YAML section/fill number " + Twine(I));
    if (!IsSection || HasError)
      continue;

    if (!SN2I.addName(C->Name, SecNdx))
      llvm_unreachable("buildSectionIndex() failed");
    DotShStrtab.add(ELFYAML::dropUniqueSuffix(C->Name));
  }

  DotShStrtab.finalize();
}

template <class ELFT> void ELFState<ELFT>::buildSymbolIndexes() {
  auto Build = [this](ArrayRef<ELFYAML::Symbol> V, NameToIdxMap &Map) {
    for (size_t I = 0, S = V.size(); I < S; ++I) {
      const ELFYAML::Symbol &Sym = V[I];
      if (!Sym.Name.empty() && !Map.addName(Sym.Name, I + 1))
        reportError("repeated symbol name: '" + Sym.Name + "'");
    }
  };

  if (Doc.Symbols)
    Build(*Doc.Symbols, SymN2I);
  Build(Doc.DynamicSymbols, DynSymN2I);
}

template <class ELFT> void ELFState<ELFT>::finalizeStrings() {
  // Add the regular symbol names to .strtab section.
  if (Doc.Symbols)
    for (const ELFYAML::Symbol &Sym : *Doc.Symbols)
      DotStrtab.add(ELFYAML::dropUniqueSuffix(Sym.Name));
  DotStrtab.finalize();

  // Add the dynamic symbol names to .dynstr section.
  for (const ELFYAML::Symbol &Sym : Doc.DynamicSymbols)
    DotDynstr.add(ELFYAML::dropUniqueSuffix(Sym.Name));

  // SHT_GNU_verdef and SHT_GNU_verneed sections might also
  // add strings to .dynstr section.
  for (const ELFYAML::Chunk *Sec : Doc.getSections()) {
    if (auto VerNeed = dyn_cast<ELFYAML::VerneedSection>(Sec)) {
      for (const ELFYAML::VerneedEntry &VE : VerNeed->VerneedV) {
        DotDynstr.add(VE.File);
        for (const ELFYAML::VernauxEntry &Aux : VE.AuxV)
          DotDynstr.add(Aux.Name);
      }
    } else if (auto VerDef = dyn_cast<ELFYAML::VerdefSection>(Sec)) {
      for (const ELFYAML::VerdefEntry &E : VerDef->Entries)
        for (StringRef Name : E.VerNames)
          DotDynstr.add(Name);
    }
  }

  DotDynstr.finalize();
}

template <class ELFT>
bool ELFState<ELFT>::writeELF(raw_ostream &OS, ELFYAML::Object &Doc,
                              yaml::ErrorHandler EH) {
  ELFState<ELFT> State(Doc, EH);

  // Finalize .strtab and .dynstr sections. We do that early because want to
  // finalize the string table builders before writing the content of the
  // sections that might want to use them.
  State.finalizeStrings();

  State.buildSectionIndex();
  if (State.HasError)
    return false;

  State.buildSymbolIndexes();

  std::vector<Elf_Phdr> PHeaders;
  State.initProgramHeaders(PHeaders);

  // XXX: This offset is tightly coupled with the order that we write
  // things to `OS`.
  const size_t SectionContentBeginOffset =
      sizeof(Elf_Ehdr) + sizeof(Elf_Phdr) * Doc.ProgramHeaders.size();
  ContiguousBlobAccumulator CBA(SectionContentBeginOffset);

  std::vector<Elf_Shdr> SHeaders;
  State.initSectionHeaders(SHeaders, CBA);

  // Now we can decide segment offsets.
  State.setProgramHeaderLayout(PHeaders, SHeaders);

  if (State.HasError)
    return false;

  State.writeELFHeader(CBA, OS);
  writeArrayData(OS, makeArrayRef(PHeaders));
  CBA.writeBlobToStream(OS);
  writeArrayData(OS, makeArrayRef(SHeaders));
  return true;
}

namespace llvm {
namespace yaml {

bool yaml2elf(llvm::ELFYAML::Object &Doc, raw_ostream &Out, ErrorHandler EH) {
  bool IsLE = Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB);
  bool Is64Bit = Doc.Header.Class == ELFYAML::ELF_ELFCLASS(ELF::ELFCLASS64);
  if (Is64Bit) {
    if (IsLE)
      return ELFState<object::ELF64LE>::writeELF(Out, Doc, EH);
    return ELFState<object::ELF64BE>::writeELF(Out, Doc, EH);
  }
  if (IsLE)
    return ELFState<object::ELF32LE>::writeELF(Out, Doc, EH);
  return ELFState<object::ELF32BE>::writeELF(Out, Doc, EH);
}

} // namespace yaml
} // namespace llvm
