//===- yaml2elf - Convert YAML to a ELF object file -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief The ELF component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "yaml2obj.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/Support/MemoryBuffer.h"
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
    for (; CurrentOffset != AlignedOffset; ++CurrentOffset)
      OS.write('\0');
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
/// \brief "Single point of truth" for the ELF file construction.
/// TODO: This class still has a ways to go before it is truly a "single
/// point of truth".
template <class ELFT>
class ELFState {
  typedef typename object::ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  typedef typename object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename object::ELFFile<ELFT>::Elf_Rela Elf_Rela;

  /// \brief The future ".strtab" section.
  StringTableBuilder DotStrtab{StringTableBuilder::ELF};

  /// \brief The future ".shstrtab" section.
  StringTableBuilder DotShStrtab{StringTableBuilder::ELF};

  NameToIdxMap SN2I;
  NameToIdxMap SymN2I;
  const ELFYAML::Object &Doc;

  bool buildSectionIndex();
  bool buildSymbolIndex(std::size_t &StartIndex,
                        const std::vector<ELFYAML::Symbol> &Symbols);
  void initELFHeader(Elf_Ehdr &Header);
  bool initSectionHeaders(std::vector<Elf_Shdr> &SHeaders,
                          ContiguousBlobAccumulator &CBA);
  void initSymtabSectionHeader(Elf_Shdr &SHeader,
                               ContiguousBlobAccumulator &CBA);
  void initStrtabSectionHeader(Elf_Shdr &SHeader, StringRef Name,
                               StringTableBuilder &STB,
                               ContiguousBlobAccumulator &CBA);
  void addSymbols(const std::vector<ELFYAML::Symbol> &Symbols,
                  std::vector<Elf_Sym> &Syms, unsigned SymbolBinding);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::RawContentSection &Section,
                           ContiguousBlobAccumulator &CBA);
  bool writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::RelocationSection &Section,
                           ContiguousBlobAccumulator &CBA);
  bool writeSectionContent(Elf_Shdr &SHeader, const ELFYAML::Group &Group,
                           ContiguousBlobAccumulator &CBA);
  bool writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::MipsABIFlags &Section,
                           ContiguousBlobAccumulator &CBA);

  // - SHT_NULL entry (placed first, i.e. 0'th entry)
  // - symbol table (.symtab) (placed third to last)
  // - string table (.strtab) (placed second to last)
  // - section header string table (.shstrtab) (placed last)
  unsigned getDotSymTabSecNo() const { return Doc.Sections.size() + 1; }
  unsigned getDotStrTabSecNo() const { return Doc.Sections.size() + 2; }
  unsigned getDotShStrTabSecNo() const { return Doc.Sections.size() + 3; }
  unsigned getSectionCount() const { return Doc.Sections.size() + 4; }

  ELFState(const ELFYAML::Object &D) : Doc(D) {}

public:
  static int writeELF(raw_ostream &OS, const ELFYAML::Object &Doc);
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
  bool IsLittleEndian = ELFT::TargetEndianness == support::little;
  Header.e_ident[EI_DATA] = IsLittleEndian ? ELFDATA2LSB : ELFDATA2MSB;
  Header.e_ident[EI_VERSION] = EV_CURRENT;
  Header.e_ident[EI_OSABI] = Doc.Header.OSABI;
  Header.e_ident[EI_ABIVERSION] = 0;
  Header.e_type = Doc.Header.Type;
  Header.e_machine = Doc.Header.Machine;
  Header.e_version = EV_CURRENT;
  Header.e_entry = Doc.Header.Entry;
  Header.e_flags = Doc.Header.Flags;
  Header.e_ehsize = sizeof(Elf_Ehdr);
  Header.e_shentsize = sizeof(Elf_Shdr);
  // Immediately following the ELF header.
  Header.e_shoff = sizeof(Header);
  Header.e_shnum = getSectionCount();
  Header.e_shstrndx = getDotShStrTabSecNo();
}

template <class ELFT>
bool ELFState<ELFT>::initSectionHeaders(std::vector<Elf_Shdr> &SHeaders,
                                        ContiguousBlobAccumulator &CBA) {
  // Ensure SHN_UNDEF entry is present. An all-zero section header is a
  // valid SHN_UNDEF entry since SHT_NULL == 0.
  Elf_Shdr SHeader;
  zero(SHeader);
  SHeaders.push_back(SHeader);

  for (const auto &Sec : Doc.Sections)
    DotShStrtab.add(Sec->Name);
  DotShStrtab.finalize();

  for (const auto &Sec : Doc.Sections) {
    zero(SHeader);
    SHeader.sh_name = DotShStrtab.getOffset(Sec->Name);
    SHeader.sh_type = Sec->Type;
    SHeader.sh_flags = Sec->Flags;
    SHeader.sh_addr = Sec->Address;
    SHeader.sh_addralign = Sec->AddressAlign;

    if (!Sec->Link.empty()) {
      unsigned Index;
      if (SN2I.lookup(Sec->Link, Index)) {
        errs() << "error: Unknown section referenced: '" << Sec->Link
               << "' at YAML section '" << Sec->Name << "'.\n";
        return false;
      }
      SHeader.sh_link = Index;
    }

    if (auto S = dyn_cast<ELFYAML::RawContentSection>(Sec.get()))
      writeSectionContent(SHeader, *S, CBA);
    else if (auto S = dyn_cast<ELFYAML::RelocationSection>(Sec.get())) {
      if (S->Link.empty())
        // For relocation section set link to .symtab by default.
        SHeader.sh_link = getDotSymTabSecNo();

      unsigned Index;
      if (SN2I.lookup(S->Info, Index)) {
        if (S->Info.getAsInteger(0, Index)) {
          errs() << "error: Unknown section referenced: '" << S->Info
                 << "' at YAML section '" << S->Name << "'.\n";
          return false;
        }
      }
      SHeader.sh_info = Index;

      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else if (auto S = dyn_cast<ELFYAML::Group>(Sec.get())) {
      unsigned SymIdx;
      if (SymN2I.lookup(S->Info, SymIdx)) {
        errs() << "error: Unknown symbol referenced: '" << S->Info
               << "' at YAML section '" << S->Name << "'.\n";
        return false;
      }
      SHeader.sh_info = SymIdx;
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
    } else
      llvm_unreachable("Unknown section type");

    SHeaders.push_back(SHeader);
  }
  return true;
}

template <class ELFT>
void ELFState<ELFT>::initSymtabSectionHeader(Elf_Shdr &SHeader,
                                             ContiguousBlobAccumulator &CBA) {
  zero(SHeader);
  SHeader.sh_name = DotShStrtab.getOffset(".symtab");
  SHeader.sh_type = ELF::SHT_SYMTAB;
  SHeader.sh_link = getDotStrTabSecNo();
  // One greater than symbol table index of the last local symbol.
  SHeader.sh_info = Doc.Symbols.Local.size() + 1;
  SHeader.sh_entsize = sizeof(Elf_Sym);
  SHeader.sh_addralign = 8;

  std::vector<Elf_Sym> Syms;
  {
    // Ensure STN_UNDEF is present
    Elf_Sym Sym;
    zero(Sym);
    Syms.push_back(Sym);
  }

  // Add symbol names to .strtab.
  for (const auto &Sym : Doc.Symbols.Local)
    DotStrtab.add(Sym.Name);
  for (const auto &Sym : Doc.Symbols.Global)
    DotStrtab.add(Sym.Name);
  for (const auto &Sym : Doc.Symbols.Weak)
    DotStrtab.add(Sym.Name);
  DotStrtab.finalize();

  addSymbols(Doc.Symbols.Local, Syms, ELF::STB_LOCAL);
  addSymbols(Doc.Symbols.Global, Syms, ELF::STB_GLOBAL);
  addSymbols(Doc.Symbols.Weak, Syms, ELF::STB_WEAK);

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
}

template <class ELFT>
void ELFState<ELFT>::addSymbols(const std::vector<ELFYAML::Symbol> &Symbols,
                                std::vector<Elf_Sym> &Syms,
                                unsigned SymbolBinding) {
  for (const auto &Sym : Symbols) {
    Elf_Sym Symbol;
    zero(Symbol);
    if (!Sym.Name.empty())
      Symbol.st_name = DotStrtab.getOffset(Sym.Name);
    Symbol.setBindingAndType(SymbolBinding, Sym.Type);
    if (!Sym.Section.empty()) {
      unsigned Index;
      if (SN2I.lookup(Sym.Section, Index)) {
        errs() << "error: Unknown section referenced: '" << Sym.Section
               << "' by YAML symbol " << Sym.Name << ".\n";
        exit(1);
      }
      Symbol.st_shndx = Index;
    } // else Symbol.st_shndex == SHN_UNDEF (== 0), since it was zero'd earlier.
    Symbol.st_value = Sym.Value;
    Symbol.st_other = Sym.Other;
    Symbol.st_size = Sym.Size;
    Syms.push_back(Symbol);
  }
}

template <class ELFT>
void
ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                    const ELFYAML::RawContentSection &Section,
                                    ContiguousBlobAccumulator &CBA) {
  assert(Section.Size >= Section.Content.binary_size() &&
         "Section size and section content are inconsistent");
  raw_ostream &OS =
      CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);
  Section.Content.writeAsBinary(OS);
  for (auto i = Section.Content.binary_size(); i < Section.Size; ++i)
    OS.write(0);
  SHeader.sh_entsize = 0;
  SHeader.sh_size = Section.Size;
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

  auto &OS = CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  for (const auto &Rel : Section.Relocations) {
    unsigned SymIdx = 0;
    // Some special relocation, R_ARM_v4BX for instance, does not have
    // an external reference.  So it ignores the return value of lookup()
    // here.
    SymN2I.lookup(Rel.Symbol, SymIdx);

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
  typedef typename object::ELFFile<ELFT>::Elf_Word Elf_Word;
  assert(Section.Type == llvm::ELF::SHT_GROUP &&
         "Section type is not SHT_GROUP");

  SHeader.sh_entsize = sizeof(Elf_Word);
  SHeader.sh_size = SHeader.sh_entsize * Section.Members.size();

  auto &OS = CBA.getOSAndAlignedOffset(SHeader.sh_offset, SHeader.sh_addralign);

  for (auto member : Section.Members) {
    Elf_Word SIdx;
    unsigned int sectionIndex = 0;
    if (member.sectionNameOrType == "GRP_COMDAT")
      sectionIndex = llvm::ELF::GRP_COMDAT;
    else if (SN2I.lookup(member.sectionNameOrType, sectionIndex)) {
      errs() << "error: Unknown section referenced: '"
             << member.sectionNameOrType << "' at YAML section' "
             << Section.Name << "\n";
      return false;
    }
    SIdx = sectionIndex;
    OS.write((const char *)&SIdx, sizeof(SIdx));
  }
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

template <class ELFT> bool ELFState<ELFT>::buildSectionIndex() {
  SN2I.addName(".symtab", getDotSymTabSecNo());
  SN2I.addName(".strtab", getDotStrTabSecNo());
  SN2I.addName(".shstrtab", getDotShStrTabSecNo());

  for (unsigned i = 0, e = Doc.Sections.size(); i != e; ++i) {
    StringRef Name = Doc.Sections[i]->Name;
    if (Name.empty())
      continue;
    // "+ 1" to take into account the SHT_NULL entry.
    if (SN2I.addName(Name, i + 1)) {
      errs() << "error: Repeated section name: '" << Name
             << "' at YAML section number " << i << ".\n";
      return false;
    }
  }
  return true;
}

template <class ELFT>
bool
ELFState<ELFT>::buildSymbolIndex(std::size_t &StartIndex,
                                 const std::vector<ELFYAML::Symbol> &Symbols) {
  for (const auto &Sym : Symbols) {
    ++StartIndex;
    if (Sym.Name.empty())
      continue;
    if (SymN2I.addName(Sym.Name, StartIndex)) {
      errs() << "error: Repeated symbol name: '" << Sym.Name << "'.\n";
      return false;
    }
  }
  return true;
}

template <class ELFT>
int ELFState<ELFT>::writeELF(raw_ostream &OS, const ELFYAML::Object &Doc) {
  ELFState<ELFT> State(Doc);
  if (!State.buildSectionIndex())
    return 1;

  std::size_t StartSymIndex = 0;
  if (!State.buildSymbolIndex(StartSymIndex, Doc.Symbols.Local) ||
      !State.buildSymbolIndex(StartSymIndex, Doc.Symbols.Global) ||
      !State.buildSymbolIndex(StartSymIndex, Doc.Symbols.Weak))
    return 1;

  Elf_Ehdr Header;
  State.initELFHeader(Header);

  // TODO: Flesh out section header support.
  // TODO: Program headers.

  // XXX: This offset is tightly coupled with the order that we write
  // things to `OS`.
  const size_t SectionContentBeginOffset =
      Header.e_ehsize + Header.e_shentsize * Header.e_shnum;
  ContiguousBlobAccumulator CBA(SectionContentBeginOffset);

  // Doc might not contain .symtab, .strtab and .shstrtab sections,
  // but we will emit them, so make sure to add them to ShStrTabSHeader.
  State.DotShStrtab.add(".symtab");
  State.DotShStrtab.add(".strtab");
  State.DotShStrtab.add(".shstrtab");

  std::vector<Elf_Shdr> SHeaders;
  if(!State.initSectionHeaders(SHeaders, CBA))
    return 1;

  // .symtab section.
  Elf_Shdr SymtabSHeader;
  State.initSymtabSectionHeader(SymtabSHeader, CBA);
  SHeaders.push_back(SymtabSHeader);

  // .strtab string table header.
  Elf_Shdr DotStrTabSHeader;
  State.initStrtabSectionHeader(DotStrTabSHeader, ".strtab", State.DotStrtab,
                                CBA);
  SHeaders.push_back(DotStrTabSHeader);

  // .shstrtab string table header.
  Elf_Shdr ShStrTabSHeader;
  State.initStrtabSectionHeader(ShStrTabSHeader, ".shstrtab", State.DotShStrtab,
                                CBA);
  SHeaders.push_back(ShStrTabSHeader);

  OS.write((const char *)&Header, sizeof(Header));
  writeArrayData(OS, makeArrayRef(SHeaders));
  CBA.writeBlobToStream(OS);
  return 0;
}

static bool is64Bit(const ELFYAML::Object &Doc) {
  return Doc.Header.Class == ELFYAML::ELF_ELFCLASS(ELF::ELFCLASS64);
}

static bool isLittleEndian(const ELFYAML::Object &Doc) {
  return Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB);
}

int yaml2elf(llvm::ELFYAML::Object &Doc, raw_ostream &Out) {
  using object::ELFType;
  typedef ELFType<support::little, true> LE64;
  typedef ELFType<support::big, true> BE64;
  typedef ELFType<support::little, false> LE32;
  typedef ELFType<support::big, false> BE32;
  if (is64Bit(Doc)) {
    if (isLittleEndian(Doc))
      return ELFState<LE64>::writeELF(Out, Doc);
    else
      return ELFState<BE64>::writeELF(Out, Doc);
  } else {
    if (isLittleEndian(Doc))
      return ELFState<LE32>::writeELF(Out, Doc);
    else
      return ELFState<BE32>::writeELF(Out, Doc);
  }
}
