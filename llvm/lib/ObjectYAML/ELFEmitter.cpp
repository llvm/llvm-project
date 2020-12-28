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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"
#include "llvm/ObjectYAML/DWARFYAML.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// This class is used to build up a contiguous binary blob while keeping
// track of an offset in the output (which notionally begins at
// `InitialOffset`).
// The blob might be limited to an arbitrary size. All attempts to write data
// are ignored and the error condition is remembered once the limit is reached.
// Such an approach allows us to simplify the code by delaying error reporting
// and doing it at a convenient time.
namespace {
class ContiguousBlobAccumulator {
  const uint64_t InitialOffset;
  const uint64_t MaxSize;

  SmallVector<char, 128> Buf;
  raw_svector_ostream OS;
  Error ReachedLimitErr = Error::success();

  bool checkLimit(uint64_t Size) {
    if (!ReachedLimitErr && getOffset() + Size <= MaxSize)
      return true;
    if (!ReachedLimitErr)
      ReachedLimitErr = createStringError(errc::invalid_argument,
                                          "reached the output size limit");
    return false;
  }

public:
  ContiguousBlobAccumulator(uint64_t BaseOffset, uint64_t SizeLimit)
      : InitialOffset(BaseOffset), MaxSize(SizeLimit), OS(Buf) {}

  uint64_t tell() const { return OS.tell(); }
  uint64_t getOffset() const { return InitialOffset + OS.tell(); }
  void writeBlobToStream(raw_ostream &Out) const { Out << OS.str(); }

  Error takeLimitError() {
    // Request to write 0 bytes to check we did not reach the limit.
    checkLimit(0);
    return std::move(ReachedLimitErr);
  }

  /// \returns The new offset.
  uint64_t padToAlignment(unsigned Align) {
    uint64_t CurrentOffset = getOffset();
    if (ReachedLimitErr)
      return CurrentOffset;

    uint64_t AlignedOffset = alignTo(CurrentOffset, Align == 0 ? 1 : Align);
    uint64_t PaddingSize = AlignedOffset - CurrentOffset;
    if (!checkLimit(PaddingSize))
      return CurrentOffset;

    writeZeros(PaddingSize);
    return AlignedOffset;
  }

  raw_ostream *getRawOS(uint64_t Size) {
    if (checkLimit(Size))
      return &OS;
    return nullptr;
  }

  void writeAsBinary(const yaml::BinaryRef &Bin, uint64_t N = UINT64_MAX) {
    if (!checkLimit(Bin.binary_size()))
      return;
    Bin.writeAsBinary(OS, N);
  }

  void writeZeros(uint64_t Num) {
    if (checkLimit(Num))
      OS.write_zeros(Num);
  }

  void write(const char *Ptr, size_t Size) {
    if (checkLimit(Size))
      OS.write(Ptr, Size);
  }

  void write(unsigned char C) {
    if (checkLimit(1))
      OS.write(C);
  }

  unsigned writeULEB128(uint64_t Val) {
    if (!checkLimit(sizeof(uint64_t)))
      return 0;
    return encodeULEB128(Val, OS);
  }

  template <typename T> void write(T Val, support::endianness E) {
    if (checkLimit(sizeof(T)))
      support::endian::write<T>(OS, Val, E);
  }
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
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)

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

  StringSet<> ExcludedSectionHeaders;

  uint64_t LocationCounter = 0;
  bool HasError = false;
  yaml::ErrorHandler ErrHandler;
  void reportError(const Twine &Msg);
  void reportError(Error Err);

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
  void initDWARFSectionHeader(Elf_Shdr &SHeader, StringRef Name,
                              ContiguousBlobAccumulator &CBA,
                              ELFYAML::Section *YAMLSec);
  void setProgramHeaderLayout(std::vector<Elf_Phdr> &PHeaders,
                              std::vector<Elf_Shdr> &SHeaders);

  std::vector<Fragment>
  getPhdrFragments(const ELFYAML::ProgramHeader &Phdr,
                   ArrayRef<typename ELFT::Shdr> SHeaders);

  void finalizeStrings();
  void writeELFHeader(raw_ostream &OS, Optional<uint64_t> SHOff);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::NoBitsSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::RawContentSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::RelocationSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::RelrSection &Section,
                           ContiguousBlobAccumulator &CBA);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::GroupSection &Group,
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
                           const ELFYAML::ARMIndexTableSection &Section,
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
                           const ELFYAML::BBAddrMapSection &Section,
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
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::CallGraphProfileSection &Section,
                           ContiguousBlobAccumulator &CBA);

  void writeFill(ELFYAML::Fill &Fill, ContiguousBlobAccumulator &CBA);

  ELFState(ELFYAML::Object &D, yaml::ErrorHandler EH);

  void assignSectionAddress(Elf_Shdr &SHeader, ELFYAML::Section *YAMLSec);

  DenseMap<StringRef, size_t> buildSectionHeaderReorderMap();

  BumpPtrAllocator StringAlloc;
  uint64_t alignToOffset(ContiguousBlobAccumulator &CBA, uint64_t Align,
                         llvm::Optional<llvm::yaml::Hex64> Offset);

  uint64_t getSectionNameOffset(StringRef Name);

public:
  static bool writeELF(raw_ostream &OS, ELFYAML::Object &Doc,
                       yaml::ErrorHandler EH, uint64_t MaxSize);
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
  // Insert SHT_NULL section implicitly when it is not defined in YAML.
  if (Sections.empty() || Sections.front()->Type != ELF::SHT_NULL)
    Doc.Chunks.insert(
        Doc.Chunks.begin(),
        std::make_unique<ELFYAML::Section>(
            ELFYAML::Chunk::ChunkKind::RawContent, /*IsImplicit=*/true));

  // We add a technical suffix for each unnamed section/fill. It does not affect
  // the output, but allows us to map them by name in the code and report better
  // error messages.
  StringSet<> DocSections;
  for (size_t I = 0; I < Doc.Chunks.size(); ++I) {
    const std::unique_ptr<ELFYAML::Chunk> &C = Doc.Chunks[I];
    if (C->Name.empty()) {
      std::string NewName = ELFYAML::appendUniqueSuffix(
          /*Name=*/"", "index " + Twine(I));
      C->Name = StringRef(NewName).copy(StringAlloc);
      assert(ELFYAML::dropUniqueSuffix(C->Name).empty());
    }

    if (!DocSections.insert(C->Name).second)
      reportError("repeated section/fill name: '" + C->Name +
                  "' at YAML section/fill number " + Twine(I));
  }

  std::vector<StringRef> ImplicitSections;
  if (Doc.DynamicSymbols)
    ImplicitSections.insert(ImplicitSections.end(), {".dynsym", ".dynstr"});
  if (Doc.Symbols)
    ImplicitSections.push_back(".symtab");
  if (Doc.DWARF)
    for (StringRef DebugSecName : Doc.DWARF->getNonEmptySectionNames()) {
      std::string SecName = ("." + DebugSecName).str();
      ImplicitSections.push_back(StringRef(SecName).copy(StringAlloc));
    }
  ImplicitSections.insert(ImplicitSections.end(), {".strtab"});
  if (!Doc.SectionHeaders || !Doc.SectionHeaders->NoHeaders.getValueOr(false))
    ImplicitSections.insert(ImplicitSections.end(), {".shstrtab"});

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
void ELFState<ELFT>::writeELFHeader(raw_ostream &OS, Optional<uint64_t> SHOff) {
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

  if (Doc.Header.Machine)
    Header.e_machine = *Doc.Header.Machine;
  else
    Header.e_machine = EM_NONE;

  Header.e_version = EV_CURRENT;
  Header.e_entry = Doc.Header.Entry;
  Header.e_flags = Doc.Header.Flags;
  Header.e_ehsize = sizeof(Elf_Ehdr);

  if (Doc.Header.EPhOff)
    Header.e_phoff = *Doc.Header.EPhOff;
  else if (!Doc.ProgramHeaders.empty())
    Header.e_phoff = sizeof(Header);
  else
    Header.e_phoff = 0;

  if (Doc.Header.EPhEntSize)
    Header.e_phentsize = *Doc.Header.EPhEntSize;
  else if (!Doc.ProgramHeaders.empty())
    Header.e_phentsize = sizeof(Elf_Phdr);
  else
    Header.e_phentsize = 0;

  if (Doc.Header.EPhNum)
    Header.e_phnum = *Doc.Header.EPhNum;
  else if (!Doc.ProgramHeaders.empty())
    Header.e_phnum = Doc.ProgramHeaders.size();
  else
    Header.e_phnum = 0;

  Header.e_shentsize = Doc.Header.EShEntSize ? (uint16_t)*Doc.Header.EShEntSize
                                             : sizeof(Elf_Shdr);

  if (Doc.Header.EShOff)
    Header.e_shoff = *Doc.Header.EShOff;
  else if (SHOff)
    Header.e_shoff = *SHOff;
  else
    Header.e_shoff = 0;

  if (Doc.Header.EShNum)
    Header.e_shnum = *Doc.Header.EShNum;
  else if (!Doc.SectionHeaders ||
           (Doc.SectionHeaders->NoHeaders && !*Doc.SectionHeaders->NoHeaders))
    Header.e_shnum = Doc.getSections().size();
  else if (!SHOff)
    Header.e_shnum = 0;
  else
    Header.e_shnum =
        (Doc.SectionHeaders->Sections ? Doc.SectionHeaders->Sections->size()
                                      : 0) +
        /*Null section*/ 1;

  if (Doc.Header.EShStrNdx)
    Header.e_shstrndx = *Doc.Header.EShStrNdx;
  else if (SHOff && !ExcludedSectionHeaders.count(".shstrtab"))
    Header.e_shstrndx = SN2I.get(".shstrtab");
  else
    Header.e_shstrndx = 0;

  OS.write((const char *)&Header, sizeof(Header));
}

template <class ELFT>
void ELFState<ELFT>::initProgramHeaders(std::vector<Elf_Phdr> &PHeaders) {
  DenseMap<StringRef, ELFYAML::Fill *> NameToFill;
  DenseMap<StringRef, size_t> NameToIndex;
  for (size_t I = 0, E = Doc.Chunks.size(); I != E; ++I) {
    if (auto S = dyn_cast<ELFYAML::Fill>(Doc.Chunks[I].get()))
      NameToFill[S->Name] = S;
    NameToIndex[Doc.Chunks[I]->Name] = I + 1;
  }

  std::vector<ELFYAML::Section *> Sections = Doc.getSections();
  for (size_t I = 0, E = Doc.ProgramHeaders.size(); I != E; ++I) {
    ELFYAML::ProgramHeader &YamlPhdr = Doc.ProgramHeaders[I];
    Elf_Phdr Phdr;
    zero(Phdr);
    Phdr.p_type = YamlPhdr.Type;
    Phdr.p_flags = YamlPhdr.Flags;
    Phdr.p_vaddr = YamlPhdr.VAddr;
    Phdr.p_paddr = YamlPhdr.PAddr;
    PHeaders.push_back(Phdr);

    if (!YamlPhdr.FirstSec && !YamlPhdr.LastSec)
      continue;

    // Get the index of the section, or 0 in the case when the section doesn't exist.
    size_t First = NameToIndex[*YamlPhdr.FirstSec];
    if (!First)
      reportError("unknown section or fill referenced: '" + *YamlPhdr.FirstSec +
                  "' by the 'FirstSec' key of the program header with index " +
                  Twine(I));
    size_t Last = NameToIndex[*YamlPhdr.LastSec];
    if (!Last)
      reportError("unknown section or fill referenced: '" + *YamlPhdr.LastSec +
                  "' by the 'LastSec' key of the program header with index " +
                  Twine(I));
    if (!First || !Last)
      continue;

    if (First > Last)
      reportError("program header with index " + Twine(I) +
                  ": the section index of " + *YamlPhdr.FirstSec +
                  " is greater than the index of " + *YamlPhdr.LastSec);

    for (size_t I = First; I <= Last; ++I)
      YamlPhdr.Chunks.push_back(Doc.Chunks[I - 1].get());
  }
}

template <class ELFT>
unsigned ELFState<ELFT>::toSectionIndex(StringRef S, StringRef LocSec,
                                        StringRef LocSym) {
  assert(LocSec.empty() || LocSym.empty());

  unsigned Index;
  if (!SN2I.lookup(S, Index) && !to_integer(S, Index)) {
    if (!LocSym.empty())
      reportError("unknown section referenced: '" + S + "' by YAML symbol '" +
                  LocSym + "'");
    else
      reportError("unknown section referenced: '" + S + "' by YAML section '" +
                  LocSec + "'");
    return 0;
  }

  if (!Doc.SectionHeaders || (Doc.SectionHeaders->NoHeaders &&
                              !Doc.SectionHeaders->NoHeaders.getValue()))
    return Index;

  assert(!Doc.SectionHeaders->NoHeaders.getValueOr(false) ||
         !Doc.SectionHeaders->Sections);
  size_t FirstExcluded =
      Doc.SectionHeaders->Sections ? Doc.SectionHeaders->Sections->size() : 0;
  if (Index >= FirstExcluded) {
    if (LocSym.empty())
      reportError("unable to link '" + LocSec + "' to excluded section '" + S +
                  "'");
    else
      reportError("excluded section referenced: '" + S + "'  by symbol '" +
                  LocSym + "'");
  }
  return Index;
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
static void overrideFields(ELFYAML::Section *From, typename ELFT::Shdr &To) {
  if (!From)
    return;
  if (From->ShAddrAlign)
    To.sh_addralign = *From->ShAddrAlign;
  if (From->ShFlags)
    To.sh_flags = *From->ShFlags;
  if (From->ShName)
    To.sh_name = *From->ShName;
  if (From->ShOffset)
    To.sh_offset = *From->ShOffset;
  if (From->ShSize)
    To.sh_size = *From->ShSize;
  if (From->ShType)
    To.sh_type = *From->ShType;
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
  else if (SecName.startswith(".debug_")) {
    // If a ".debug_*" section's type is a preserved one, e.g., SHT_DYNAMIC, we
    // will not treat it as a debug section.
    if (YAMLSec && !isa<ELFYAML::RawContentSection>(YAMLSec))
      return false;
    initDWARFSectionHeader(Header, SecName, CBA, YAMLSec);
  } else
    return false;

  LocationCounter += Header.sh_size;

  // Override section fields if requested.
  overrideFields<ELFT>(YAMLSec, Header);
  return true;
}

constexpr char SuffixStart = '(';
constexpr char SuffixEnd = ')';

std::string llvm::ELFYAML::appendUniqueSuffix(StringRef Name,
                                              const Twine &Msg) {
  // Do not add a space when a Name is empty.
  std::string Ret = Name.empty() ? "" : Name.str() + ' ';
  return Ret + (Twine(SuffixStart) + Msg + Twine(SuffixEnd)).str();
}

StringRef llvm::ELFYAML::dropUniqueSuffix(StringRef S) {
  if (S.empty() || S.back() != SuffixEnd)
    return S;

  // A special case for empty names. See appendUniqueSuffix() above.
  size_t SuffixPos = S.rfind(SuffixStart);
  if (SuffixPos == 0)
    return "";

  if (SuffixPos == StringRef::npos || S[SuffixPos - 1] != ' ')
    return S;
  return S.substr(0, SuffixPos - 1);
}

template <class ELFT>
uint64_t ELFState<ELFT>::getSectionNameOffset(StringRef Name) {
  // If a section is excluded from section headers, we do not save its name in
  // the string table.
  if (ExcludedSectionHeaders.count(Name))
    return 0;
  return DotShStrtab.getOffset(Name);
}

static uint64_t writeContent(ContiguousBlobAccumulator &CBA,
                             const Optional<yaml::BinaryRef> &Content,
                             const Optional<llvm::yaml::Hex64> &Size) {
  size_t ContentSize = 0;
  if (Content) {
    CBA.writeAsBinary(*Content);
    ContentSize = Content->binary_size();
  }

  if (!Size)
    return ContentSize;

  CBA.writeZeros(*Size - ContentSize);
  return *Size;
}

template <class ELFT>
void ELFState<ELFT>::initSectionHeaders(std::vector<Elf_Shdr> &SHeaders,
                                        ContiguousBlobAccumulator &CBA) {
  // Ensure SHN_UNDEF entry is present. An all-zero section header is a
  // valid SHN_UNDEF entry since SHT_NULL == 0.
  SHeaders.resize(Doc.getSections().size());

  for (const std::unique_ptr<ELFYAML::Chunk> &D : Doc.Chunks) {
    if (ELFYAML::Fill *S = dyn_cast<ELFYAML::Fill>(D.get())) {
      S->Offset = alignToOffset(CBA, /*Align=*/1, S->Offset);
      writeFill(*S, CBA);
      LocationCounter += S->Size;
      continue;
    }

    ELFYAML::Section *Sec = cast<ELFYAML::Section>(D.get());
    bool IsFirstUndefSection = D == Doc.Chunks.front();
    if (IsFirstUndefSection && Sec->IsImplicit)
      continue;

    // We have a few sections like string or symbol tables that are usually
    // added implicitly to the end. However, if they are explicitly specified
    // in the YAML, we need to write them here. This ensures the file offset
    // remains correct.
    Elf_Shdr &SHeader = SHeaders[SN2I.get(Sec->Name)];
    if (initImplicitHeader(CBA, SHeader, Sec->Name,
                           Sec->IsImplicit ? nullptr : Sec))
      continue;

    assert(Sec && "It can't be null unless it is an implicit section. But all "
                  "implicit sections should already have been handled above.");

    SHeader.sh_name =
        getSectionNameOffset(ELFYAML::dropUniqueSuffix(Sec->Name));
    SHeader.sh_type = Sec->Type;
    if (Sec->Flags)
      SHeader.sh_flags = *Sec->Flags;
    SHeader.sh_addralign = Sec->AddressAlign;

    // Set the offset for all sections, except the SHN_UNDEF section with index
    // 0 when not explicitly requested.
    if (!IsFirstUndefSection || Sec->Offset)
      SHeader.sh_offset = alignToOffset(CBA, SHeader.sh_addralign, Sec->Offset);

    assignSectionAddress(SHeader, Sec);

    if (Sec->Link)
      SHeader.sh_link = toSectionIndex(*Sec->Link, Sec->Name);

    if (IsFirstUndefSection) {
      if (auto RawSec = dyn_cast<ELFYAML::RawContentSection>(Sec)) {
        // We do not write any content for special SHN_UNDEF section.
        if (RawSec->Size)
          SHeader.sh_size = *RawSec->Size;
        if (RawSec->Info)
          SHeader.sh_info = *RawSec->Info;
      }
      if (Sec->EntSize)
        SHeader.sh_entsize = *Sec->EntSize;

      LocationCounter += SHeader.sh_size;
      overrideFields<ELFT>(Sec, SHeader);
      continue;
    }

    if (!isa<ELFYAML::NoBitsSection>(Sec) && (Sec->Content || Sec->Size))
      SHeader.sh_size = writeContent(CBA, Sec->Content, Sec->Size);

    if (auto S = dyn_cast<ELFYAML::RawContentSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::SymtabShndxSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::RelocationSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::RelrSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::GroupSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::ARMIndexTableSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::MipsABIFlags>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::NoBitsSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
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
    } else if (auto S = dyn_cast<ELFYAML::CallGraphProfileSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else if (auto S = dyn_cast<ELFYAML::BBAddrMapSection>(Sec)) {
      writeSectionContent(SHeader, *S, CBA);
    } else {
      llvm_unreachable("Unknown section type");
    }

    LocationCounter += SHeader.sh_size;

    // Override section fields if requested.
    overrideFields<ELFT>(Sec, SHeader);
  }
}

template <class ELFT>
void ELFState<ELFT>::assignSectionAddress(Elf_Shdr &SHeader,
                                          ELFYAML::Section *YAMLSec) {
  if (YAMLSec && YAMLSec->Address) {
    SHeader.sh_addr = *YAMLSec->Address;
    LocationCounter = *YAMLSec->Address;
    return;
  }

  // sh_addr represents the address in the memory image of a process. Sections
  // in a relocatable object file or non-allocatable sections do not need
  // sh_addr assignment.
  if (Doc.Header.Type.value == ELF::ET_REL ||
      !(SHeader.sh_flags & ELF::SHF_ALLOC))
    return;

  LocationCounter =
      alignTo(LocationCounter, SHeader.sh_addralign ? SHeader.sh_addralign : 1);
  SHeader.sh_addr = LocationCounter;
}

static size_t findFirstNonGlobal(ArrayRef<ELFYAML::Symbol> Symbols) {
  for (size_t I = 0; I < Symbols.size(); ++I)
    if (Symbols[I].Binding.value != ELF::STB_LOCAL)
      return I;
  return Symbols.size();
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
    if (Sym.StName)
      Symbol.st_name = *Sym.StName;
    else if (!Sym.Name.empty())
      Symbol.st_name = Strtab.getOffset(ELFYAML::dropUniqueSuffix(Sym.Name));

    Symbol.setBindingAndType(Sym.Binding, Sym.Type);
    if (Sym.Section)
      Symbol.st_shndx = toSectionIndex(*Sym.Section, "", Sym.Name);
    else if (Sym.Index)
      Symbol.st_shndx = *Sym.Index;

    Symbol.st_value = Sym.Value.getValueOr(yaml::Hex64(0));
    Symbol.st_other = Sym.Other ? *Sym.Other : 0;
    Symbol.st_size = Sym.Size.getValueOr(yaml::Hex64(0));
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
  else if (!IsStatic && Doc.DynamicSymbols)
    Symbols = *Doc.DynamicSymbols;

  ELFYAML::RawContentSection *RawSec =
      dyn_cast_or_null<ELFYAML::RawContentSection>(YAMLSec);
  if (RawSec && (RawSec->Content || RawSec->Size)) {
    bool HasSymbolsDescription =
        (IsStatic && Doc.Symbols) || (!IsStatic && Doc.DynamicSymbols);
    if (HasSymbolsDescription) {
      StringRef Property = (IsStatic ? "`Symbols`" : "`DynamicSymbols`");
      if (RawSec->Content)
        reportError("cannot specify both `Content` and " + Property +
                    " for symbol table section '" + RawSec->Name + "'");
      if (RawSec->Size)
        reportError("cannot specify both `Size` and " + Property +
                    " for symbol table section '" + RawSec->Name + "'");
      return;
    }
  }

  zero(SHeader);
  SHeader.sh_name = getSectionNameOffset(IsStatic ? ".symtab" : ".dynsym");

  if (YAMLSec)
    SHeader.sh_type = YAMLSec->Type;
  else
    SHeader.sh_type = IsStatic ? ELF::SHT_SYMTAB : ELF::SHT_DYNSYM;

  if (RawSec && RawSec->Link) {
    // If the Link field is explicitly defined in the document,
    // we should use it.
    SHeader.sh_link = toSectionIndex(*RawSec->Link, RawSec->Name);
  } else {
    // When we describe the .dynsym section in the document explicitly, it is
    // allowed to omit the "DynamicSymbols" tag. In this case .dynstr is not
    // added implicitly and we should be able to leave the Link zeroed if
    // .dynstr is not defined.
    unsigned Link = 0;
    if (IsStatic) {
      if (!ExcludedSectionHeaders.count(".strtab"))
        Link = SN2I.get(".strtab");
    } else {
      if (!ExcludedSectionHeaders.count(".dynstr"))
        SN2I.lookup(".dynstr", Link);
    }
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

  assignSectionAddress(SHeader, YAMLSec);

  SHeader.sh_offset =
      alignToOffset(CBA, SHeader.sh_addralign, RawSec ? RawSec->Offset : None);

  if (RawSec && (RawSec->Content || RawSec->Size)) {
    assert(Symbols.empty());
    SHeader.sh_size = writeContent(CBA, RawSec->Content, RawSec->Size);
    return;
  }

  std::vector<Elf_Sym> Syms =
      toELFSymbols(Symbols, IsStatic ? DotStrtab : DotDynstr);
  SHeader.sh_size = Syms.size() * sizeof(Elf_Sym);
  CBA.write((const char *)Syms.data(), SHeader.sh_size);
}

template <class ELFT>
void ELFState<ELFT>::initStrtabSectionHeader(Elf_Shdr &SHeader, StringRef Name,
                                             StringTableBuilder &STB,
                                             ContiguousBlobAccumulator &CBA,
                                             ELFYAML::Section *YAMLSec) {
  zero(SHeader);
  SHeader.sh_name = getSectionNameOffset(Name);
  SHeader.sh_type = YAMLSec ? YAMLSec->Type : ELF::SHT_STRTAB;
  SHeader.sh_addralign = YAMLSec ? (uint64_t)YAMLSec->AddressAlign : 1;

  ELFYAML::RawContentSection *RawSec =
      dyn_cast_or_null<ELFYAML::RawContentSection>(YAMLSec);

  SHeader.sh_offset = alignToOffset(CBA, SHeader.sh_addralign,
                                    YAMLSec ? YAMLSec->Offset : None);

  if (RawSec && (RawSec->Content || RawSec->Size)) {
    SHeader.sh_size = writeContent(CBA, RawSec->Content, RawSec->Size);
  } else {
    if (raw_ostream *OS = CBA.getRawOS(STB.getSize()))
      STB.write(*OS);
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

  assignSectionAddress(SHeader, YAMLSec);
}

static bool shouldEmitDWARF(DWARFYAML::Data &DWARF, StringRef Name) {
  SetVector<StringRef> DebugSecNames = DWARF.getNonEmptySectionNames();
  return Name.consume_front(".") && DebugSecNames.count(Name);
}

template <class ELFT>
Expected<uint64_t> emitDWARF(typename ELFT::Shdr &SHeader, StringRef Name,
                             const DWARFYAML::Data &DWARF,
                             ContiguousBlobAccumulator &CBA) {
  // We are unable to predict the size of debug data, so we request to write 0
  // bytes. This should always return us an output stream unless CBA is already
  // in an error state.
  raw_ostream *OS = CBA.getRawOS(0);
  if (!OS)
    return 0;

  uint64_t BeginOffset = CBA.tell();

  auto EmitFunc = DWARFYAML::getDWARFEmitterByName(Name.substr(1));
  if (Error Err = EmitFunc(*OS, DWARF))
    return std::move(Err);

  return CBA.tell() - BeginOffset;
}

template <class ELFT>
void ELFState<ELFT>::initDWARFSectionHeader(Elf_Shdr &SHeader, StringRef Name,
                                            ContiguousBlobAccumulator &CBA,
                                            ELFYAML::Section *YAMLSec) {
  zero(SHeader);
  SHeader.sh_name = getSectionNameOffset(ELFYAML::dropUniqueSuffix(Name));
  SHeader.sh_type = YAMLSec ? YAMLSec->Type : ELF::SHT_PROGBITS;
  SHeader.sh_addralign = YAMLSec ? (uint64_t)YAMLSec->AddressAlign : 1;
  SHeader.sh_offset = alignToOffset(CBA, SHeader.sh_addralign,
                                    YAMLSec ? YAMLSec->Offset : None);

  ELFYAML::RawContentSection *RawSec =
      dyn_cast_or_null<ELFYAML::RawContentSection>(YAMLSec);
  if (Doc.DWARF && shouldEmitDWARF(*Doc.DWARF, Name)) {
    if (RawSec && (RawSec->Content || RawSec->Size))
      reportError("cannot specify section '" + Name +
                  "' contents in the 'DWARF' entry and the 'Content' "
                  "or 'Size' in the 'Sections' entry at the same time");
    else {
      if (Expected<uint64_t> ShSizeOrErr =
              emitDWARF<ELFT>(SHeader, Name, *Doc.DWARF, CBA))
        SHeader.sh_size = *ShSizeOrErr;
      else
        reportError(ShSizeOrErr.takeError());
    }
  } else if (RawSec)
    SHeader.sh_size = writeContent(CBA, RawSec->Content, RawSec->Size);
  else
    llvm_unreachable("debug sections can only be initialized via the 'DWARF' "
                     "entry or a RawContentSection");

  if (YAMLSec && YAMLSec->EntSize)
    SHeader.sh_entsize = *YAMLSec->EntSize;
  else if (Name == ".debug_str")
    SHeader.sh_entsize = 1;

  if (RawSec && RawSec->Info)
    SHeader.sh_info = *RawSec->Info;

  if (YAMLSec && YAMLSec->Flags)
    SHeader.sh_flags = *YAMLSec->Flags;
  else if (Name == ".debug_str")
    SHeader.sh_flags = ELF::SHF_MERGE | ELF::SHF_STRINGS;

  if (YAMLSec && YAMLSec->Link)
    SHeader.sh_link = toSectionIndex(*YAMLSec->Link, Name);

  assignSectionAddress(SHeader, YAMLSec);
}

template <class ELFT> void ELFState<ELFT>::reportError(const Twine &Msg) {
  ErrHandler(Msg);
  HasError = true;
}

template <class ELFT> void ELFState<ELFT>::reportError(Error Err) {
  handleAllErrors(std::move(Err), [&](const ErrorInfoBase &Err) {
    reportError(Err.message());
  });
}

template <class ELFT>
std::vector<Fragment>
ELFState<ELFT>::getPhdrFragments(const ELFYAML::ProgramHeader &Phdr,
                                 ArrayRef<Elf_Shdr> SHeaders) {
  std::vector<Fragment> Ret;
  for (const ELFYAML::Chunk *C : Phdr.Chunks) {
    if (const ELFYAML::Fill *F = dyn_cast<ELFYAML::Fill>(C)) {
      Ret.push_back({*F->Offset, F->Size, llvm::ELF::SHT_PROGBITS,
                     /*ShAddrAlign=*/1});
      continue;
    }

    const ELFYAML::Section *S = cast<ELFYAML::Section>(C);
    const Elf_Shdr &H = SHeaders[SN2I.get(S->Name)];
    Ret.push_back({H.sh_offset, H.sh_size, H.sh_type, H.sh_addralign});
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
    if (!llvm::is_sorted(Fragments, [](const Fragment &A, const Fragment &B) {
          return A.Offset < B.Offset;
        }))
      reportError("sections in the program header with index " +
                  Twine(PhdrIdx) + " are not sorted by their file offset");

    if (YamlPhdr.Offset) {
      if (!Fragments.empty() && *YamlPhdr.Offset > Fragments.front().Offset)
        reportError("'Offset' for segment with index " + Twine(PhdrIdx) +
                    " must be less than or equal to the minimum file offset of "
                    "all included sections (0x" +
                    Twine::utohexstr(Fragments.front().Offset) + ")");
      PHeader.p_offset = *YamlPhdr.Offset;
    } else if (!Fragments.empty()) {
      PHeader.p_offset = Fragments.front().Offset;
    }

    // Set the file size if not set explicitly.
    if (YamlPhdr.FileSize) {
      PHeader.p_filesz = *YamlPhdr.FileSize;
    } else if (!Fragments.empty()) {
      uint64_t FileSize = Fragments.back().Offset - PHeader.p_offset;
      // SHT_NOBITS sections occupy no physical space in a file, we should not
      // take their sizes into account when calculating the file size of a
      // segment.
      if (Fragments.back().Type != llvm::ELF::SHT_NOBITS)
        FileSize += Fragments.back().Size;
      PHeader.p_filesz = FileSize;
    }

    // Find the maximum offset of the end of a section in order to set p_memsz.
    uint64_t MemOffset = PHeader.p_offset;
    for (const Fragment &F : Fragments)
      MemOffset = std::max(MemOffset, F.Offset + F.Size);
    // Set the memory size if not set explicitly.
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

bool llvm::ELFYAML::shouldAllocateFileSpace(
    ArrayRef<ELFYAML::ProgramHeader> Phdrs, const ELFYAML::NoBitsSection &S) {
  for (const ELFYAML::ProgramHeader &PH : Phdrs) {
    auto It = llvm::find_if(
        PH.Chunks, [&](ELFYAML::Chunk *C) { return C->Name == S.Name; });
    if (std::any_of(It, PH.Chunks.end(), [](ELFYAML::Chunk *C) {
          return (isa<ELFYAML::Fill>(C) ||
                  cast<ELFYAML::Section>(C)->Type != ELF::SHT_NOBITS);
        }))
      return true;
  }
  return false;
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::NoBitsSection &S,
                                         ContiguousBlobAccumulator &CBA) {
  if (!S.Size)
    return;

  SHeader.sh_size = *S.Size;

  // When a nobits section is followed by a non-nobits section or fill
  // in the same segment, we allocate the file space for it. This behavior
  // matches linkers.
  if (shouldAllocateFileSpace(Doc.ProgramHeaders, S))
    CBA.writeZeros(*S.Size);
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::RawContentSection &Section,
    ContiguousBlobAccumulator &CBA) {
  if (Section.EntSize)
    SHeader.sh_entsize = *Section.EntSize;

  if (Section.Info)
    SHeader.sh_info = *Section.Info;
}

static bool isMips64EL(const ELFYAML::Object &Obj) {
  return Obj.getMachine() == llvm::ELF::EM_MIPS &&
         Obj.Header.Class == ELFYAML::ELF_ELFCLASS(ELF::ELFCLASS64) &&
         Obj.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB);
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::RelocationSection &Section,
    ContiguousBlobAccumulator &CBA) {
  assert((Section.Type == llvm::ELF::SHT_REL ||
          Section.Type == llvm::ELF::SHT_RELA) &&
         "Section type is not SHT_REL nor SHT_RELA");

  bool IsRela = Section.Type == llvm::ELF::SHT_RELA;
  if (Section.EntSize)
    SHeader.sh_entsize = *Section.EntSize;
  else
    SHeader.sh_entsize = IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);

  // For relocation section set link to .symtab by default.
  unsigned Link = 0;
  if (!Section.Link && !ExcludedSectionHeaders.count(".symtab") &&
      SN2I.lookup(".symtab", Link))
    SHeader.sh_link = Link;

  if (!Section.RelocatableSec.empty())
    SHeader.sh_info = toSectionIndex(Section.RelocatableSec, Section.Name);

  if (!Section.Relocations)
    return;

  for (const ELFYAML::Relocation &Rel : *Section.Relocations) {
    const bool IsDynamic = Section.Link && (*Section.Link == ".dynsym");
    unsigned SymIdx =
        Rel.Symbol ? toSymbolIndex(*Rel.Symbol, Section.Name, IsDynamic) : 0;
    if (IsRela) {
      Elf_Rela REntry;
      zero(REntry);
      REntry.r_offset = Rel.Offset;
      REntry.r_addend = Rel.Addend;
      REntry.setSymbolAndType(SymIdx, Rel.Type, isMips64EL(Doc));
      CBA.write((const char *)&REntry, sizeof(REntry));
    } else {
      Elf_Rel REntry;
      zero(REntry);
      REntry.r_offset = Rel.Offset;
      REntry.setSymbolAndType(SymIdx, Rel.Type, isMips64EL(Doc));
      CBA.write((const char *)&REntry, sizeof(REntry));
    }
  }

  SHeader.sh_size = (IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel)) *
                    Section.Relocations->size();
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::RelrSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  SHeader.sh_entsize =
      Section.EntSize ? uint64_t(*Section.EntSize) : sizeof(Elf_Relr);

  if (!Section.Entries)
    return;

  for (llvm::yaml::Hex64 E : *Section.Entries) {
    if (!ELFT::Is64Bits && E > UINT32_MAX)
      reportError(Section.Name + ": the value is too large for 32-bits: 0x" +
                  Twine::utohexstr(E));
    CBA.write<uintX_t>(E, ELFT::TargetEndianness);
  }

  SHeader.sh_size = sizeof(uintX_t) * Section.Entries->size();
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::SymtabShndxSection &Shndx,
    ContiguousBlobAccumulator &CBA) {
  SHeader.sh_entsize = Shndx.EntSize ? (uint64_t)*Shndx.EntSize : 4;

  if (Shndx.Content || Shndx.Size) {
    SHeader.sh_size = writeContent(CBA, Shndx.Content, Shndx.Size);
    return;
  }

  if (!Shndx.Entries)
    return;

  for (uint32_t E : *Shndx.Entries)
    CBA.write<uint32_t>(E, ELFT::TargetEndianness);
  SHeader.sh_size = Shndx.Entries->size() * SHeader.sh_entsize;
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::GroupSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  assert(Section.Type == llvm::ELF::SHT_GROUP &&
         "Section type is not SHT_GROUP");

  unsigned Link = 0;
  if (!Section.Link && !ExcludedSectionHeaders.count(".symtab") &&
      SN2I.lookup(".symtab", Link))
    SHeader.sh_link = Link;

  if (Section.EntSize)
    SHeader.sh_entsize = *Section.EntSize;
  else
    SHeader.sh_entsize = sizeof(typename ELFT::Word);

  if (Section.Signature)
    SHeader.sh_info =
        toSymbolIndex(*Section.Signature, Section.Name, /*IsDynamic=*/false);

  if (!Section.Members)
    return;

  for (const ELFYAML::SectionOrType &Member : *Section.Members) {
    unsigned int SectionIndex = 0;
    if (Member.sectionNameOrType == "GRP_COMDAT")
      SectionIndex = llvm::ELF::GRP_COMDAT;
    else
      SectionIndex = toSectionIndex(Member.sectionNameOrType, Section.Name);
    CBA.write<uint32_t>(SectionIndex, ELFT::TargetEndianness);
  }
  SHeader.sh_size = SHeader.sh_entsize * Section.Members->size();
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::SymverSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  SHeader.sh_entsize = Section.EntSize ? (uint64_t)*Section.EntSize : 2;

  if (!Section.Entries)
    return;

  for (uint16_t Version : *Section.Entries)
    CBA.write<uint16_t>(Version, ELFT::TargetEndianness);
  SHeader.sh_size = Section.Entries->size() * SHeader.sh_entsize;
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::StackSizesSection &Section,
    ContiguousBlobAccumulator &CBA) {
  if (!Section.Entries)
    return;

  if (!Section.Entries)
    return;

  for (const ELFYAML::StackSizeEntry &E : *Section.Entries) {
    CBA.write<uintX_t>(E.Address, ELFT::TargetEndianness);
    SHeader.sh_size += sizeof(uintX_t) + CBA.writeULEB128(E.Size);
  }
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::BBAddrMapSection &Section,
    ContiguousBlobAccumulator &CBA) {
  if (!Section.Entries)
    return;

  for (const ELFYAML::BBAddrMapEntry &E : *Section.Entries) {
    // Write the address of the function.
    CBA.write<uintX_t>(E.Address, ELFT::TargetEndianness);
    // Write number of BBEntries (number of basic blocks in the function).
    size_t NumBlocks = E.BBEntries ? E.BBEntries->size() : 0;
    SHeader.sh_size += sizeof(uintX_t) + CBA.writeULEB128(NumBlocks);
    if (!NumBlocks)
      continue;
    // Write all BBEntries.
    for (const ELFYAML::BBAddrMapEntry::BBEntry &BBE : *E.BBEntries)
      SHeader.sh_size += CBA.writeULEB128(BBE.AddressOffset) +
                         CBA.writeULEB128(BBE.Size) +
                         CBA.writeULEB128(BBE.Metadata);
  }
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::LinkerOptionsSection &Section,
    ContiguousBlobAccumulator &CBA) {
  if (!Section.Options)
    return;

  for (const ELFYAML::LinkerOption &LO : *Section.Options) {
    CBA.write(LO.Key.data(), LO.Key.size());
    CBA.write('\0');
    CBA.write(LO.Value.data(), LO.Value.size());
    CBA.write('\0');
    SHeader.sh_size += (LO.Key.size() + LO.Value.size() + 2);
  }
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::DependentLibrariesSection &Section,
    ContiguousBlobAccumulator &CBA) {
  if (!Section.Libs)
    return;

  for (StringRef Lib : *Section.Libs) {
    CBA.write(Lib.data(), Lib.size());
    CBA.write('\0');
    SHeader.sh_size += Lib.size() + 1;
  }
}

template <class ELFT>
uint64_t
ELFState<ELFT>::alignToOffset(ContiguousBlobAccumulator &CBA, uint64_t Align,
                              llvm::Optional<llvm::yaml::Hex64> Offset) {
  uint64_t CurrentOffset = CBA.getOffset();
  uint64_t AlignedOffset;

  if (Offset) {
    if ((uint64_t)*Offset < CurrentOffset) {
      reportError("the 'Offset' value (0x" +
                  Twine::utohexstr((uint64_t)*Offset) + ") goes backward");
      return CurrentOffset;
    }

    // We ignore an alignment when an explicit offset has been requested.
    AlignedOffset = *Offset;
  } else {
    AlignedOffset = alignTo(CurrentOffset, std::max(Align, (uint64_t)1));
  }

  CBA.writeZeros(AlignedOffset - CurrentOffset);
  return AlignedOffset;
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::CallGraphProfileSection &Section,
    ContiguousBlobAccumulator &CBA) {
  if (Section.EntSize)
    SHeader.sh_entsize = *Section.EntSize;
  else
    SHeader.sh_entsize = 16;

  unsigned Link = 0;
  if (!Section.Link && !ExcludedSectionHeaders.count(".symtab") &&
      SN2I.lookup(".symtab", Link))
    SHeader.sh_link = Link;

  if (!Section.Entries)
    return;

  for (const ELFYAML::CallGraphEntry &E : *Section.Entries) {
    unsigned From = toSymbolIndex(E.From, Section.Name, /*IsDynamic=*/false);
    unsigned To = toSymbolIndex(E.To, Section.Name, /*IsDynamic=*/false);

    CBA.write<uint32_t>(From, ELFT::TargetEndianness);
    CBA.write<uint32_t>(To, ELFT::TargetEndianness);
    CBA.write<uint64_t>(E.Weight, ELFT::TargetEndianness);
    SHeader.sh_size += 16;
  }
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::HashSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  unsigned Link = 0;
  if (!Section.Link && !ExcludedSectionHeaders.count(".dynsym") &&
      SN2I.lookup(".dynsym", Link))
    SHeader.sh_link = Link;

  if (Section.EntSize)
    SHeader.sh_entsize = *Section.EntSize;
  else
    SHeader.sh_entsize = sizeof(typename ELFT::Word);

  if (!Section.Bucket)
    return;

  if (!Section.Bucket)
    return;

  CBA.write<uint32_t>(
      Section.NBucket.getValueOr(llvm::yaml::Hex64(Section.Bucket->size())),
      ELFT::TargetEndianness);
  CBA.write<uint32_t>(
      Section.NChain.getValueOr(llvm::yaml::Hex64(Section.Chain->size())),
      ELFT::TargetEndianness);

  for (uint32_t Val : *Section.Bucket)
    CBA.write<uint32_t>(Val, ELFT::TargetEndianness);
  for (uint32_t Val : *Section.Chain)
    CBA.write<uint32_t>(Val, ELFT::TargetEndianness);

  SHeader.sh_size = (2 + Section.Bucket->size() + Section.Chain->size()) * 4;
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::VerdefSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  SHeader.sh_info = Section.Info;

  if (!Section.Entries)
    return;

  uint64_t AuxCnt = 0;
  for (size_t I = 0; I < Section.Entries->size(); ++I) {
    const ELFYAML::VerdefEntry &E = (*Section.Entries)[I];

    Elf_Verdef VerDef;
    VerDef.vd_version = E.Version;
    VerDef.vd_flags = E.Flags;
    VerDef.vd_ndx = E.VersionNdx;
    VerDef.vd_hash = E.Hash;
    VerDef.vd_aux = sizeof(Elf_Verdef);
    VerDef.vd_cnt = E.VerNames.size();
    if (I == Section.Entries->size() - 1)
      VerDef.vd_next = 0;
    else
      VerDef.vd_next =
          sizeof(Elf_Verdef) + E.VerNames.size() * sizeof(Elf_Verdaux);
    CBA.write((const char *)&VerDef, sizeof(Elf_Verdef));

    for (size_t J = 0; J < E.VerNames.size(); ++J, ++AuxCnt) {
      Elf_Verdaux VernAux;
      VernAux.vda_name = DotDynstr.getOffset(E.VerNames[J]);
      if (J == E.VerNames.size() - 1)
        VernAux.vda_next = 0;
      else
        VernAux.vda_next = sizeof(Elf_Verdaux);
      CBA.write((const char *)&VernAux, sizeof(Elf_Verdaux));
    }
  }

  SHeader.sh_size = Section.Entries->size() * sizeof(Elf_Verdef) +
                    AuxCnt * sizeof(Elf_Verdaux);
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::VerneedSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  SHeader.sh_info = Section.Info;

  if (!Section.VerneedV)
    return;

  uint64_t AuxCnt = 0;
  for (size_t I = 0; I < Section.VerneedV->size(); ++I) {
    const ELFYAML::VerneedEntry &VE = (*Section.VerneedV)[I];

    Elf_Verneed VerNeed;
    VerNeed.vn_version = VE.Version;
    VerNeed.vn_file = DotDynstr.getOffset(VE.File);
    if (I == Section.VerneedV->size() - 1)
      VerNeed.vn_next = 0;
    else
      VerNeed.vn_next =
          sizeof(Elf_Verneed) + VE.AuxV.size() * sizeof(Elf_Vernaux);
    VerNeed.vn_cnt = VE.AuxV.size();
    VerNeed.vn_aux = sizeof(Elf_Verneed);
    CBA.write((const char *)&VerNeed, sizeof(Elf_Verneed));

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
      CBA.write((const char *)&VernAux, sizeof(Elf_Vernaux));
    }
  }

  SHeader.sh_size = Section.VerneedV->size() * sizeof(Elf_Verneed) +
                    AuxCnt * sizeof(Elf_Vernaux);
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(
    Elf_Shdr &SHeader, const ELFYAML::ARMIndexTableSection &Section,
    ContiguousBlobAccumulator &CBA) {
  if (!Section.Entries)
    return;

  for (const ELFYAML::ARMIndexTableEntry &E : *Section.Entries) {
    CBA.write<uint32_t>(E.Offset, ELFT::TargetEndianness);
    CBA.write<uint32_t>(E.Value, ELFT::TargetEndianness);
  }
  SHeader.sh_size = Section.Entries->size() * 8;
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
  CBA.write((const char *)&Flags, sizeof(Flags));
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::DynamicSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  assert(Section.Type == llvm::ELF::SHT_DYNAMIC &&
         "Section type is not SHT_DYNAMIC");

  if (Section.EntSize)
    SHeader.sh_entsize = *Section.EntSize;
  else
    SHeader.sh_entsize = 2 * sizeof(uintX_t);

  if (!Section.Entries)
    return;

  for (const ELFYAML::DynamicEntry &DE : *Section.Entries) {
    CBA.write<uintX_t>(DE.Tag, ELFT::TargetEndianness);
    CBA.write<uintX_t>(DE.Val, ELFT::TargetEndianness);
  }
  SHeader.sh_size = 2 * sizeof(uintX_t) * Section.Entries->size();
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::AddrsigSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  unsigned Link = 0;
  if (!Section.Link && !ExcludedSectionHeaders.count(".symtab") &&
      SN2I.lookup(".symtab", Link))
    SHeader.sh_link = Link;

  if (!Section.Symbols)
    return;

  if (!Section.Symbols)
    return;

  for (StringRef Sym : *Section.Symbols)
    SHeader.sh_size +=
        CBA.writeULEB128(toSymbolIndex(Sym, Section.Name, /*IsDynamic=*/false));
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::NoteSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  if (!Section.Notes)
    return;

  uint64_t Offset = CBA.tell();
  for (const ELFYAML::NoteEntry &NE : *Section.Notes) {
    // Write name size.
    if (NE.Name.empty())
      CBA.write<uint32_t>(0, ELFT::TargetEndianness);
    else
      CBA.write<uint32_t>(NE.Name.size() + 1, ELFT::TargetEndianness);

    // Write description size.
    if (NE.Desc.binary_size() == 0)
      CBA.write<uint32_t>(0, ELFT::TargetEndianness);
    else
      CBA.write<uint32_t>(NE.Desc.binary_size(), ELFT::TargetEndianness);

    // Write type.
    CBA.write<uint32_t>(NE.Type, ELFT::TargetEndianness);

    // Write name, null terminator and padding.
    if (!NE.Name.empty()) {
      CBA.write(NE.Name.data(), NE.Name.size());
      CBA.write('\0');
      CBA.padToAlignment(4);
    }

    // Write description and padding.
    if (NE.Desc.binary_size() != 0) {
      CBA.writeAsBinary(NE.Desc);
      CBA.padToAlignment(4);
    }
  }

  SHeader.sh_size = CBA.tell() - Offset;
}

template <class ELFT>
void ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                         const ELFYAML::GnuHashSection &Section,
                                         ContiguousBlobAccumulator &CBA) {
  unsigned Link = 0;
  if (!Section.Link && !ExcludedSectionHeaders.count(".dynsym") &&
      SN2I.lookup(".dynsym", Link))
    SHeader.sh_link = Link;

  if (!Section.HashBuckets)
    return;

  if (!Section.Header)
    return;

  // We write the header first, starting with the hash buckets count. Normally
  // it is the number of entries in HashBuckets, but the "NBuckets" property can
  // be used to override this field, which is useful for producing broken
  // objects.
  if (Section.Header->NBuckets)
    CBA.write<uint32_t>(*Section.Header->NBuckets, ELFT::TargetEndianness);
  else
    CBA.write<uint32_t>(Section.HashBuckets->size(), ELFT::TargetEndianness);

  // Write the index of the first symbol in the dynamic symbol table accessible
  // via the hash table.
  CBA.write<uint32_t>(Section.Header->SymNdx, ELFT::TargetEndianness);

  // Write the number of words in the Bloom filter. As above, the "MaskWords"
  // property can be used to set this field to any value.
  if (Section.Header->MaskWords)
    CBA.write<uint32_t>(*Section.Header->MaskWords, ELFT::TargetEndianness);
  else
    CBA.write<uint32_t>(Section.BloomFilter->size(), ELFT::TargetEndianness);

  // Write the shift constant used by the Bloom filter.
  CBA.write<uint32_t>(Section.Header->Shift2, ELFT::TargetEndianness);

  // We've finished writing the header. Now write the Bloom filter.
  for (llvm::yaml::Hex64 Val : *Section.BloomFilter)
    CBA.write<uintX_t>(Val, ELFT::TargetEndianness);

  // Write an array of hash buckets.
  for (llvm::yaml::Hex32 Val : *Section.HashBuckets)
    CBA.write<uint32_t>(Val, ELFT::TargetEndianness);

  // Write an array of hash values.
  for (llvm::yaml::Hex32 Val : *Section.HashValues)
    CBA.write<uint32_t>(Val, ELFT::TargetEndianness);

  SHeader.sh_size = 16 /*Header size*/ +
                    Section.BloomFilter->size() * sizeof(typename ELFT::uint) +
                    Section.HashBuckets->size() * 4 +
                    Section.HashValues->size() * 4;
}

template <class ELFT>
void ELFState<ELFT>::writeFill(ELFYAML::Fill &Fill,
                               ContiguousBlobAccumulator &CBA) {
  size_t PatternSize = Fill.Pattern ? Fill.Pattern->binary_size() : 0;
  if (!PatternSize) {
    CBA.writeZeros(Fill.Size);
    return;
  }

  // Fill the content with the specified pattern.
  uint64_t Written = 0;
  for (; Written + PatternSize <= Fill.Size; Written += PatternSize)
    CBA.writeAsBinary(*Fill.Pattern);
  CBA.writeAsBinary(*Fill.Pattern, Fill.Size - Written);
}

template <class ELFT>
DenseMap<StringRef, size_t> ELFState<ELFT>::buildSectionHeaderReorderMap() {
  if (!Doc.SectionHeaders || Doc.SectionHeaders->NoHeaders)
    return DenseMap<StringRef, size_t>();

  DenseMap<StringRef, size_t> Ret;
  size_t SecNdx = 0;
  StringSet<> Seen;

  auto AddSection = [&](const ELFYAML::SectionHeader &Hdr) {
    if (!Ret.try_emplace(Hdr.Name, ++SecNdx).second)
      reportError("repeated section name: '" + Hdr.Name +
                  "' in the section header description");
    Seen.insert(Hdr.Name);
  };

  if (Doc.SectionHeaders->Sections)
    for (const ELFYAML::SectionHeader &Hdr : *Doc.SectionHeaders->Sections)
      AddSection(Hdr);

  if (Doc.SectionHeaders->Excluded)
    for (const ELFYAML::SectionHeader &Hdr : *Doc.SectionHeaders->Excluded)
      AddSection(Hdr);

  for (const ELFYAML::Section *S : Doc.getSections()) {
    // Ignore special first SHT_NULL section.
    if (S == Doc.getSections().front())
      continue;
    if (!Seen.count(S->Name))
      reportError("section '" + S->Name +
                  "' should be present in the 'Sections' or 'Excluded' lists");
    Seen.erase(S->Name);
  }

  for (const auto &It : Seen)
    reportError("section header contains undefined section '" + It.getKey() +
                "'");
  return Ret;
}

template <class ELFT> void ELFState<ELFT>::buildSectionIndex() {
  // A YAML description can have an explicit section header declaration that
  // allows to change the order of section headers.
  DenseMap<StringRef, size_t> ReorderMap = buildSectionHeaderReorderMap();

  if (HasError)
    return;

  // Build excluded section headers map.
  std::vector<ELFYAML::Section *> Sections = Doc.getSections();
  if (Doc.SectionHeaders) {
    if (Doc.SectionHeaders->Excluded)
      for (const ELFYAML::SectionHeader &Hdr : *Doc.SectionHeaders->Excluded)
        if (!ExcludedSectionHeaders.insert(Hdr.Name).second)
          llvm_unreachable("buildSectionIndex() failed");

    if (Doc.SectionHeaders->NoHeaders.getValueOr(false))
      for (const ELFYAML::Section *S : Sections)
        if (!ExcludedSectionHeaders.insert(S->Name).second)
          llvm_unreachable("buildSectionIndex() failed");
  }

  size_t SecNdx = -1;
  for (const ELFYAML::Section *S : Sections) {
    ++SecNdx;

    size_t Index = ReorderMap.empty() ? SecNdx : ReorderMap.lookup(S->Name);
    if (!SN2I.addName(S->Name, Index))
      llvm_unreachable("buildSectionIndex() failed");

    if (!ExcludedSectionHeaders.count(S->Name))
      DotShStrtab.add(ELFYAML::dropUniqueSuffix(S->Name));
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
  if (Doc.DynamicSymbols)
    Build(*Doc.DynamicSymbols, DynSymN2I);
}

template <class ELFT> void ELFState<ELFT>::finalizeStrings() {
  // Add the regular symbol names to .strtab section.
  if (Doc.Symbols)
    for (const ELFYAML::Symbol &Sym : *Doc.Symbols)
      DotStrtab.add(ELFYAML::dropUniqueSuffix(Sym.Name));
  DotStrtab.finalize();

  // Add the dynamic symbol names to .dynstr section.
  if (Doc.DynamicSymbols)
    for (const ELFYAML::Symbol &Sym : *Doc.DynamicSymbols)
      DotDynstr.add(ELFYAML::dropUniqueSuffix(Sym.Name));

  // SHT_GNU_verdef and SHT_GNU_verneed sections might also
  // add strings to .dynstr section.
  for (const ELFYAML::Chunk *Sec : Doc.getSections()) {
    if (auto VerNeed = dyn_cast<ELFYAML::VerneedSection>(Sec)) {
      if (VerNeed->VerneedV) {
        for (const ELFYAML::VerneedEntry &VE : *VerNeed->VerneedV) {
          DotDynstr.add(VE.File);
          for (const ELFYAML::VernauxEntry &Aux : VE.AuxV)
            DotDynstr.add(Aux.Name);
        }
      }
    } else if (auto VerDef = dyn_cast<ELFYAML::VerdefSection>(Sec)) {
      if (VerDef->Entries)
        for (const ELFYAML::VerdefEntry &E : *VerDef->Entries)
          for (StringRef Name : E.VerNames)
            DotDynstr.add(Name);
    }
  }

  DotDynstr.finalize();
}

template <class ELFT>
bool ELFState<ELFT>::writeELF(raw_ostream &OS, ELFYAML::Object &Doc,
                              yaml::ErrorHandler EH, uint64_t MaxSize) {
  ELFState<ELFT> State(Doc, EH);
  if (State.HasError)
    return false;

  // Finalize .strtab and .dynstr sections. We do that early because want to
  // finalize the string table builders before writing the content of the
  // sections that might want to use them.
  State.finalizeStrings();

  State.buildSectionIndex();
  State.buildSymbolIndexes();

  if (State.HasError)
    return false;

  std::vector<Elf_Phdr> PHeaders;
  State.initProgramHeaders(PHeaders);

  // XXX: This offset is tightly coupled with the order that we write
  // things to `OS`.
  const size_t SectionContentBeginOffset =
      sizeof(Elf_Ehdr) + sizeof(Elf_Phdr) * Doc.ProgramHeaders.size();
  // It is quite easy to accidentally create output with yaml2obj that is larger
  // than intended, for example, due to an issue in the YAML description.
  // We limit the maximum allowed output size, but also provide a command line
  // option to change this limitation.
  ContiguousBlobAccumulator CBA(SectionContentBeginOffset, MaxSize);

  std::vector<Elf_Shdr> SHeaders;
  State.initSectionHeaders(SHeaders, CBA);

  // Now we can decide segment offsets.
  State.setProgramHeaderLayout(PHeaders, SHeaders);

  // If needed, align the start of the section header table, which is written
  // after all section data.
  const bool HasSectionHeaders =
      !Doc.SectionHeaders || !Doc.SectionHeaders->NoHeaders.getValueOr(false);
  Optional<uint64_t> SHOff;
  if (HasSectionHeaders)
    SHOff = State.alignToOffset(CBA, sizeof(typename ELFT::uint),
                                /*Offset=*/None);
  bool ReachedLimit = SHOff.getValueOr(CBA.getOffset()) +
                          arrayDataSize(makeArrayRef(SHeaders)) >
                      MaxSize;
  if (Error E = CBA.takeLimitError()) {
    // We report a custom error message instead below.
    consumeError(std::move(E));
    ReachedLimit = true;
  }

  if (ReachedLimit)
    State.reportError(
        "the desired output size is greater than permitted. Use the "
        "--max-size option to change the limit");

  if (State.HasError)
    return false;

  State.writeELFHeader(OS, SHOff);
  writeArrayData(OS, makeArrayRef(PHeaders));
  CBA.writeBlobToStream(OS);
  if (HasSectionHeaders)
    writeArrayData(OS, makeArrayRef(SHeaders));
  return true;
}

namespace llvm {
namespace yaml {

bool yaml2elf(llvm::ELFYAML::Object &Doc, raw_ostream &Out, ErrorHandler EH,
              uint64_t MaxSize) {
  bool IsLE = Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB);
  bool Is64Bit = Doc.Header.Class == ELFYAML::ELF_ELFCLASS(ELF::ELFCLASS64);
  if (Is64Bit) {
    if (IsLE)
      return ELFState<object::ELF64LE>::writeELF(Out, Doc, EH, MaxSize);
    return ELFState<object::ELF64BE>::writeELF(Out, Doc, EH, MaxSize);
  }
  if (IsLE)
    return ELFState<object::ELF32LE>::writeELF(Out, Doc, EH, MaxSize);
  return ELFState<object::ELF32BE>::writeELF(Out, Doc, EH, MaxSize);
}

} // namespace yaml
} // namespace llvm
