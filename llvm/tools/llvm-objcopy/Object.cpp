//===- Object.cpp ---------------------------------------------------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Object.h"
#include "llvm-objcopy.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

using namespace llvm;
using namespace object;
using namespace ELF;

template <class ELFT> void Segment::writeHeader(FileOutputBuffer &Out) const {
  using Elf_Ehdr = typename ELFT::Ehdr;
  using Elf_Phdr = typename ELFT::Phdr;

  uint8_t *Buf = Out.getBufferStart();
  Buf += sizeof(Elf_Ehdr) + Index * sizeof(Elf_Phdr);
  Elf_Phdr &Phdr = *reinterpret_cast<Elf_Phdr *>(Buf);
  Phdr.p_type = Type;
  Phdr.p_flags = Flags;
  Phdr.p_offset = Offset;
  Phdr.p_vaddr = VAddr;
  Phdr.p_paddr = PAddr;
  Phdr.p_filesz = FileSize;
  Phdr.p_memsz = MemSize;
  Phdr.p_align = Align;
}

void Segment::writeSegment(FileOutputBuffer &Out) const {
  uint8_t *Buf = Out.getBufferStart() + Offset;
  // We want to maintain segments' interstitial data and contents exactly.
  // This lets us just copy segments directly.
  std::copy(std::begin(Contents), std::end(Contents), Buf);
}

void SectionBase::removeSectionReferences(const SectionBase *Sec) {}
void SectionBase::initialize(SectionTableRef SecTable) {}
void SectionBase::finalize() {}

template <class ELFT>
void SectionBase::writeHeader(FileOutputBuffer &Out) const {
  uint8_t *Buf = Out.getBufferStart();
  Buf += HeaderOffset;
  typename ELFT::Shdr &Shdr = *reinterpret_cast<typename ELFT::Shdr *>(Buf);
  Shdr.sh_name = NameIndex;
  Shdr.sh_type = Type;
  Shdr.sh_flags = Flags;
  Shdr.sh_addr = Addr;
  Shdr.sh_offset = Offset;
  Shdr.sh_size = Size;
  Shdr.sh_link = Link;
  Shdr.sh_info = Info;
  Shdr.sh_addralign = Align;
  Shdr.sh_entsize = EntrySize;
}

void Section::writeSection(FileOutputBuffer &Out) const {
  if (Type == SHT_NOBITS)
    return;
  uint8_t *Buf = Out.getBufferStart() + Offset;
  std::copy(std::begin(Contents), std::end(Contents), Buf);
}

void OwnedDataSection::writeSection(FileOutputBuffer &Out) const {
  uint8_t *Buf = Out.getBufferStart() + Offset;
  std::copy(std::begin(Data), std::end(Data), Buf);
}

void StringTableSection::addString(StringRef Name) {
  StrTabBuilder.add(Name);
  Size = StrTabBuilder.getSize();
}

uint32_t StringTableSection::findIndex(StringRef Name) const {
  return StrTabBuilder.getOffset(Name);
}

void StringTableSection::finalize() { StrTabBuilder.finalize(); }

void StringTableSection::writeSection(FileOutputBuffer &Out) const {
  StrTabBuilder.write(Out.getBufferStart() + Offset);
}

static bool isValidReservedSectionIndex(uint16_t Index, uint16_t Machine) {
  switch (Index) {
  case SHN_ABS:
  case SHN_COMMON:
    return true;
  }
  if (Machine == EM_HEXAGON) {
    switch (Index) {
    case SHN_HEXAGON_SCOMMON:
    case SHN_HEXAGON_SCOMMON_2:
    case SHN_HEXAGON_SCOMMON_4:
    case SHN_HEXAGON_SCOMMON_8:
      return true;
    }
  }
  return false;
}

uint16_t Symbol::getShndx() const {
  if (DefinedIn != nullptr) {
    return DefinedIn->Index;
  }
  switch (ShndxType) {
  // This means that we don't have a defined section but we do need to
  // output a legitimate section index.
  case SYMBOL_SIMPLE_INDEX:
    return SHN_UNDEF;
  case SYMBOL_ABS:
  case SYMBOL_COMMON:
  case SYMBOL_HEXAGON_SCOMMON:
  case SYMBOL_HEXAGON_SCOMMON_2:
  case SYMBOL_HEXAGON_SCOMMON_4:
  case SYMBOL_HEXAGON_SCOMMON_8:
    return static_cast<uint16_t>(ShndxType);
  }
  llvm_unreachable("Symbol with invalid ShndxType encountered");
}

void SymbolTableSection::addSymbol(StringRef Name, uint8_t Bind, uint8_t Type,
                                   SectionBase *DefinedIn, uint64_t Value,
                                   uint8_t Visibility, uint16_t Shndx,
                                   uint64_t Sz) {
  Symbol Sym;
  Sym.Name = Name;
  Sym.Binding = Bind;
  Sym.Type = Type;
  Sym.DefinedIn = DefinedIn;
  if (DefinedIn == nullptr) {
    if (Shndx >= SHN_LORESERVE)
      Sym.ShndxType = static_cast<SymbolShndxType>(Shndx);
    else
      Sym.ShndxType = SYMBOL_SIMPLE_INDEX;
  }
  Sym.Value = Value;
  Sym.Visibility = Visibility;
  Sym.Size = Sz;
  Sym.Index = Symbols.size();
  Symbols.emplace_back(llvm::make_unique<Symbol>(Sym));
  Size += this->EntrySize;
}

void SymbolTableSection::removeSectionReferences(const SectionBase *Sec) {
  if (SymbolNames == Sec) {
    error("String table " + SymbolNames->Name +
          " cannot be removed because it is referenced by the symbol table " +
          this->Name);
  }
  auto Iter =
      std::remove_if(std::begin(Symbols), std::end(Symbols),
                     [=](const SymPtr &Sym) { return Sym->DefinedIn == Sec; });
  Size -= (std::end(Symbols) - Iter) * this->EntrySize;
  Symbols.erase(Iter, std::end(Symbols));
}

void SymbolTableSection::localize(
    std::function<bool(const Symbol &)> ToLocalize) {
  for (const auto &Sym : Symbols) {
    if (ToLocalize(*Sym))
      Sym->Binding = STB_LOCAL;
  }

  // Now that the local symbols aren't grouped at the start we have to reorder
  // the symbols to respect this property.
  std::stable_partition(
      std::begin(Symbols), std::end(Symbols),
      [](const SymPtr &Sym) { return Sym->Binding == STB_LOCAL; });

  // Lastly we fix the symbol indexes.
  uint32_t Index = 0;
  for (auto &Sym : Symbols)
    Sym->Index = Index++;
}

void SymbolTableSection::initialize(SectionTableRef SecTable) {
  Size = 0;
  setStrTab(SecTable.getSectionOfType<StringTableSection>(
      Link,
      "Symbol table has link index of " + Twine(Link) +
          " which is not a valid index",
      "Symbol table has link index of " + Twine(Link) +
          " which is not a string table"));
}

void SymbolTableSection::finalize() {
  // Make sure SymbolNames is finalized before getting name indexes.
  SymbolNames->finalize();

  uint32_t MaxLocalIndex = 0;
  for (auto &Sym : Symbols) {
    Sym->NameIndex = SymbolNames->findIndex(Sym->Name);
    if (Sym->Binding == STB_LOCAL)
      MaxLocalIndex = std::max(MaxLocalIndex, Sym->Index);
  }
  // Now we need to set the Link and Info fields.
  Link = SymbolNames->Index;
  Info = MaxLocalIndex + 1;
}

void SymbolTableSection::addSymbolNames() {
  // Add all of our strings to SymbolNames so that SymbolNames has the right
  // size before layout is decided.
  for (auto &Sym : Symbols)
    SymbolNames->addString(Sym->Name);
}

const Symbol *SymbolTableSection::getSymbolByIndex(uint32_t Index) const {
  if (Symbols.size() <= Index)
    error("Invalid symbol index: " + Twine(Index));
  return Symbols[Index].get();
}

template <class ELFT>
void SymbolTableSectionImpl<ELFT>::writeSection(FileOutputBuffer &Out) const {
  uint8_t *Buf = Out.getBufferStart();
  Buf += Offset;
  typename ELFT::Sym *Sym = reinterpret_cast<typename ELFT::Sym *>(Buf);
  // Loop though symbols setting each entry of the symbol table.
  for (auto &Symbol : Symbols) {
    Sym->st_name = Symbol->NameIndex;
    Sym->st_value = Symbol->Value;
    Sym->st_size = Symbol->Size;
    Sym->st_other = Symbol->Visibility;
    Sym->setBinding(Symbol->Binding);
    Sym->setType(Symbol->Type);
    Sym->st_shndx = Symbol->getShndx();
    ++Sym;
  }
}

template <class SymTabType>
void RelocSectionWithSymtabBase<SymTabType>::removeSectionReferences(
    const SectionBase *Sec) {
  if (Symbols == Sec) {
    error("Symbol table " + Symbols->Name + " cannot be removed because it is "
                                            "referenced by the relocation "
                                            "section " +
          this->Name);
  }
}

template <class SymTabType>
void RelocSectionWithSymtabBase<SymTabType>::initialize(
    SectionTableRef SecTable) {
  setSymTab(SecTable.getSectionOfType<SymTabType>(
      Link,
      "Link field value " + Twine(Link) + " in section " + Name + " is invalid",
      "Link field value " + Twine(Link) + " in section " + Name +
          " is not a symbol table"));

  if (Info != SHN_UNDEF)
    setSection(SecTable.getSection(Info,
                                   "Info field value " + Twine(Info) +
                                       " in section " + Name + " is invalid"));
  else
    setSection(nullptr);
}

template <class SymTabType>
void RelocSectionWithSymtabBase<SymTabType>::finalize() {
  this->Link = Symbols->Index;
  if (SecToApplyRel != nullptr)
    this->Info = SecToApplyRel->Index;
}

template <class ELFT>
void setAddend(Elf_Rel_Impl<ELFT, false> &Rel, uint64_t Addend) {}

template <class ELFT>
void setAddend(Elf_Rel_Impl<ELFT, true> &Rela, uint64_t Addend) {
  Rela.r_addend = Addend;
}

template <class ELFT>
template <class T>
void RelocationSection<ELFT>::writeRel(T *Buf) const {
  for (const auto &Reloc : Relocations) {
    Buf->r_offset = Reloc.Offset;
    setAddend(*Buf, Reloc.Addend);
    Buf->setSymbolAndType(Reloc.RelocSymbol->Index, Reloc.Type, false);
    ++Buf;
  }
}

template <class ELFT>
void RelocationSection<ELFT>::writeSection(FileOutputBuffer &Out) const {
  uint8_t *Buf = Out.getBufferStart() + Offset;
  if (Type == SHT_REL)
    writeRel(reinterpret_cast<Elf_Rel *>(Buf));
  else
    writeRel(reinterpret_cast<Elf_Rela *>(Buf));
}

void DynamicRelocationSection::writeSection(FileOutputBuffer &Out) const {
  std::copy(std::begin(Contents), std::end(Contents),
            Out.getBufferStart() + Offset);
}

void SectionWithStrTab::removeSectionReferences(const SectionBase *Sec) {
  if (StrTab == Sec) {
    error("String table " + StrTab->Name + " cannot be removed because it is "
                                           "referenced by the section " +
          this->Name);
  }
}

bool SectionWithStrTab::classof(const SectionBase *S) {
  return isa<DynamicSymbolTableSection>(S) || isa<DynamicSection>(S);
}

void SectionWithStrTab::initialize(SectionTableRef SecTable) {
  auto StrTab = SecTable.getSection(Link,
                                    "Link field value " + Twine(Link) +
                                        " in section " + Name + " is invalid");
  if (StrTab->Type != SHT_STRTAB) {
    error("Link field value " + Twine(Link) + " in section " + Name +
          " is not a string table");
  }
  setStrTab(StrTab);
}

void SectionWithStrTab::finalize() { this->Link = StrTab->Index; }

template <class ELFT>
void GnuDebugLinkSection<ELFT>::init(StringRef File, StringRef Data) {
  FileName = sys::path::stem(File);
  // The format for the .gnu_debuglink starts with the stemmed file name and is
  // followed by a null terminator and then the CRC32 of the file. The CRC32
  // should be 4 byte aligned. So we add the FileName size, a 1 for the null
  // byte, and then finally push the size to alignment and add 4.
  Size = alignTo(FileName.size() + 1, 4) + 4;
  // The CRC32 will only be aligned if we align the whole section.
  Align = 4;
  Type = ELF::SHT_PROGBITS;
  Name = ".gnu_debuglink";
  // For sections not found in segments, OriginalOffset is only used to
  // establish the order that sections should go in. By using the maximum
  // possible offset we cause this section to wind up at the end.
  OriginalOffset = std::numeric_limits<uint64_t>::max();
  JamCRC crc;
  crc.update(ArrayRef<char>(Data.data(), Data.size()));
  // The CRC32 value needs to be complemented because the JamCRC dosn't
  // finalize the CRC32 value. It also dosn't negate the initial CRC32 value
  // but it starts by default at 0xFFFFFFFF which is the complement of zero.
  CRC32 = ~crc.getCRC();
}

template <class ELFT>
GnuDebugLinkSection<ELFT>::GnuDebugLinkSection(StringRef File)
    : FileName(File) {
  // Read in the file to compute the CRC of it.
  auto DebugOrErr = MemoryBuffer::getFile(File);
  if (!DebugOrErr)
    error("'" + File + "': " + DebugOrErr.getError().message());
  auto Debug = std::move(*DebugOrErr);
  init(File, Debug->getBuffer());
}

template <class ELFT>
void GnuDebugLinkSection<ELFT>::writeSection(FileOutputBuffer &Out) const {
  auto Buf = Out.getBufferStart() + Offset;
  char *File = reinterpret_cast<char *>(Buf);
  Elf_Word *CRC = reinterpret_cast<Elf_Word *>(Buf + Size - sizeof(Elf_Word));
  *CRC = CRC32;
  std::copy(std::begin(FileName), std::end(FileName), File);
}

// Returns true IFF a section is wholly inside the range of a segment
static bool sectionWithinSegment(const SectionBase &Section,
                                 const Segment &Segment) {
  // If a section is empty it should be treated like it has a size of 1. This is
  // to clarify the case when an empty section lies on a boundary between two
  // segments and ensures that the section "belongs" to the second segment and
  // not the first.
  uint64_t SecSize = Section.Size ? Section.Size : 1;
  return Segment.Offset <= Section.OriginalOffset &&
         Segment.Offset + Segment.FileSize >= Section.OriginalOffset + SecSize;
}

// Returns true IFF a segment's original offset is inside of another segment's
// range.
static bool segmentOverlapsSegment(const Segment &Child,
                                   const Segment &Parent) {

  return Parent.OriginalOffset <= Child.OriginalOffset &&
         Parent.OriginalOffset + Parent.FileSize > Child.OriginalOffset;
}

static bool compareSegments(const Segment *A, const Segment *B) {
  // Any segment without a parent segment should come before a segment
  // that has a parent segment.
  if (A->OriginalOffset < B->OriginalOffset)
    return true;
  if (A->OriginalOffset > B->OriginalOffset)
    return false;
  return A->Index < B->Index;
}

template <class ELFT>
void Object<ELFT>::readProgramHeaders(const ELFFile<ELFT> &ElfFile) {
  uint32_t Index = 0;
  for (const auto &Phdr : unwrapOrError(ElfFile.program_headers())) {
    ArrayRef<uint8_t> Data{ElfFile.base() + Phdr.p_offset,
                           (size_t)Phdr.p_filesz};
    Segments.emplace_back(llvm::make_unique<Segment>(Data));
    Segment &Seg = *Segments.back();
    Seg.Type = Phdr.p_type;
    Seg.Flags = Phdr.p_flags;
    Seg.OriginalOffset = Phdr.p_offset;
    Seg.Offset = Phdr.p_offset;
    Seg.VAddr = Phdr.p_vaddr;
    Seg.PAddr = Phdr.p_paddr;
    Seg.FileSize = Phdr.p_filesz;
    Seg.MemSize = Phdr.p_memsz;
    Seg.Align = Phdr.p_align;
    Seg.Index = Index++;
    for (auto &Section : Sections) {
      if (sectionWithinSegment(*Section, Seg)) {
        Seg.addSection(&*Section);
        if (!Section->ParentSegment ||
            Section->ParentSegment->Offset > Seg.Offset) {
          Section->ParentSegment = &Seg;
        }
      }
    }
  }
  // Now we do an O(n^2) loop through the segments in order to match up
  // segments.
  for (auto &Child : Segments) {
    for (auto &Parent : Segments) {
      // Every segment will overlap with itself but we don't want a segment to
      // be it's own parent so we avoid that situation.
      if (&Child != &Parent && segmentOverlapsSegment(*Child, *Parent)) {
        // We want a canonical "most parental" segment but this requires
        // inspecting the ParentSegment.
        if (compareSegments(Parent.get(), Child.get()))
          if (Child->ParentSegment == nullptr ||
              compareSegments(Parent.get(), Child->ParentSegment)) {
            Child->ParentSegment = Parent.get();
          }
      }
    }
  }
}

template <class ELFT>
void Object<ELFT>::initSymbolTable(const object::ELFFile<ELFT> &ElfFile,
                                   SymbolTableSection *SymTab,
                                   SectionTableRef SecTable) {
  const Elf_Shdr &Shdr = *unwrapOrError(ElfFile.getSection(SymTab->Index));
  StringRef StrTabData = unwrapOrError(ElfFile.getStringTableForSymtab(Shdr));

  for (const auto &Sym : unwrapOrError(ElfFile.symbols(&Shdr))) {
    SectionBase *DefSection = nullptr;
    StringRef Name = unwrapOrError(Sym.getName(StrTabData));

    if (Sym.st_shndx >= SHN_LORESERVE) {
      if (!isValidReservedSectionIndex(Sym.st_shndx, Machine)) {
        error(
            "Symbol '" + Name +
            "' has unsupported value greater than or equal to SHN_LORESERVE: " +
            Twine(Sym.st_shndx));
      }
    } else if (Sym.st_shndx != SHN_UNDEF) {
      DefSection = SecTable.getSection(
          Sym.st_shndx,
          "Symbol '" + Name + "' is defined in invalid section with index " +
              Twine(Sym.st_shndx));
    }

    SymTab->addSymbol(Name, Sym.getBinding(), Sym.getType(), DefSection,
                      Sym.getValue(), Sym.st_other, Sym.st_shndx, Sym.st_size);
  }
}

template <class ELFT>
static void getAddend(uint64_t &ToSet, const Elf_Rel_Impl<ELFT, false> &Rel) {}

template <class ELFT>
static void getAddend(uint64_t &ToSet, const Elf_Rel_Impl<ELFT, true> &Rela) {
  ToSet = Rela.r_addend;
}

template <class ELFT, class T>
void initRelocations(RelocationSection<ELFT> *Relocs,
                     SymbolTableSection *SymbolTable, T RelRange) {
  for (const auto &Rel : RelRange) {
    Relocation ToAdd;
    ToAdd.Offset = Rel.r_offset;
    getAddend(ToAdd.Addend, Rel);
    ToAdd.Type = Rel.getType(false);
    ToAdd.RelocSymbol = SymbolTable->getSymbolByIndex(Rel.getSymbol(false));
    Relocs->addRelocation(ToAdd);
  }
}

SectionBase *SectionTableRef::getSection(uint16_t Index, Twine ErrMsg) {
  if (Index == SHN_UNDEF || Index > Sections.size())
    error(ErrMsg);
  return Sections[Index - 1].get();
}

template <class T>
T *SectionTableRef::getSectionOfType(uint16_t Index, Twine IndexErrMsg,
                                     Twine TypeErrMsg) {
  if (T *Sec = dyn_cast<T>(getSection(Index, IndexErrMsg)))
    return Sec;
  error(TypeErrMsg);
}

template <class ELFT>
std::unique_ptr<SectionBase>
Object<ELFT>::makeSection(const object::ELFFile<ELFT> &ElfFile,
                          const Elf_Shdr &Shdr) {
  ArrayRef<uint8_t> Data;
  switch (Shdr.sh_type) {
  case SHT_REL:
  case SHT_RELA:
    if (Shdr.sh_flags & SHF_ALLOC) {
      Data = unwrapOrError(ElfFile.getSectionContents(&Shdr));
      return llvm::make_unique<DynamicRelocationSection>(Data);
    }
    return llvm::make_unique<RelocationSection<ELFT>>();
  case SHT_STRTAB:
    // If a string table is allocated we don't want to mess with it. That would
    // mean altering the memory image. There are no special link types or
    // anything so we can just use a Section.
    if (Shdr.sh_flags & SHF_ALLOC) {
      Data = unwrapOrError(ElfFile.getSectionContents(&Shdr));
      return llvm::make_unique<Section>(Data);
    }
    return llvm::make_unique<StringTableSection>();
  case SHT_HASH:
  case SHT_GNU_HASH:
    // Hash tables should refer to SHT_DYNSYM which we're not going to change.
    // Because of this we don't need to mess with the hash tables either.
    Data = unwrapOrError(ElfFile.getSectionContents(&Shdr));
    return llvm::make_unique<Section>(Data);
  case SHT_DYNSYM:
    Data = unwrapOrError(ElfFile.getSectionContents(&Shdr));
    return llvm::make_unique<DynamicSymbolTableSection>(Data);
  case SHT_DYNAMIC:
    Data = unwrapOrError(ElfFile.getSectionContents(&Shdr));
    return llvm::make_unique<DynamicSection>(Data);
  case SHT_SYMTAB: {
    auto SymTab = llvm::make_unique<SymbolTableSectionImpl<ELFT>>();
    SymbolTable = SymTab.get();
    return std::move(SymTab);
  }
  case SHT_NOBITS:
    return llvm::make_unique<Section>(Data);
  default:
    Data = unwrapOrError(ElfFile.getSectionContents(&Shdr));
    return llvm::make_unique<Section>(Data);
  }
}

template <class ELFT>
SectionTableRef Object<ELFT>::readSectionHeaders(const ELFFile<ELFT> &ElfFile) {
  uint32_t Index = 0;
  for (const auto &Shdr : unwrapOrError(ElfFile.sections())) {
    if (Index == 0) {
      ++Index;
      continue;
    }
    SecPtr Sec = makeSection(ElfFile, Shdr);
    Sec->Name = unwrapOrError(ElfFile.getSectionName(&Shdr));
    Sec->Type = Shdr.sh_type;
    Sec->Flags = Shdr.sh_flags;
    Sec->Addr = Shdr.sh_addr;
    Sec->Offset = Shdr.sh_offset;
    Sec->OriginalOffset = Shdr.sh_offset;
    Sec->Size = Shdr.sh_size;
    Sec->Link = Shdr.sh_link;
    Sec->Info = Shdr.sh_info;
    Sec->Align = Shdr.sh_addralign;
    Sec->EntrySize = Shdr.sh_entsize;
    Sec->Index = Index++;
    Sections.push_back(std::move(Sec));
  }

  SectionTableRef SecTable(Sections);

  // Now that all of the sections have been added we can fill out some extra
  // details about symbol tables. We need the symbol table filled out before
  // any relocations.
  if (SymbolTable) {
    SymbolTable->initialize(SecTable);
    initSymbolTable(ElfFile, SymbolTable, SecTable);
  }

  // Now that all sections and symbols have been added we can add
  // relocations that reference symbols and set the link and info fields for
  // relocation sections.
  for (auto &Section : Sections) {
    if (Section.get() == SymbolTable)
      continue;
    Section->initialize(SecTable);
    if (auto RelSec = dyn_cast<RelocationSection<ELFT>>(Section.get())) {
      auto Shdr = unwrapOrError(ElfFile.sections()).begin() + RelSec->Index;
      if (RelSec->Type == SHT_REL)
        initRelocations(RelSec, SymbolTable, unwrapOrError(ElfFile.rels(Shdr)));
      else
        initRelocations(RelSec, SymbolTable,
                        unwrapOrError(ElfFile.relas(Shdr)));
    }
  }

  return SecTable;
}

template <class ELFT> Object<ELFT>::Object(const ELFObjectFile<ELFT> &Obj) {
  const auto &ElfFile = *Obj.getELFFile();
  const auto &Ehdr = *ElfFile.getHeader();

  std::copy(Ehdr.e_ident, Ehdr.e_ident + 16, Ident);
  Type = Ehdr.e_type;
  Machine = Ehdr.e_machine;
  Version = Ehdr.e_version;
  Entry = Ehdr.e_entry;
  Flags = Ehdr.e_flags;

  SectionTableRef SecTable = readSectionHeaders(ElfFile);
  readProgramHeaders(ElfFile);

  SectionNames = SecTable.getSectionOfType<StringTableSection>(
      Ehdr.e_shstrndx,
      "e_shstrndx field value " + Twine(Ehdr.e_shstrndx) + " in elf header " +
          " is invalid",
      "e_shstrndx field value " + Twine(Ehdr.e_shstrndx) + " in elf header " +
          " is not a string table");
}

template <class ELFT>
void Object<ELFT>::writeHeader(FileOutputBuffer &Out) const {
  uint8_t *Buf = Out.getBufferStart();
  Elf_Ehdr &Ehdr = *reinterpret_cast<Elf_Ehdr *>(Buf);
  std::copy(Ident, Ident + 16, Ehdr.e_ident);
  Ehdr.e_type = Type;
  Ehdr.e_machine = Machine;
  Ehdr.e_version = Version;
  Ehdr.e_entry = Entry;
  Ehdr.e_phoff = sizeof(Elf_Ehdr);
  Ehdr.e_flags = Flags;
  Ehdr.e_ehsize = sizeof(Elf_Ehdr);
  Ehdr.e_phentsize = sizeof(Elf_Phdr);
  Ehdr.e_phnum = Segments.size();
  Ehdr.e_shentsize = sizeof(Elf_Shdr);
  if (WriteSectionHeaders) {
    Ehdr.e_shoff = SHOffset;
    Ehdr.e_shnum = Sections.size() + 1;
    Ehdr.e_shstrndx = SectionNames->Index;
  } else {
    Ehdr.e_shoff = 0;
    Ehdr.e_shnum = 0;
    Ehdr.e_shstrndx = 0;
  }
}

template <class ELFT>
void Object<ELFT>::writeProgramHeaders(FileOutputBuffer &Out) const {
  for (auto &Phdr : Segments)
    Phdr->template writeHeader<ELFT>(Out);
}

template <class ELFT>
void Object<ELFT>::writeSectionHeaders(FileOutputBuffer &Out) const {
  uint8_t *Buf = Out.getBufferStart() + SHOffset;
  // This reference serves to write the dummy section header at the begining
  // of the file. It is not used for anything else
  Elf_Shdr &Shdr = *reinterpret_cast<Elf_Shdr *>(Buf);
  Shdr.sh_name = 0;
  Shdr.sh_type = SHT_NULL;
  Shdr.sh_flags = 0;
  Shdr.sh_addr = 0;
  Shdr.sh_offset = 0;
  Shdr.sh_size = 0;
  Shdr.sh_link = 0;
  Shdr.sh_info = 0;
  Shdr.sh_addralign = 0;
  Shdr.sh_entsize = 0;

  for (auto &Section : Sections)
    Section->template writeHeader<ELFT>(Out);
}

template <class ELFT>
void Object<ELFT>::writeSectionData(FileOutputBuffer &Out) const {
  for (auto &Section : Sections)
    Section->writeSection(Out);
}

template <class ELFT>
void Object<ELFT>::removeSections(
    std::function<bool(const SectionBase &)> ToRemove) {

  auto Iter = std::stable_partition(
      std::begin(Sections), std::end(Sections), [=](const SecPtr &Sec) {
        if (ToRemove(*Sec))
          return false;
        if (auto RelSec = dyn_cast<RelocationSectionBase>(Sec.get())) {
          if (auto ToRelSec = RelSec->getSection())
            return !ToRemove(*ToRelSec);
        }
        return true;
      });
  if (SymbolTable != nullptr && ToRemove(*SymbolTable))
    SymbolTable = nullptr;
  if (ToRemove(*SectionNames)) {
    if (WriteSectionHeaders)
      error("Cannot remove " + SectionNames->Name +
            " because it is the section header string table.");
    SectionNames = nullptr;
  }
  // Now make sure there are no remaining references to the sections that will
  // be removed. Sometimes it is impossible to remove a reference so we emit
  // an error here instead.
  for (auto &RemoveSec : make_range(Iter, std::end(Sections))) {
    for (auto &Segment : Segments)
      Segment->removeSection(RemoveSec.get());
    for (auto &KeepSec : make_range(std::begin(Sections), Iter))
      KeepSec->removeSectionReferences(RemoveSec.get());
  }
  // Now finally get rid of them all togethor.
  Sections.erase(Iter, std::end(Sections));
}

template <class ELFT>
void Object<ELFT>::addSection(StringRef SecName, ArrayRef<uint8_t> Data) {
  auto Sec = llvm::make_unique<OwnedDataSection>(SecName, Data);
  Sec->OriginalOffset = ~0ULL;
  Sections.push_back(std::move(Sec));
}

template <class ELFT> void Object<ELFT>::addGnuDebugLink(StringRef File) {
  Sections.emplace_back(llvm::make_unique<GnuDebugLinkSection<ELFT>>(File));
}

template <class ELFT> void ELFObject<ELFT>::sortSections() {
  // Put all sections in offset order. Maintain the ordering as closely as
  // possible while meeting that demand however.
  auto CompareSections = [](const SecPtr &A, const SecPtr &B) {
    return A->OriginalOffset < B->OriginalOffset;
  };
  std::stable_sort(std::begin(this->Sections), std::end(this->Sections),
                   CompareSections);
}

static uint64_t alignToAddr(uint64_t Offset, uint64_t Addr, uint64_t Align) {
  // Calculate Diff such that (Offset + Diff) & -Align == Addr & -Align.
  if (Align == 0)
    Align = 1;
  auto Diff =
      static_cast<int64_t>(Addr % Align) - static_cast<int64_t>(Offset % Align);
  // We only want to add to Offset, however, so if Diff < 0 we can add Align and
  // (Offset + Diff) & -Align == Addr & -Align will still hold.
  if (Diff < 0)
    Diff += Align;
  return Offset + Diff;
}

// Orders segments such that if x = y->ParentSegment then y comes before x.
static void OrderSegments(std::vector<Segment *> &Segments) {
  std::stable_sort(std::begin(Segments), std::end(Segments), compareSegments);
}

// This function finds a consistent layout for a list of segments starting from
// an Offset. It assumes that Segments have been sorted by OrderSegments and
// returns an Offset one past the end of the last segment.
static uint64_t LayoutSegments(std::vector<Segment *> &Segments,
                               uint64_t Offset) {
  assert(std::is_sorted(std::begin(Segments), std::end(Segments),
                        compareSegments));
  // The only way a segment should move is if a section was between two
  // segments and that section was removed. If that section isn't in a segment
  // then it's acceptable, but not ideal, to simply move it to after the
  // segments. So we can simply layout segments one after the other accounting
  // for alignment.
  for (auto &Segment : Segments) {
    // We assume that segments have been ordered by OriginalOffset and Index
    // such that a parent segment will always come before a child segment in
    // OrderedSegments. This means that the Offset of the ParentSegment should
    // already be set and we can set our offset relative to it.
    if (Segment->ParentSegment != nullptr) {
      auto Parent = Segment->ParentSegment;
      Segment->Offset =
          Parent->Offset + Segment->OriginalOffset - Parent->OriginalOffset;
    } else {
      Offset = alignToAddr(Offset, Segment->VAddr, Segment->Align);
      Segment->Offset = Offset;
    }
    Offset = std::max(Offset, Segment->Offset + Segment->FileSize);
  }
  return Offset;
}

// This function finds a consistent layout for a list of sections. It assumes
// that the ->ParentSegment of each section has already been laid out. The
// supplied starting Offset is used for the starting offset of any section that
// does not have a ParentSegment. It returns either the offset given if all
// sections had a ParentSegment or an offset one past the last section if there
// was a section that didn't have a ParentSegment.
template <class SecPtr>
static uint64_t LayoutSections(std::vector<SecPtr> &Sections, uint64_t Offset) {
  // Now the offset of every segment has been set we can assign the offsets
  // of each section. For sections that are covered by a segment we should use
  // the segment's original offset and the section's original offset to compute
  // the offset from the start of the segment. Using the offset from the start
  // of the segment we can assign a new offset to the section. For sections not
  // covered by segments we can just bump Offset to the next valid location.
  uint32_t Index = 1;
  for (auto &Section : Sections) {
    Section->Index = Index++;
    if (Section->ParentSegment != nullptr) {
      auto Segment = Section->ParentSegment;
      Section->Offset =
          Segment->Offset + (Section->OriginalOffset - Segment->OriginalOffset);
    } else {
      Offset = alignTo(Offset, Section->Align == 0 ? 1 : Section->Align);
      Section->Offset = Offset;
      if (Section->Type != SHT_NOBITS)
        Offset += Section->Size;
    }
  }
  return Offset;
}

template <class ELFT> void ELFObject<ELFT>::assignOffsets() {
  // We need a temporary list of segments that has a special order to it
  // so that we know that anytime ->ParentSegment is set that segment has
  // already had its offset properly set.
  std::vector<Segment *> OrderedSegments;
  for (auto &Segment : this->Segments)
    OrderedSegments.push_back(Segment.get());
  OrderSegments(OrderedSegments);
  // The size of ELF + program headers will not change so it is ok to assume
  // that the first offset of the first segment is a good place to start
  // outputting sections. This covers both the standard case and the PT_PHDR
  // case.
  uint64_t Offset;
  if (!OrderedSegments.empty()) {
    Offset = OrderedSegments[0]->Offset;
  } else {
    Offset = sizeof(Elf_Ehdr);
  }
  Offset = LayoutSegments(OrderedSegments, Offset);
  Offset = LayoutSections(this->Sections, Offset);
  // If we need to write the section header table out then we need to align the
  // Offset so that SHOffset is valid.
  if (this->WriteSectionHeaders)
    Offset = alignTo(Offset, sizeof(typename ELFT::Addr));
  this->SHOffset = Offset;
}

template <class ELFT> size_t ELFObject<ELFT>::totalSize() const {
  // We already have the section header offset so we can calculate the total
  // size by just adding up the size of each section header.
  auto NullSectionSize = this->WriteSectionHeaders ? sizeof(Elf_Shdr) : 0;
  return this->SHOffset + this->Sections.size() * sizeof(Elf_Shdr) +
         NullSectionSize;
}

template <class ELFT> void ELFObject<ELFT>::write(FileOutputBuffer &Out) const {
  this->writeHeader(Out);
  this->writeProgramHeaders(Out);
  this->writeSectionData(Out);
  if (this->WriteSectionHeaders)
    this->writeSectionHeaders(Out);
}

template <class ELFT> void ELFObject<ELFT>::finalize() {
  // Make sure we add the names of all the sections.
  if (this->SectionNames != nullptr)
    for (const auto &Section : this->Sections) {
      this->SectionNames->addString(Section->Name);
    }
  // Make sure we add the names of all the symbols.
  if (this->SymbolTable != nullptr)
    this->SymbolTable->addSymbolNames();

  sortSections();
  assignOffsets();

  // Finalize SectionNames first so that we can assign name indexes.
  if (this->SectionNames != nullptr)
    this->SectionNames->finalize();
  // Finally now that all offsets and indexes have been set we can finalize any
  // remaining issues.
  uint64_t Offset = this->SHOffset + sizeof(Elf_Shdr);
  for (auto &Section : this->Sections) {
    Section->HeaderOffset = Offset;
    Offset += sizeof(Elf_Shdr);
    if (this->WriteSectionHeaders)
      Section->NameIndex = this->SectionNames->findIndex(Section->Name);
    Section->finalize();
  }
}

template <class ELFT> size_t BinaryObject<ELFT>::totalSize() const {
  return TotalSize;
}

template <class ELFT>
void BinaryObject<ELFT>::write(FileOutputBuffer &Out) const {
  for (auto &Section : this->Sections) {
    if ((Section->Flags & SHF_ALLOC) == 0)
      continue;
    Section->writeSection(Out);
  }
}

template <class ELFT> void BinaryObject<ELFT>::finalize() {
  // TODO: Create a filter range to construct OrderedSegments from so that this
  // code can be deduped with assignOffsets above. This should also solve the
  // todo below for LayoutSections.
  // We need a temporary list of segments that has a special order to it
  // so that we know that anytime ->ParentSegment is set that segment has
  // already had it's offset properly set. We only want to consider the segments
  // that will affect layout of allocated sections so we only add those.
  std::vector<Segment *> OrderedSegments;
  for (auto &Section : this->Sections) {
    if ((Section->Flags & SHF_ALLOC) != 0 &&
        Section->ParentSegment != nullptr) {
      OrderedSegments.push_back(Section->ParentSegment);
    }
  }
  OrderSegments(OrderedSegments);
  // Because we add a ParentSegment for each section we might have duplicate
  // segments in OrderedSegments. If there were duplicates then LayoutSegments
  // would do very strange things.
  auto End =
      std::unique(std::begin(OrderedSegments), std::end(OrderedSegments));
  OrderedSegments.erase(End, std::end(OrderedSegments));

  // Modify the first segment so that there is no gap at the start. This allows
  // our layout algorithm to proceed as expected while not out writing out the
  // gap at the start.
  if (!OrderedSegments.empty()) {
    auto Seg = OrderedSegments[0];
    auto Sec = Seg->firstSection();
    auto Diff = Sec->OriginalOffset - Seg->OriginalOffset;
    Seg->OriginalOffset += Diff;
    // The size needs to be shrunk as well
    Seg->FileSize -= Diff;
    Seg->MemSize -= Diff;
    // The VAddr needs to be adjusted so that the alignment is correct as well
    Seg->VAddr += Diff;
    Seg->PAddr = Seg->VAddr;
    // We don't want this to be shifted by alignment so we need to set the
    // alignment to zero.
    Seg->Align = 0;
  }

  uint64_t Offset = LayoutSegments(OrderedSegments, 0);

  // TODO: generalize LayoutSections to take a range. Pass a special range
  // constructed from an iterator that skips values for which a predicate does
  // not hold. Then pass such a range to LayoutSections instead of constructing
  // AllocatedSections here.
  std::vector<SectionBase *> AllocatedSections;
  for (auto &Section : this->Sections) {
    if ((Section->Flags & SHF_ALLOC) == 0)
      continue;
    AllocatedSections.push_back(Section.get());
  }
  LayoutSections(AllocatedSections, Offset);

  // Now that every section has been laid out we just need to compute the total
  // file size. This might not be the same as the offset returned by
  // LayoutSections, because we want to truncate the last segment to the end of
  // its last section, to match GNU objcopy's behaviour.
  TotalSize = 0;
  for (const auto &Section : AllocatedSections) {
    if (Section->Type != SHT_NOBITS)
      TotalSize = std::max(TotalSize, Section->Offset + Section->Size);
  }
}

namespace llvm {

template class Object<ELF64LE>;
template class Object<ELF64BE>;
template class Object<ELF32LE>;
template class Object<ELF32BE>;

template class ELFObject<ELF64LE>;
template class ELFObject<ELF64BE>;
template class ELFObject<ELF32LE>;
template class ELFObject<ELF32BE>;

template class BinaryObject<ELF64LE>;
template class BinaryObject<ELF64BE>;
template class BinaryObject<ELF32LE>;
template class BinaryObject<ELF32BE>;

} // end namespace llvm
