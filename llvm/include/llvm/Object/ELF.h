//===- ELF.h - ELF object file implementation -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ELFFile template class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ELF_H
#define LLVM_OBJECT_ELF_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace llvm {
namespace object {

StringRef getELFRelocationTypeName(uint32_t Machine, uint32_t Type);
uint32_t getELFRelativeRelocationType(uint32_t Machine);
StringRef getELFSectionTypeName(uint32_t Machine, uint32_t Type);

// Subclasses of ELFFile may need this for template instantiation
inline std::pair<unsigned char, unsigned char>
getElfArchType(StringRef Object) {
  if (Object.size() < ELF::EI_NIDENT)
    return std::make_pair((uint8_t)ELF::ELFCLASSNONE,
                          (uint8_t)ELF::ELFDATANONE);
  return std::make_pair((uint8_t)Object[ELF::EI_CLASS],
                        (uint8_t)Object[ELF::EI_DATA]);
}

static inline Error createError(const Twine &Err) {
  return make_error<StringError>(Err, object_error::parse_failed);
}

template <class ELFT> class ELFFile;

template <class ELFT>
std::string getSecIndexForError(const ELFFile<ELFT> *Obj,
                                const typename ELFT::Shdr *Sec) {
  auto TableOrErr = Obj->sections();
  if (TableOrErr)
    return "[index " + std::to_string(Sec - &TableOrErr->front()) + "]";
  // To make this helper be more convenient for error reporting purposes we
  // drop the error. But really it should never be triggered. Before this point,
  // our code should have called 'sections()' and reported a proper error on
  // failure.
  llvm::consumeError(TableOrErr.takeError());
  return "[unknown index]";
}

static inline Error defaultWarningHandler(const Twine &Msg) {
  return createError(Msg);
}

template <class ELFT>
class ELFFile {
public:
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  using uintX_t = typename ELFT::uint;
  using Elf_Ehdr = typename ELFT::Ehdr;
  using Elf_Shdr = typename ELFT::Shdr;
  using Elf_Sym = typename ELFT::Sym;
  using Elf_Dyn = typename ELFT::Dyn;
  using Elf_Phdr = typename ELFT::Phdr;
  using Elf_Rel = typename ELFT::Rel;
  using Elf_Rela = typename ELFT::Rela;
  using Elf_Relr = typename ELFT::Relr;
  using Elf_Verdef = typename ELFT::Verdef;
  using Elf_Verdaux = typename ELFT::Verdaux;
  using Elf_Verneed = typename ELFT::Verneed;
  using Elf_Vernaux = typename ELFT::Vernaux;
  using Elf_Versym = typename ELFT::Versym;
  using Elf_Hash = typename ELFT::Hash;
  using Elf_GnuHash = typename ELFT::GnuHash;
  using Elf_Nhdr = typename ELFT::Nhdr;
  using Elf_Note = typename ELFT::Note;
  using Elf_Note_Iterator = typename ELFT::NoteIterator;
  using Elf_Dyn_Range = typename ELFT::DynRange;
  using Elf_Shdr_Range = typename ELFT::ShdrRange;
  using Elf_Sym_Range = typename ELFT::SymRange;
  using Elf_Rel_Range = typename ELFT::RelRange;
  using Elf_Rela_Range = typename ELFT::RelaRange;
  using Elf_Relr_Range = typename ELFT::RelrRange;
  using Elf_Phdr_Range = typename ELFT::PhdrRange;

  // This is a callback that can be passed to a number of functions.
  // It can be used to ignore non-critical errors (warnings), which is
  // useful for dumpers, like llvm-readobj.
  // It accepts a warning message string and returns a success
  // when the warning should be ignored or an error otherwise.
  using WarningHandler = llvm::function_ref<Error(const Twine &Msg)>;

  const uint8_t *base() const { return Buf.bytes_begin(); }

  size_t getBufSize() const { return Buf.size(); }

private:
  StringRef Buf;

  ELFFile(StringRef Object);

public:
  const Elf_Ehdr *getHeader() const {
    return reinterpret_cast<const Elf_Ehdr *>(base());
  }

  template <typename T>
  Expected<const T *> getEntry(uint32_t Section, uint32_t Entry) const;
  template <typename T>
  Expected<const T *> getEntry(const Elf_Shdr *Section, uint32_t Entry) const;

  Expected<StringRef>
  getStringTable(const Elf_Shdr *Section,
                 WarningHandler WarnHandler = &defaultWarningHandler) const;
  Expected<StringRef> getStringTableForSymtab(const Elf_Shdr &Section) const;
  Expected<StringRef> getStringTableForSymtab(const Elf_Shdr &Section,
                                              Elf_Shdr_Range Sections) const;

  Expected<ArrayRef<Elf_Word>> getSHNDXTable(const Elf_Shdr &Section) const;
  Expected<ArrayRef<Elf_Word>> getSHNDXTable(const Elf_Shdr &Section,
                                             Elf_Shdr_Range Sections) const;

  StringRef getRelocationTypeName(uint32_t Type) const;
  void getRelocationTypeName(uint32_t Type,
                             SmallVectorImpl<char> &Result) const;
  uint32_t getRelativeRelocationType() const;

  std::string getDynamicTagAsString(unsigned Arch, uint64_t Type) const;
  std::string getDynamicTagAsString(uint64_t Type) const;

  /// Get the symbol for a given relocation.
  Expected<const Elf_Sym *> getRelocationSymbol(const Elf_Rel *Rel,
                                                const Elf_Shdr *SymTab) const;

  static Expected<ELFFile> create(StringRef Object);

  bool isLE() const {
    return getHeader()->getDataEncoding() == ELF::ELFDATA2LSB;
  }

  bool isMipsELF64() const {
    return getHeader()->e_machine == ELF::EM_MIPS &&
           getHeader()->getFileClass() == ELF::ELFCLASS64;
  }

  bool isMips64EL() const { return isMipsELF64() && isLE(); }

  Expected<Elf_Shdr_Range> sections() const;

  Expected<Elf_Dyn_Range> dynamicEntries() const;

  Expected<const uint8_t *> toMappedAddr(uint64_t VAddr) const;

  Expected<Elf_Sym_Range> symbols(const Elf_Shdr *Sec) const {
    if (!Sec)
      return makeArrayRef<Elf_Sym>(nullptr, nullptr);
    return getSectionContentsAsArray<Elf_Sym>(Sec);
  }

  Expected<Elf_Rela_Range> relas(const Elf_Shdr *Sec) const {
    return getSectionContentsAsArray<Elf_Rela>(Sec);
  }

  Expected<Elf_Rel_Range> rels(const Elf_Shdr *Sec) const {
    return getSectionContentsAsArray<Elf_Rel>(Sec);
  }

  Expected<Elf_Relr_Range> relrs(const Elf_Shdr *Sec) const {
    return getSectionContentsAsArray<Elf_Relr>(Sec);
  }

  Expected<std::vector<Elf_Rela>> decode_relrs(Elf_Relr_Range relrs) const;

  Expected<std::vector<Elf_Rela>> android_relas(const Elf_Shdr *Sec) const;

  /// Iterate over program header table.
  Expected<Elf_Phdr_Range> program_headers() const {
    if (getHeader()->e_phnum && getHeader()->e_phentsize != sizeof(Elf_Phdr))
      return createError("invalid e_phentsize: " +
                         Twine(getHeader()->e_phentsize));
    if (getHeader()->e_phoff +
            (getHeader()->e_phnum * getHeader()->e_phentsize) >
        getBufSize())
      return createError("program headers are longer than binary of size " +
                         Twine(getBufSize()) + ": e_phoff = 0x" +
                         Twine::utohexstr(getHeader()->e_phoff) +
                         ", e_phnum = " + Twine(getHeader()->e_phnum) +
                         ", e_phentsize = " + Twine(getHeader()->e_phentsize));
    auto *Begin =
        reinterpret_cast<const Elf_Phdr *>(base() + getHeader()->e_phoff);
    return makeArrayRef(Begin, Begin + getHeader()->e_phnum);
  }

  /// Get an iterator over notes in a program header.
  ///
  /// The program header must be of type \c PT_NOTE.
  ///
  /// \param Phdr the program header to iterate over.
  /// \param Err [out] an error to support fallible iteration, which should
  ///  be checked after iteration ends.
  Elf_Note_Iterator notes_begin(const Elf_Phdr &Phdr, Error &Err) const {
    assert(Phdr.p_type == ELF::PT_NOTE && "Phdr is not of type PT_NOTE");
    ErrorAsOutParameter ErrAsOutParam(&Err);
    if (Phdr.p_offset + Phdr.p_filesz > getBufSize()) {
      Err = createError("PT_NOTE header has invalid offset (0x" +
                        Twine::utohexstr(Phdr.p_offset) + ") or size (0x" +
                        Twine::utohexstr(Phdr.p_filesz) + ")");
      return Elf_Note_Iterator(Err);
    }
    return Elf_Note_Iterator(base() + Phdr.p_offset, Phdr.p_filesz, Err);
  }

  /// Get an iterator over notes in a section.
  ///
  /// The section must be of type \c SHT_NOTE.
  ///
  /// \param Shdr the section to iterate over.
  /// \param Err [out] an error to support fallible iteration, which should
  ///  be checked after iteration ends.
  Elf_Note_Iterator notes_begin(const Elf_Shdr &Shdr, Error &Err) const {
    assert(Shdr.sh_type == ELF::SHT_NOTE && "Shdr is not of type SHT_NOTE");
    ErrorAsOutParameter ErrAsOutParam(&Err);
    if (Shdr.sh_offset + Shdr.sh_size > getBufSize()) {
      Err = createError("SHT_NOTE section " + getSecIndexForError(this, &Shdr) +
                        " has invalid offset (0x" +
                        Twine::utohexstr(Shdr.sh_offset) + ") or size (0x" +
                        Twine::utohexstr(Shdr.sh_size) + ")");
      return Elf_Note_Iterator(Err);
    }
    return Elf_Note_Iterator(base() + Shdr.sh_offset, Shdr.sh_size, Err);
  }

  /// Get the end iterator for notes.
  Elf_Note_Iterator notes_end() const {
    return Elf_Note_Iterator();
  }

  /// Get an iterator range over notes of a program header.
  ///
  /// The program header must be of type \c PT_NOTE.
  ///
  /// \param Phdr the program header to iterate over.
  /// \param Err [out] an error to support fallible iteration, which should
  ///  be checked after iteration ends.
  iterator_range<Elf_Note_Iterator> notes(const Elf_Phdr &Phdr,
                                          Error &Err) const {
    return make_range(notes_begin(Phdr, Err), notes_end());
  }

  /// Get an iterator range over notes of a section.
  ///
  /// The section must be of type \c SHT_NOTE.
  ///
  /// \param Shdr the section to iterate over.
  /// \param Err [out] an error to support fallible iteration, which should
  ///  be checked after iteration ends.
  iterator_range<Elf_Note_Iterator> notes(const Elf_Shdr &Shdr,
                                          Error &Err) const {
    return make_range(notes_begin(Shdr, Err), notes_end());
  }

  Expected<StringRef> getSectionStringTable(
      Elf_Shdr_Range Sections,
      WarningHandler WarnHandler = &defaultWarningHandler) const;
  Expected<uint32_t> getSectionIndex(const Elf_Sym *Sym, Elf_Sym_Range Syms,
                                     ArrayRef<Elf_Word> ShndxTable) const;
  Expected<const Elf_Shdr *> getSection(const Elf_Sym *Sym,
                                        const Elf_Shdr *SymTab,
                                        ArrayRef<Elf_Word> ShndxTable) const;
  Expected<const Elf_Shdr *> getSection(const Elf_Sym *Sym,
                                        Elf_Sym_Range Symtab,
                                        ArrayRef<Elf_Word> ShndxTable) const;
  Expected<const Elf_Shdr *> getSection(uint32_t Index) const;

  Expected<const Elf_Sym *> getSymbol(const Elf_Shdr *Sec,
                                      uint32_t Index) const;

  Expected<StringRef>
  getSectionName(const Elf_Shdr *Section,
                 WarningHandler WarnHandler = &defaultWarningHandler) const;
  Expected<StringRef> getSectionName(const Elf_Shdr *Section,
                                     StringRef DotShstrtab) const;
  template <typename T>
  Expected<ArrayRef<T>> getSectionContentsAsArray(const Elf_Shdr *Sec) const;
  Expected<ArrayRef<uint8_t>> getSectionContents(const Elf_Shdr *Sec) const;
};

using ELF32LEFile = ELFFile<ELF32LE>;
using ELF64LEFile = ELFFile<ELF64LE>;
using ELF32BEFile = ELFFile<ELF32BE>;
using ELF64BEFile = ELFFile<ELF64BE>;

template <class ELFT>
inline Expected<const typename ELFT::Shdr *>
getSection(typename ELFT::ShdrRange Sections, uint32_t Index) {
  if (Index >= Sections.size())
    return createError("invalid section index: " + Twine(Index));
  return &Sections[Index];
}

template <class ELFT>
inline Expected<uint32_t>
getExtendedSymbolTableIndex(const typename ELFT::Sym *Sym,
                            const typename ELFT::Sym *FirstSym,
                            ArrayRef<typename ELFT::Word> ShndxTable) {
  assert(Sym->st_shndx == ELF::SHN_XINDEX);
  unsigned Index = Sym - FirstSym;
  if (Index >= ShndxTable.size())
    return createError(
        "extended symbol index (" + Twine(Index) +
        ") is past the end of the SHT_SYMTAB_SHNDX section of size " +
        Twine(ShndxTable.size()));

  // The size of the table was checked in getSHNDXTable.
  return ShndxTable[Index];
}

template <class ELFT>
Expected<uint32_t>
ELFFile<ELFT>::getSectionIndex(const Elf_Sym *Sym, Elf_Sym_Range Syms,
                               ArrayRef<Elf_Word> ShndxTable) const {
  uint32_t Index = Sym->st_shndx;
  if (Index == ELF::SHN_XINDEX) {
    auto ErrorOrIndex = getExtendedSymbolTableIndex<ELFT>(
        Sym, Syms.begin(), ShndxTable);
    if (!ErrorOrIndex)
      return ErrorOrIndex.takeError();
    return *ErrorOrIndex;
  }
  if (Index == ELF::SHN_UNDEF || Index >= ELF::SHN_LORESERVE)
    return 0;
  return Index;
}

template <class ELFT>
Expected<const typename ELFT::Shdr *>
ELFFile<ELFT>::getSection(const Elf_Sym *Sym, const Elf_Shdr *SymTab,
                          ArrayRef<Elf_Word> ShndxTable) const {
  auto SymsOrErr = symbols(SymTab);
  if (!SymsOrErr)
    return SymsOrErr.takeError();
  return getSection(Sym, *SymsOrErr, ShndxTable);
}

template <class ELFT>
Expected<const typename ELFT::Shdr *>
ELFFile<ELFT>::getSection(const Elf_Sym *Sym, Elf_Sym_Range Symbols,
                          ArrayRef<Elf_Word> ShndxTable) const {
  auto IndexOrErr = getSectionIndex(Sym, Symbols, ShndxTable);
  if (!IndexOrErr)
    return IndexOrErr.takeError();
  uint32_t Index = *IndexOrErr;
  if (Index == 0)
    return nullptr;
  return getSection(Index);
}

template <class ELFT>
Expected<const typename ELFT::Sym *>
ELFFile<ELFT>::getSymbol(const Elf_Shdr *Sec, uint32_t Index) const {
  auto SymsOrErr = symbols(Sec);
  if (!SymsOrErr)
    return SymsOrErr.takeError();

  Elf_Sym_Range Symbols = *SymsOrErr;
  if (Index >= Symbols.size())
    return createError("unable to get symbol from section " +
                       getSecIndexForError(this, Sec) +
                       ": invalid symbol index (" + Twine(Index) + ")");
  return &Symbols[Index];
}

template <class ELFT>
template <typename T>
Expected<ArrayRef<T>>
ELFFile<ELFT>::getSectionContentsAsArray(const Elf_Shdr *Sec) const {
  if (Sec->sh_entsize != sizeof(T) && sizeof(T) != 1)
    return createError("section " + getSecIndexForError(this, Sec) +
                       " has an invalid sh_entsize: " + Twine(Sec->sh_entsize));

  uintX_t Offset = Sec->sh_offset;
  uintX_t Size = Sec->sh_size;

  if (Size % sizeof(T))
    return createError("section " + getSecIndexForError(this, Sec) +
                       " has an invalid sh_size (" + Twine(Size) +
                       ") which is not a multiple of its sh_entsize (" +
                       Twine(Sec->sh_entsize) + ")");
  if (std::numeric_limits<uintX_t>::max() - Offset < Size)
    return createError("section " + getSecIndexForError(this, Sec) +
                       " has a sh_offset (0x" + Twine::utohexstr(Offset) +
                       ") + sh_size (0x" + Twine::utohexstr(Size) +
                       ") that cannot be represented");
  if (Offset + Size > Buf.size())
    return createError("section " + getSecIndexForError(this, Sec) +
                       " has a sh_offset (0x" + Twine::utohexstr(Offset) +
                       ") + sh_size (0x" + Twine::utohexstr(Size) +
                       ") that is greater than the file size (0x" +
                       Twine::utohexstr(Buf.size()) + ")");

  if (Offset % alignof(T))
    // TODO: this error is untested.
    return createError("unaligned data");

  const T *Start = reinterpret_cast<const T *>(base() + Offset);
  return makeArrayRef(Start, Size / sizeof(T));
}

template <class ELFT>
Expected<ArrayRef<uint8_t>>
ELFFile<ELFT>::getSectionContents(const Elf_Shdr *Sec) const {
  return getSectionContentsAsArray<uint8_t>(Sec);
}

template <class ELFT>
StringRef ELFFile<ELFT>::getRelocationTypeName(uint32_t Type) const {
  return getELFRelocationTypeName(getHeader()->e_machine, Type);
}

template <class ELFT>
void ELFFile<ELFT>::getRelocationTypeName(uint32_t Type,
                                          SmallVectorImpl<char> &Result) const {
  if (!isMipsELF64()) {
    StringRef Name = getRelocationTypeName(Type);
    Result.append(Name.begin(), Name.end());
  } else {
    // The Mips N64 ABI allows up to three operations to be specified per
    // relocation record. Unfortunately there's no easy way to test for the
    // presence of N64 ELFs as they have no special flag that identifies them
    // as being N64. We can safely assume at the moment that all Mips
    // ELFCLASS64 ELFs are N64. New Mips64 ABIs should provide enough
    // information to disambiguate between old vs new ABIs.
    uint8_t Type1 = (Type >> 0) & 0xFF;
    uint8_t Type2 = (Type >> 8) & 0xFF;
    uint8_t Type3 = (Type >> 16) & 0xFF;

    // Concat all three relocation type names.
    StringRef Name = getRelocationTypeName(Type1);
    Result.append(Name.begin(), Name.end());

    Name = getRelocationTypeName(Type2);
    Result.append(1, '/');
    Result.append(Name.begin(), Name.end());

    Name = getRelocationTypeName(Type3);
    Result.append(1, '/');
    Result.append(Name.begin(), Name.end());
  }
}

template <class ELFT>
uint32_t ELFFile<ELFT>::getRelativeRelocationType() const {
  return getELFRelativeRelocationType(getHeader()->e_machine);
}

template <class ELFT>
Expected<const typename ELFT::Sym *>
ELFFile<ELFT>::getRelocationSymbol(const Elf_Rel *Rel,
                                   const Elf_Shdr *SymTab) const {
  uint32_t Index = Rel->getSymbol(isMips64EL());
  if (Index == 0)
    return nullptr;
  return getEntry<Elf_Sym>(SymTab, Index);
}

template <class ELFT>
Expected<StringRef>
ELFFile<ELFT>::getSectionStringTable(Elf_Shdr_Range Sections,
                                     WarningHandler WarnHandler) const {
  uint32_t Index = getHeader()->e_shstrndx;
  if (Index == ELF::SHN_XINDEX)
    Index = Sections[0].sh_link;

  if (!Index) // no section string table.
    return "";
  if (Index >= Sections.size())
    return createError("section header string table index " + Twine(Index) +
                       " does not exist");
  return getStringTable(&Sections[Index], WarnHandler);
}

template <class ELFT> ELFFile<ELFT>::ELFFile(StringRef Object) : Buf(Object) {}

template <class ELFT>
Expected<ELFFile<ELFT>> ELFFile<ELFT>::create(StringRef Object) {
  if (sizeof(Elf_Ehdr) > Object.size())
    return createError("invalid buffer: the size (" + Twine(Object.size()) +
                       ") is smaller than an ELF header (" +
                       Twine(sizeof(Elf_Ehdr)) + ")");
  return ELFFile(Object);
}

template <class ELFT>
Expected<typename ELFT::ShdrRange> ELFFile<ELFT>::sections() const {
  const uintX_t SectionTableOffset = getHeader()->e_shoff;
  if (SectionTableOffset == 0)
    return ArrayRef<Elf_Shdr>();

  if (getHeader()->e_shentsize != sizeof(Elf_Shdr))
    return createError("invalid e_shentsize in ELF header: " +
                       Twine(getHeader()->e_shentsize));

  const uint64_t FileSize = Buf.size();
  if (SectionTableOffset + sizeof(Elf_Shdr) > FileSize ||
      SectionTableOffset + (uintX_t)sizeof(Elf_Shdr) < SectionTableOffset)
    return createError(
        "section header table goes past the end of the file: e_shoff = 0x" +
        Twine::utohexstr(SectionTableOffset));

  // Invalid address alignment of section headers
  if (SectionTableOffset & (alignof(Elf_Shdr) - 1))
    // TODO: this error is untested.
    return createError("invalid alignment of section headers");

  const Elf_Shdr *First =
      reinterpret_cast<const Elf_Shdr *>(base() + SectionTableOffset);

  uintX_t NumSections = getHeader()->e_shnum;
  if (NumSections == 0)
    NumSections = First->sh_size;

  if (NumSections > UINT64_MAX / sizeof(Elf_Shdr))
    return createError("invalid number of sections specified in the NULL "
                       "section's sh_size field (" +
                       Twine(NumSections) + ")");

  const uint64_t SectionTableSize = NumSections * sizeof(Elf_Shdr);
  if (SectionTableOffset + SectionTableSize < SectionTableOffset)
    return createError(
        "invalid section header table offset (e_shoff = 0x" +
        Twine::utohexstr(SectionTableOffset) +
        ") or invalid number of sections specified in the first section "
        "header's sh_size field (0x" +
        Twine::utohexstr(NumSections) + ")");

  // Section table goes past end of file!
  if (SectionTableOffset + SectionTableSize > FileSize)
    return createError("section table goes past the end of file");
  return makeArrayRef(First, NumSections);
}

template <class ELFT>
template <typename T>
Expected<const T *> ELFFile<ELFT>::getEntry(uint32_t Section,
                                            uint32_t Entry) const {
  auto SecOrErr = getSection(Section);
  if (!SecOrErr)
    return SecOrErr.takeError();
  return getEntry<T>(*SecOrErr, Entry);
}

template <class ELFT>
template <typename T>
Expected<const T *> ELFFile<ELFT>::getEntry(const Elf_Shdr *Section,
                                            uint32_t Entry) const {
  if (sizeof(T) != Section->sh_entsize)
    return createError("section " + getSecIndexForError(this, Section) +
                       " has invalid sh_entsize: expected " + Twine(sizeof(T)) +
                       ", but got " + Twine(Section->sh_entsize));
  uint64_t Pos = Section->sh_offset + Entry * sizeof(T);
  if (Pos + sizeof(T) > Buf.size())
    return createError("unable to access section " +
                       getSecIndexForError(this, Section) + " data at 0x" +
                       Twine::utohexstr(Pos) +
                       ": offset goes past the end of file");
  return reinterpret_cast<const T *>(base() + Pos);
}

template <class ELFT>
Expected<const typename ELFT::Shdr *>
ELFFile<ELFT>::getSection(uint32_t Index) const {
  auto TableOrErr = sections();
  if (!TableOrErr)
    return TableOrErr.takeError();
  return object::getSection<ELFT>(*TableOrErr, Index);
}

template <class ELFT>
Expected<StringRef>
ELFFile<ELFT>::getStringTable(const Elf_Shdr *Section,
                              WarningHandler WarnHandler) const {
  if (Section->sh_type != ELF::SHT_STRTAB)
    if (Error E = WarnHandler("invalid sh_type for string table section " +
                              getSecIndexForError(this, Section) +
                              ": expected SHT_STRTAB, but got " +
                              object::getELFSectionTypeName(
                                  getHeader()->e_machine, Section->sh_type)))
      return std::move(E);

  auto V = getSectionContentsAsArray<char>(Section);
  if (!V)
    return V.takeError();
  ArrayRef<char> Data = *V;
  if (Data.empty())
    return createError("SHT_STRTAB string table section " +
                       getSecIndexForError(this, Section) + " is empty");
  if (Data.back() != '\0')
    return createError("SHT_STRTAB string table section " +
                       getSecIndexForError(this, Section) +
                       " is non-null terminated");
  return StringRef(Data.begin(), Data.size());
}

template <class ELFT>
Expected<ArrayRef<typename ELFT::Word>>
ELFFile<ELFT>::getSHNDXTable(const Elf_Shdr &Section) const {
  auto SectionsOrErr = sections();
  if (!SectionsOrErr)
    return SectionsOrErr.takeError();
  return getSHNDXTable(Section, *SectionsOrErr);
}

template <class ELFT>
Expected<ArrayRef<typename ELFT::Word>>
ELFFile<ELFT>::getSHNDXTable(const Elf_Shdr &Section,
                             Elf_Shdr_Range Sections) const {
  assert(Section.sh_type == ELF::SHT_SYMTAB_SHNDX);
  auto VOrErr = getSectionContentsAsArray<Elf_Word>(&Section);
  if (!VOrErr)
    return VOrErr.takeError();
  ArrayRef<Elf_Word> V = *VOrErr;
  auto SymTableOrErr = object::getSection<ELFT>(Sections, Section.sh_link);
  if (!SymTableOrErr)
    return SymTableOrErr.takeError();
  const Elf_Shdr &SymTable = **SymTableOrErr;
  if (SymTable.sh_type != ELF::SHT_SYMTAB &&
      SymTable.sh_type != ELF::SHT_DYNSYM)
    return createError("SHT_SYMTAB_SHNDX section is linked with " +
                       object::getELFSectionTypeName(getHeader()->e_machine,
                                                     SymTable.sh_type) +
                       " section (expected SHT_SYMTAB/SHT_DYNSYM)");

  uint64_t Syms = SymTable.sh_size / sizeof(Elf_Sym);
  if (V.size() != Syms)
    return createError("SHT_SYMTAB_SHNDX has " + Twine(V.size()) +
                       " entries, but the symbol table associated has " +
                       Twine(Syms));

  return V;
}

template <class ELFT>
Expected<StringRef>
ELFFile<ELFT>::getStringTableForSymtab(const Elf_Shdr &Sec) const {
  auto SectionsOrErr = sections();
  if (!SectionsOrErr)
    return SectionsOrErr.takeError();
  return getStringTableForSymtab(Sec, *SectionsOrErr);
}

template <class ELFT>
Expected<StringRef>
ELFFile<ELFT>::getStringTableForSymtab(const Elf_Shdr &Sec,
                                       Elf_Shdr_Range Sections) const {

  if (Sec.sh_type != ELF::SHT_SYMTAB && Sec.sh_type != ELF::SHT_DYNSYM)
    return createError(
        "invalid sh_type for symbol table, expected SHT_SYMTAB or SHT_DYNSYM");
  auto SectionOrErr = object::getSection<ELFT>(Sections, Sec.sh_link);
  if (!SectionOrErr)
    return SectionOrErr.takeError();
  return getStringTable(*SectionOrErr);
}

template <class ELFT>
Expected<StringRef>
ELFFile<ELFT>::getSectionName(const Elf_Shdr *Section,
                              WarningHandler WarnHandler) const {
  auto SectionsOrErr = sections();
  if (!SectionsOrErr)
    return SectionsOrErr.takeError();
  auto Table = getSectionStringTable(*SectionsOrErr, WarnHandler);
  if (!Table)
    return Table.takeError();
  return getSectionName(Section, *Table);
}

template <class ELFT>
Expected<StringRef> ELFFile<ELFT>::getSectionName(const Elf_Shdr *Section,
                                                  StringRef DotShstrtab) const {
  uint32_t Offset = Section->sh_name;
  if (Offset == 0)
    return StringRef();
  if (Offset >= DotShstrtab.size())
    return createError("a section " + getSecIndexForError(this, Section) +
                       " has an invalid sh_name (0x" +
                       Twine::utohexstr(Offset) +
                       ") offset which goes past the end of the "
                       "section name string table");
  return StringRef(DotShstrtab.data() + Offset);
}

/// This function returns the hash value for a symbol in the .dynsym section
/// Name of the API remains consistent as specified in the libelf
/// REF : http://www.sco.com/developers/gabi/latest/ch5.dynamic.html#hash
inline unsigned hashSysV(StringRef SymbolName) {
  unsigned h = 0, g;
  for (char C : SymbolName) {
    h = (h << 4) + C;
    g = h & 0xf0000000L;
    if (g != 0)
      h ^= g >> 24;
    h &= ~g;
  }
  return h;
}

} // end namespace object
} // end namespace llvm

#endif // LLVM_OBJECT_ELF_H
