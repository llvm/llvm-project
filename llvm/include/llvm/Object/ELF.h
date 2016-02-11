//===- ELF.h - ELF object file implementation -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ELFFile template class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ELF_H
#define LLVM_OBJECT_ELF_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace object {

StringRef getELFRelocationTypeName(uint32_t Machine, uint32_t Type);

// Subclasses of ELFFile may need this for template instantiation
inline std::pair<unsigned char, unsigned char>
getElfArchType(StringRef Object) {
  if (Object.size() < ELF::EI_NIDENT)
    return std::make_pair((uint8_t)ELF::ELFCLASSNONE,
                          (uint8_t)ELF::ELFDATANONE);
  return std::make_pair((uint8_t)Object[ELF::EI_CLASS],
                        (uint8_t)Object[ELF::EI_DATA]);
}

template <class ELFT>
class ELFFile {
public:
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  typedef typename std::conditional<ELFT::Is64Bits,
                                    uint64_t, uint32_t>::type uintX_t;

  typedef Elf_Ehdr_Impl<ELFT> Elf_Ehdr;
  typedef Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef Elf_Dyn_Impl<ELFT> Elf_Dyn;
  typedef Elf_Phdr_Impl<ELFT> Elf_Phdr;
  typedef Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef Elf_Rel_Impl<ELFT, true> Elf_Rela;
  typedef Elf_Verdef_Impl<ELFT> Elf_Verdef;
  typedef Elf_Verdaux_Impl<ELFT> Elf_Verdaux;
  typedef Elf_Verneed_Impl<ELFT> Elf_Verneed;
  typedef Elf_Vernaux_Impl<ELFT> Elf_Vernaux;
  typedef Elf_Versym_Impl<ELFT> Elf_Versym;
  typedef Elf_Hash_Impl<ELFT> Elf_Hash;
  typedef Elf_GnuHash_Impl<ELFT> Elf_GnuHash;
  typedef iterator_range<const Elf_Dyn *> Elf_Dyn_Range;
  typedef iterator_range<const Elf_Shdr *> Elf_Shdr_Range;
  typedef iterator_range<const Elf_Sym *> Elf_Sym_Range;

  const uint8_t *base() const {
    return reinterpret_cast<const uint8_t *>(Buf.data());
  }

  size_t getBufSize() const { return Buf.size(); }

private:

  StringRef Buf;

  const Elf_Ehdr *Header;
  const Elf_Shdr *SectionHeaderTable = nullptr;
  StringRef DotShstrtab;                    // Section header string table.

public:
  template<typename T>
  const T        *getEntry(uint32_t Section, uint32_t Entry) const;
  template <typename T>
  const T *getEntry(const Elf_Shdr *Section, uint32_t Entry) const;

  ErrorOr<StringRef> getStringTable(const Elf_Shdr *Section) const;
  ErrorOr<StringRef> getStringTableForSymtab(const Elf_Shdr &Section) const;

  ErrorOr<ArrayRef<Elf_Word>> getSHNDXTable(const Elf_Shdr &Section) const;

  void VerifyStrTab(const Elf_Shdr *sh) const;

  StringRef getRelocationTypeName(uint32_t Type) const;
  void getRelocationTypeName(uint32_t Type,
                             SmallVectorImpl<char> &Result) const;

  /// \brief Get the symbol for a given relocation.
  const Elf_Sym *getRelocationSymbol(const Elf_Rel *Rel,
                                     const Elf_Shdr *SymTab) const;

  ELFFile(StringRef Object, std::error_code &EC);

  bool isMipsELF64() const {
    return Header->e_machine == ELF::EM_MIPS &&
      Header->getFileClass() == ELF::ELFCLASS64;
  }

  bool isMips64EL() const {
    return Header->e_machine == ELF::EM_MIPS &&
      Header->getFileClass() == ELF::ELFCLASS64 &&
      Header->getDataEncoding() == ELF::ELFDATA2LSB;
  }

  ErrorOr<const Elf_Dyn *> dynamic_table_begin(const Elf_Phdr *Phdr) const;
  ErrorOr<const Elf_Dyn *> dynamic_table_end(const Elf_Phdr *Phdr) const;
  ErrorOr<Elf_Dyn_Range> dynamic_table(const Elf_Phdr *Phdr) const {
    ErrorOr<const Elf_Dyn *> Begin = dynamic_table_begin(Phdr);
    if (std::error_code EC = Begin.getError())
      return EC;
    ErrorOr<const Elf_Dyn *> End = dynamic_table_end(Phdr);
    if (std::error_code EC = End.getError())
      return EC;
    return make_range(*Begin, *End);
  }

  const Elf_Shdr *section_begin() const;
  const Elf_Shdr *section_end() const;
  Elf_Shdr_Range sections() const {
    return make_range(section_begin(), section_end());
  }

  const Elf_Sym *symbol_begin(const Elf_Shdr *Sec) const {
    if (!Sec)
      return nullptr;
    if (Sec->sh_entsize != sizeof(Elf_Sym))
      report_fatal_error("Invalid symbol size");
    return reinterpret_cast<const Elf_Sym *>(base() + Sec->sh_offset);
  }
  const Elf_Sym *symbol_end(const Elf_Shdr *Sec) const {
    if (!Sec)
      return nullptr;
    uint64_t Size = Sec->sh_size;
    if (Size % sizeof(Elf_Sym))
      report_fatal_error("Invalid symbol table size");
    return symbol_begin(Sec) + Size / sizeof(Elf_Sym);
  }
  Elf_Sym_Range symbols(const Elf_Shdr *Sec) const {
    return make_range(symbol_begin(Sec), symbol_end(Sec));
  }

  typedef iterator_range<const Elf_Rela *> Elf_Rela_Range;

  const Elf_Rela *rela_begin(const Elf_Shdr *sec) const {
    if (sec->sh_entsize != sizeof(Elf_Rela))
      report_fatal_error("Invalid relocation entry size");
    return reinterpret_cast<const Elf_Rela *>(base() + sec->sh_offset);
  }

  const Elf_Rela *rela_end(const Elf_Shdr *sec) const {
    uint64_t Size = sec->sh_size;
    if (Size % sizeof(Elf_Rela))
      report_fatal_error("Invalid relocation table size");
    return rela_begin(sec) + Size / sizeof(Elf_Rela);
  }

  Elf_Rela_Range relas(const Elf_Shdr *Sec) const {
    return make_range(rela_begin(Sec), rela_end(Sec));
  }

  const Elf_Rel *rel_begin(const Elf_Shdr *sec) const {
    if (sec->sh_entsize != sizeof(Elf_Rel))
      report_fatal_error("Invalid relocation entry size");
    return reinterpret_cast<const Elf_Rel *>(base() + sec->sh_offset);
  }

  const Elf_Rel *rel_end(const Elf_Shdr *sec) const {
    uint64_t Size = sec->sh_size;
    if (Size % sizeof(Elf_Rel))
      report_fatal_error("Invalid relocation table size");
    return rel_begin(sec) + Size / sizeof(Elf_Rel);
  }

  typedef iterator_range<const Elf_Rel *> Elf_Rel_Range;
  Elf_Rel_Range rels(const Elf_Shdr *Sec) const {
    return make_range(rel_begin(Sec), rel_end(Sec));
  }

  /// \brief Iterate over program header table.
  const Elf_Phdr *program_header_begin() const {
    if (Header->e_phnum && Header->e_phentsize != sizeof(Elf_Phdr))
      report_fatal_error("Invalid program header size");
    return reinterpret_cast<const Elf_Phdr *>(base() + Header->e_phoff);
  }

  const Elf_Phdr *program_header_end() const {
    return program_header_begin() + Header->e_phnum;
  }

  typedef iterator_range<const Elf_Phdr *> Elf_Phdr_Range;

  const Elf_Phdr_Range program_headers() const {
    return make_range(program_header_begin(), program_header_end());
  }

  uint64_t getNumSections() const;
  uintX_t getStringTableIndex() const;
  uint32_t getExtendedSymbolTableIndex(const Elf_Sym *Sym,
                                       const Elf_Shdr *SymTab,
                                       ArrayRef<Elf_Word> ShndxTable) const;
  uint32_t getExtendedSymbolTableIndex(const Elf_Sym *Sym,
                                       const Elf_Sym *FirstSym,
                                       ArrayRef<Elf_Word> ShndxTable) const;
  const Elf_Ehdr *getHeader() const { return Header; }
  ErrorOr<const Elf_Shdr *> getSection(const Elf_Sym *Sym,
                                       const Elf_Shdr *SymTab,
                                       ArrayRef<Elf_Word> ShndxTable) const;
  ErrorOr<const Elf_Shdr *> getSection(uint32_t Index) const;

  const Elf_Sym *getSymbol(const Elf_Shdr *Sec, uint32_t Index) const {
    return &*(symbol_begin(Sec) + Index);
  }

  ErrorOr<StringRef> getSectionName(const Elf_Shdr *Section) const;
  template <typename T>
  ErrorOr<ArrayRef<T>> getSectionContentsAsArray(const Elf_Shdr *Sec) const;
  ErrorOr<ArrayRef<uint8_t> > getSectionContents(const Elf_Shdr *Sec) const;
};

typedef ELFFile<ELFType<support::little, false>> ELF32LEFile;
typedef ELFFile<ELFType<support::little, true>> ELF64LEFile;
typedef ELFFile<ELFType<support::big, false>> ELF32BEFile;
typedef ELFFile<ELFType<support::big, true>> ELF64BEFile;

template <class ELFT>
uint32_t ELFFile<ELFT>::getExtendedSymbolTableIndex(
    const Elf_Sym *Sym, const Elf_Shdr *SymTab,
    ArrayRef<Elf_Word> ShndxTable) const {
  return getExtendedSymbolTableIndex(Sym, symbol_begin(SymTab), ShndxTable);
}

template <class ELFT>
uint32_t ELFFile<ELFT>::getExtendedSymbolTableIndex(
    const Elf_Sym *Sym, const Elf_Sym *FirstSym,
    ArrayRef<Elf_Word> ShndxTable) const {
  assert(Sym->st_shndx == ELF::SHN_XINDEX);
  unsigned Index = Sym - FirstSym;

  // The size of the table was checked in getSHNDXTable.
  return ShndxTable[Index];
}

template <class ELFT>
ErrorOr<const typename ELFFile<ELFT>::Elf_Shdr *>
ELFFile<ELFT>::getSection(const Elf_Sym *Sym, const Elf_Shdr *SymTab,
                          ArrayRef<Elf_Word> ShndxTable) const {
  uint32_t Index = Sym->st_shndx;
  if (Index == ELF::SHN_XINDEX)
    return getSection(
        getExtendedSymbolTableIndex(Sym, symbol_begin(SymTab), ShndxTable));

  if (Index == ELF::SHN_UNDEF || Index >= ELF::SHN_LORESERVE)
    return nullptr;
  return getSection(Sym->st_shndx);
}

template <class ELFT>
template <typename T>
ErrorOr<ArrayRef<T>>
ELFFile<ELFT>::getSectionContentsAsArray(const Elf_Shdr *Sec) const {
  uintX_t Offset = Sec->sh_offset;
  uintX_t Size = Sec->sh_size;

  if (Size % sizeof(T))
    return object_error::parse_failed;
  if (Offset + Size > Buf.size())
    return object_error::parse_failed;

  const T *Start = reinterpret_cast<const T *>(base() + Offset);
  return makeArrayRef(Start, Size / sizeof(T));
}

template <class ELFT>
ErrorOr<ArrayRef<uint8_t>>
ELFFile<ELFT>::getSectionContents(const Elf_Shdr *Sec) const {
  return getSectionContentsAsArray<uint8_t>(Sec);
}

template <class ELFT>
StringRef ELFFile<ELFT>::getRelocationTypeName(uint32_t Type) const {
  return getELFRelocationTypeName(Header->e_machine, Type);
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
const typename ELFFile<ELFT>::Elf_Sym *
ELFFile<ELFT>::getRelocationSymbol(const Elf_Rel *Rel,
                                   const Elf_Shdr *SymTab) const {
  uint32_t Index = Rel->getSymbol(isMips64EL());
  if (Index == 0)
    return nullptr;
  return getEntry<Elf_Sym>(SymTab, Index);
}

template <class ELFT>
uint64_t ELFFile<ELFT>::getNumSections() const {
  assert(Header && "Header not initialized!");
  if (Header->e_shnum == ELF::SHN_UNDEF && Header->e_shoff > 0) {
    assert(SectionHeaderTable && "SectionHeaderTable not initialized!");
    return SectionHeaderTable->sh_size;
  }
  return Header->e_shnum;
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t ELFFile<ELFT>::getStringTableIndex() const {
  if (Header->e_shnum == ELF::SHN_UNDEF) {
    if (Header->e_shstrndx == ELF::SHN_HIRESERVE)
      return SectionHeaderTable->sh_link;
    if (Header->e_shstrndx >= getNumSections())
      return 0;
  }
  return Header->e_shstrndx;
}

template <class ELFT>
ELFFile<ELFT>::ELFFile(StringRef Object, std::error_code &EC)
    : Buf(Object) {
  const uint64_t FileSize = Buf.size();

  if (sizeof(Elf_Ehdr) > FileSize) {
    // File too short!
    EC = object_error::parse_failed;
    return;
  }

  Header = reinterpret_cast<const Elf_Ehdr *>(base());

  if (Header->e_shoff == 0)
    return;

  const uint64_t SectionTableOffset = Header->e_shoff;

  if (SectionTableOffset + sizeof(Elf_Shdr) > FileSize) {
    // Section header table goes past end of file!
    EC = object_error::parse_failed;
    return;
  }

  // The getNumSections() call below depends on SectionHeaderTable being set.
  SectionHeaderTable =
    reinterpret_cast<const Elf_Shdr *>(base() + SectionTableOffset);
  const uint64_t SectionTableSize = getNumSections() * Header->e_shentsize;

  if (SectionTableOffset + SectionTableSize > FileSize) {
    // Section table goes past end of file!
    EC = object_error::parse_failed;
    return;
  }

  // Get string table sections.
  uintX_t StringTableIndex = getStringTableIndex();
  if (StringTableIndex) {
    ErrorOr<const Elf_Shdr *> StrTabSecOrErr = getSection(StringTableIndex);
    if ((EC = StrTabSecOrErr.getError()))
      return;

    ErrorOr<StringRef> StringTableOrErr = getStringTable(*StrTabSecOrErr);
    if ((EC = StringTableOrErr.getError()))
      return;
    DotShstrtab = *StringTableOrErr;
  }

  EC = std::error_code();
}

template <class ELFT>
static bool compareAddr(uint64_t VAddr, const Elf_Phdr_Impl<ELFT> *Phdr) {
  return VAddr < Phdr->p_vaddr;
}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Shdr *ELFFile<ELFT>::section_begin() const {
  if (Header->e_shentsize != sizeof(Elf_Shdr))
    report_fatal_error(
        "Invalid section header entry size (e_shentsize) in ELF header");
  return reinterpret_cast<const Elf_Shdr *>(base() + Header->e_shoff);
}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Shdr *ELFFile<ELFT>::section_end() const {
  return section_begin() + getNumSections();
}

template <class ELFT>
ErrorOr<const typename ELFFile<ELFT>::Elf_Dyn *>
ELFFile<ELFT>::dynamic_table_begin(const Elf_Phdr *Phdr) const {
  if (!Phdr)
    return nullptr;
  assert(Phdr->p_type == ELF::PT_DYNAMIC && "Got the wrong program header");
  uintX_t Offset = Phdr->p_offset;
  if (Offset > Buf.size())
    return object_error::parse_failed;
  return reinterpret_cast<const Elf_Dyn *>(base() + Offset);
}

template <class ELFT>
ErrorOr<const typename ELFFile<ELFT>::Elf_Dyn *>
ELFFile<ELFT>::dynamic_table_end(const Elf_Phdr *Phdr) const {
  if (!Phdr)
    return nullptr;
  assert(Phdr->p_type == ELF::PT_DYNAMIC && "Got the wrong program header");
  uintX_t Size = Phdr->p_filesz;
  if (Size % sizeof(Elf_Dyn))
    return object_error::elf_invalid_dynamic_table_size;
  // FIKME: Check for overflow?
  uintX_t End = Phdr->p_offset + Size;
  if (End > Buf.size())
    return object_error::parse_failed;
  return reinterpret_cast<const Elf_Dyn *>(base() + End);
}

template <class ELFT>
template <typename T>
const T *ELFFile<ELFT>::getEntry(uint32_t Section, uint32_t Entry) const {
  ErrorOr<const Elf_Shdr *> Sec = getSection(Section);
  if (std::error_code EC = Sec.getError())
    report_fatal_error(EC.message());
  return getEntry<T>(*Sec, Entry);
}

template <class ELFT>
template <typename T>
const T *ELFFile<ELFT>::getEntry(const Elf_Shdr *Section,
                                 uint32_t Entry) const {
  return reinterpret_cast<const T *>(base() + Section->sh_offset +
                                     (Entry * Section->sh_entsize));
}

template <class ELFT>
ErrorOr<const typename ELFFile<ELFT>::Elf_Shdr *>
ELFFile<ELFT>::getSection(uint32_t Index) const {
  assert(SectionHeaderTable && "SectionHeaderTable not initialized!");
  if (Index >= getNumSections())
    return object_error::invalid_section_index;

  return reinterpret_cast<const Elf_Shdr *>(
      reinterpret_cast<const char *>(SectionHeaderTable) +
      (Index * Header->e_shentsize));
}

template <class ELFT>
ErrorOr<StringRef>
ELFFile<ELFT>::getStringTable(const Elf_Shdr *Section) const {
  if (Section->sh_type != ELF::SHT_STRTAB)
    return object_error::parse_failed;
  uint64_t Offset = Section->sh_offset;
  uint64_t Size = Section->sh_size;
  if (Offset + Size > Buf.size())
    return object_error::parse_failed;
  StringRef Data((const char *)base() + Section->sh_offset, Size);
  if (Data[Size - 1] != '\0')
    return object_error::string_table_non_null_end;
  return Data;
}

template <class ELFT>
ErrorOr<ArrayRef<typename ELFFile<ELFT>::Elf_Word>>
ELFFile<ELFT>::getSHNDXTable(const Elf_Shdr &Section) const {
  assert(Section.sh_type == ELF::SHT_SYMTAB_SHNDX);
  const Elf_Word *ShndxTableBegin =
      reinterpret_cast<const Elf_Word *>(base() + Section.sh_offset);
  uintX_t Size = Section.sh_size;
  if (Size % sizeof(uint32_t))
    return object_error::parse_failed;
  uintX_t NumSymbols = Size / sizeof(uint32_t);
  const Elf_Word *ShndxTableEnd = ShndxTableBegin + NumSymbols;
  if (reinterpret_cast<const char *>(ShndxTableEnd) > Buf.end())
    return object_error::parse_failed;
  ErrorOr<const Elf_Shdr *> SymTableOrErr = getSection(Section.sh_link);
  if (std::error_code EC = SymTableOrErr.getError())
    return EC;
  const Elf_Shdr &SymTable = **SymTableOrErr;
  if (SymTable.sh_type != ELF::SHT_SYMTAB &&
      SymTable.sh_type != ELF::SHT_DYNSYM)
    return object_error::parse_failed;
  if (NumSymbols != (SymTable.sh_size / sizeof(Elf_Sym)))
    return object_error::parse_failed;
  return makeArrayRef(ShndxTableBegin, ShndxTableEnd);
}

template <class ELFT>
ErrorOr<StringRef>
ELFFile<ELFT>::getStringTableForSymtab(const Elf_Shdr &Sec) const {
  if (Sec.sh_type != ELF::SHT_SYMTAB && Sec.sh_type != ELF::SHT_DYNSYM)
    return object_error::parse_failed;
  ErrorOr<const Elf_Shdr *> SectionOrErr = getSection(Sec.sh_link);
  if (std::error_code EC = SectionOrErr.getError())
    return EC;
  return getStringTable(*SectionOrErr);
}

template <class ELFT>
ErrorOr<StringRef>
ELFFile<ELFT>::getSectionName(const Elf_Shdr *Section) const {
  uint32_t Offset = Section->sh_name;
  if (Offset == 0)
    return StringRef();
  if (Offset >= DotShstrtab.size())
    return object_error::parse_failed;
  return StringRef(DotShstrtab.data() + Offset);
}

/// This function returns the hash value for a symbol in the .dynsym section
/// Name of the API remains consistent as specified in the libelf
/// REF : http://www.sco.com/developers/gabi/latest/ch5.dynamic.html#hash
static inline unsigned elf_hash(StringRef &symbolName) {
  unsigned h = 0, g;
  for (unsigned i = 0, j = symbolName.size(); i < j; i++) {
    h = (h << 4) + symbolName[i];
    g = h & 0xf0000000L;
    if (g != 0)
      h ^= g >> 24;
    h &= ~g;
  }
  return h;
}
} // end namespace object
} // end namespace llvm

#endif
