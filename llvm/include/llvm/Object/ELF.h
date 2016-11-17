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

static inline Error createError(StringRef Err) {
  return make_error<StringError>(Err, object_error::parse_failed);
}

template <class ELFT>
class ELFFile {
public:
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Ehdr Elf_Ehdr;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::Dyn Elf_Dyn;
  typedef typename ELFT::Phdr Elf_Phdr;
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::Verdef Elf_Verdef;
  typedef typename ELFT::Verdaux Elf_Verdaux;
  typedef typename ELFT::Verneed Elf_Verneed;
  typedef typename ELFT::Vernaux Elf_Vernaux;
  typedef typename ELFT::Versym Elf_Versym;
  typedef typename ELFT::Hash Elf_Hash;
  typedef typename ELFT::GnuHash Elf_GnuHash;
  typedef typename ELFT::DynRange Elf_Dyn_Range;
  typedef typename ELFT::ShdrRange Elf_Shdr_Range;
  typedef typename ELFT::SymRange Elf_Sym_Range;
  typedef typename ELFT::RelRange Elf_Rel_Range;
  typedef typename ELFT::RelaRange Elf_Rela_Range;
  typedef typename ELFT::PhdrRange Elf_Phdr_Range;

  const uint8_t *base() const {
    return reinterpret_cast<const uint8_t *>(Buf.data());
  }

  size_t getBufSize() const { return Buf.size(); }

private:

  StringRef Buf;

public:
  const Elf_Ehdr *getHeader() const {
    return reinterpret_cast<const Elf_Ehdr *>(base());
  }

  template <typename T>
  Expected<const T *> getEntry(uint32_t Section, uint32_t Entry) const;
  template <typename T>
  Expected<const T *> getEntry(const Elf_Shdr *Section, uint32_t Entry) const;

  Expected<StringRef> getStringTable(const Elf_Shdr *Section) const;
  Expected<StringRef> getStringTableForSymtab(const Elf_Shdr &Section) const;
  Expected<StringRef> getStringTableForSymtab(const Elf_Shdr &Section,
                                              Elf_Shdr_Range Sections) const;

  Expected<ArrayRef<Elf_Word>> getSHNDXTable(const Elf_Shdr &Section) const;
  Expected<ArrayRef<Elf_Word>> getSHNDXTable(const Elf_Shdr &Section,
                                             Elf_Shdr_Range Sections) const;

  void VerifyStrTab(const Elf_Shdr *sh) const;

  StringRef getRelocationTypeName(uint32_t Type) const;
  void getRelocationTypeName(uint32_t Type,
                             SmallVectorImpl<char> &Result) const;

  /// \brief Get the symbol for a given relocation.
  Expected<const Elf_Sym *> getRelocationSymbol(const Elf_Rel *Rel,
                                                const Elf_Shdr *SymTab) const;

  ELFFile(StringRef Object);

  bool isMipsELF64() const {
    return getHeader()->e_machine == ELF::EM_MIPS &&
           getHeader()->getFileClass() == ELF::ELFCLASS64;
  }

  bool isMips64EL() const {
    return isMipsELF64() &&
           getHeader()->getDataEncoding() == ELF::ELFDATA2LSB;
  }

  Expected<Elf_Shdr_Range> sections() const;

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

  /// \brief Iterate over program header table.
  Expected<Elf_Phdr_Range> program_headers() const {
    if (getHeader()->e_phnum && getHeader()->e_phentsize != sizeof(Elf_Phdr))
      return createError("invalid e_phentsize");
    auto *Begin =
        reinterpret_cast<const Elf_Phdr *>(base() + getHeader()->e_phoff);
    return makeArrayRef(Begin, Begin + getHeader()->e_phnum);
  }

  Expected<StringRef> getSectionStringTable(Elf_Shdr_Range Sections) const;
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

  Expected<StringRef> getSectionName(const Elf_Shdr *Section) const;
  Expected<StringRef> getSectionName(const Elf_Shdr *Section,
                                     StringRef DotShstrtab) const;
  template <typename T>
  Expected<ArrayRef<T>> getSectionContentsAsArray(const Elf_Shdr *Sec) const;
  Expected<ArrayRef<uint8_t>> getSectionContents(const Elf_Shdr *Sec) const;
};

typedef ELFFile<ELFType<support::little, false>> ELF32LEFile;
typedef ELFFile<ELFType<support::little, true>> ELF64LEFile;
typedef ELFFile<ELFType<support::big, false>> ELF32BEFile;
typedef ELFFile<ELFType<support::big, true>> ELF64BEFile;

template <class ELFT>
inline Expected<const typename ELFT::Shdr *>
getSection(typename ELFT::ShdrRange Sections, uint32_t Index) {
  if (Index >= Sections.size())
    return createError("invalid section index");
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
    return createError("index past the end of the symbol table");

  // The size of the table was checked in getSHNDXTable.
  return ShndxTable[Index];
}

template <class ELFT>
Expected<uint32_t>
ELFFile<ELFT>::getSectionIndex(const Elf_Sym *Sym, Elf_Sym_Range Syms,
                               ArrayRef<Elf_Word> ShndxTable) const {
  uint32_t Index = Sym->st_shndx;
  if (Index == ELF::SHN_XINDEX) {
    auto ErrorOrIndex = object::getExtendedSymbolTableIndex<ELFT>(
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
  auto SectionsOrErr = sections();
  if (!SectionsOrErr)
    return SectionsOrErr.takeError();
  return object::getSection<ELFT>(*SectionsOrErr, Index);
}

template <class ELFT>
inline Expected<const typename ELFT::Sym *>
getSymbol(typename ELFT::SymRange Symbols, uint32_t Index) {
  if (Index >= Symbols.size())
    return createError("invalid symbol index");
  return &Symbols[Index];
}

template <class ELFT>
Expected<const typename ELFT::Sym *>
ELFFile<ELFT>::getSymbol(const Elf_Shdr *Sec, uint32_t Index) const {
  auto SymtabOrErr = symbols(Sec);
  if (!SymtabOrErr)
    return SymtabOrErr.takeError();
  return object::getSymbol<ELFT>(*SymtabOrErr, Index);
}

template <class ELFT>
template <typename T>
Expected<ArrayRef<T>>
ELFFile<ELFT>::getSectionContentsAsArray(const Elf_Shdr *Sec) const {
  if (Sec->sh_entsize != sizeof(T) && sizeof(T) != 1)
    return createError("invalid sh_entsize");

  uintX_t Offset = Sec->sh_offset;
  uintX_t Size = Sec->sh_size;

  if (Size % sizeof(T))
    return createError("size is not a multiple of sh_entsize");
  if ((std::numeric_limits<uintX_t>::max() - Offset < Size) ||
      Offset + Size > Buf.size())
    return createError("invalid section offset");

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
ELFFile<ELFT>::getSectionStringTable(Elf_Shdr_Range Sections) const {
  uint32_t Index = getHeader()->e_shstrndx;
  if (Index == ELF::SHN_XINDEX)
    Index = Sections[0].sh_link;

  if (!Index) // no section string table.
    return "";
  if (Index >= Sections.size())
    return createError("invalid section index");
  return getStringTable(&Sections[Index]);
}

template <class ELFT>
ELFFile<ELFT>::ELFFile(StringRef Object) : Buf(Object) {
  assert(sizeof(Elf_Ehdr) <= Buf.size() && "Invalid buffer");
}

template <class ELFT>
static bool compareAddr(uint64_t VAddr, const Elf_Phdr_Impl<ELFT> *Phdr) {
  return VAddr < Phdr->p_vaddr;
}

template <class ELFT>
Expected<typename ELFT::ShdrRange> ELFFile<ELFT>::sections() const {
  const uintX_t SectionTableOffset = getHeader()->e_shoff;
  if (SectionTableOffset == 0)
    return ArrayRef<Elf_Shdr>();

  if (getHeader()->e_shentsize != sizeof(Elf_Shdr))
    return createError(
        "invalid section header entry size (e_shentsize) in ELF header");

  const uint64_t FileSize = Buf.size();

  if (SectionTableOffset + sizeof(Elf_Shdr) > FileSize)
    return createError("section header table goes past the end of the file");

  // Invalid address alignment of section headers
  if (SectionTableOffset & (alignof(Elf_Shdr) - 1))
    return createError("invalid alignment of section headers");

  const Elf_Shdr *First =
      reinterpret_cast<const Elf_Shdr *>(base() + SectionTableOffset);

  uintX_t NumSections = getHeader()->e_shnum;
  if (NumSections == 0)
    NumSections = First->sh_size;

  if (NumSections > UINT64_MAX / sizeof(Elf_Shdr))
    return createError("section table goes past the end of file");

  const uint64_t SectionTableSize = NumSections * sizeof(Elf_Shdr);

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
    return createError("invalid sh_entsize");
  size_t Pos = Section->sh_offset + Entry * sizeof(T);
  if (Pos + sizeof(T) > Buf.size())
    return createError("invalid section offset");
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
ELFFile<ELFT>::getStringTable(const Elf_Shdr *Section) const {
  if (Section->sh_type != ELF::SHT_STRTAB)
    return createError("invalid sh_type for string table, expected SHT_STRTAB");
  auto V = getSectionContentsAsArray<char>(Section);
  if (!V)
    return V.takeError();
  ArrayRef<char> Data = *V;
  if (Data.empty())
    return createError("empty string table");
  if (Data.back() != '\0')
    return createError("string table non-null terminated");
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
    return createError("invalid sh_type");
  if (V.size() != (SymTable.sh_size / sizeof(Elf_Sym)))
    return createError("invalid section contents size");
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
ELFFile<ELFT>::getSectionName(const Elf_Shdr *Section) const {
  auto SectionsOrErr = sections();
  if (!SectionsOrErr)
    return SectionsOrErr.takeError();
  auto Table = getSectionStringTable(*SectionsOrErr);
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
    return createError("invalid string offset");
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

#endif
