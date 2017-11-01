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

static inline Error createError(StringRef Err) {
  return make_error<StringError>(Err, object_error::parse_failed);
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
  using Elf_Verdef = typename ELFT::Verdef;
  using Elf_Verdaux = typename ELFT::Verdaux;
  using Elf_Verneed = typename ELFT::Verneed;
  using Elf_Vernaux = typename ELFT::Vernaux;
  using Elf_Versym = typename ELFT::Versym;
  using Elf_Hash = typename ELFT::Hash;
  using Elf_GnuHash = typename ELFT::GnuHash;
  using Elf_Dyn_Range = typename ELFT::DynRange;
  using Elf_Shdr_Range = typename ELFT::ShdrRange;
  using Elf_Sym_Range = typename ELFT::SymRange;
  using Elf_Rel_Range = typename ELFT::RelRange;
  using Elf_Rela_Range = typename ELFT::RelaRange;
  using Elf_Phdr_Range = typename ELFT::PhdrRange;

  const uint8_t *base() const {
    return reinterpret_cast<const uint8_t *>(Buf.data());
  }

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

  Expected<StringRef> getStringTable(const Elf_Shdr *Section) const;
  Expected<StringRef> getStringTableForSymtab(const Elf_Shdr &Section) const;
  Expected<StringRef> getStringTableForSymtab(const Elf_Shdr &Section,
                                              Elf_Shdr_Range Sections) const;

  Expected<ArrayRef<Elf_Word>> getSHNDXTable(const Elf_Shdr &Section) const;
  Expected<ArrayRef<Elf_Word>> getSHNDXTable(const Elf_Shdr &Section,
                                             Elf_Shdr_Range Sections) const;

  StringRef getRelocationTypeName(uint32_t Type) const;
  void getRelocationTypeName(uint32_t Type,
                             SmallVectorImpl<char> &Result) const;

  /// \brief Get the symbol for a given relocation.
  Expected<const Elf_Sym *> getRelocationSymbol(const Elf_Rel *Rel,
                                                const Elf_Shdr *SymTab) const;

  static Expected<ELFFile> create(StringRef Object);

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

  Expected<std::vector<Elf_Rela>> android_relas(const Elf_Shdr *Sec) const;

  /// \brief Iterate over program header table.
  Expected<Elf_Phdr_Range> program_headers() const {
    if (getHeader()->e_phnum && getHeader()->e_phentsize != sizeof(Elf_Phdr))
      return createError("invalid e_phentsize");
    if (getHeader()->e_phoff +
            (getHeader()->e_phnum * getHeader()->e_phentsize) >
        getBufSize())
      return createError("program headers longer than binary");
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

using ELF32LEFile = ELFFile<ELFType<support::little, false>>;
using ELF64LEFile = ELFFile<ELFType<support::little, true>>;
using ELF32BEFile = ELFFile<ELFType<support::big, false>>;
using ELF64BEFile = ELFFile<ELFType<support::big, true>>;

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
inline Expected<const typename ELFT::Sym *>
getSymbol(typename ELFT::SymRange Symbols, uint32_t Index) {
  if (Index >= Symbols.size())
    return createError("invalid symbol index");
  return &Symbols[Index];
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
