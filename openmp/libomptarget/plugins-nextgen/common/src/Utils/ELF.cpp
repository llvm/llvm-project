//===-- Utils/ELF.cpp - Common ELF functionality --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common ELF functionality for target plugins.
//
//===----------------------------------------------------------------------===//

#include "Utils/ELF.h"

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

bool utils::elf::isELF(StringRef Buffer) {
  switch (identify_magic(Buffer)) {
  case file_magic::elf:
  case file_magic::elf_relocatable:
  case file_magic::elf_executable:
  case file_magic::elf_shared_object:
  case file_magic::elf_core:
    return true;
  default:
    return false;
  }
}

Expected<bool> utils::elf::checkMachine(StringRef Object, uint16_t EMachine) {
  assert(isELF(Object) && "Input is not an ELF!");

  Expected<ELF64LEObjectFile> ElfOrErr =
      ELF64LEObjectFile::create(MemoryBufferRef(Object, /*Identifier=*/""),
                                /*InitContent=*/false);
  if (!ElfOrErr)
    return ElfOrErr.takeError();

  const auto Header = ElfOrErr->getELFFile().getHeader();
  if (Header.e_ident[EI_CLASS] != ELFCLASS64)
    return createError("Only 64-bit ELF files are supported");
  if (Header.e_type != ET_EXEC && Header.e_type != ET_DYN)
    return createError("Only executable ELF files are supported");

  if (Header.e_machine == EM_AMDGPU) {
    if (Header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA)
      return createError("Invalid AMD OS/ABI, must be AMDGPU_HSA");
    if (Header.e_ident[EI_ABIVERSION] != ELFABIVERSION_AMDGPU_HSA_V4 &&
        Header.e_ident[EI_ABIVERSION] != ELFABIVERSION_AMDGPU_HSA_V5)
      return createError("Invalid AMD ABI version, must be version 4 or 5");
    if ((Header.e_flags & EF_AMDGPU_MACH) < EF_AMDGPU_MACH_AMDGCN_GFX700 ||
        (Header.e_flags & EF_AMDGPU_MACH) > EF_AMDGPU_MACH_AMDGCN_GFX1201)
      return createError("Unsupported AMDGPU architecture");
  } else if (Header.e_machine == EM_CUDA) {
    if (~Header.e_flags & EF_CUDA_64BIT_ADDRESS)
      return createError("Invalid CUDA addressing mode");
    if ((Header.e_flags & EF_CUDA_SM) < EF_CUDA_SM35 ||
        (Header.e_flags & EF_CUDA_SM) > EF_CUDA_SM90)
      return createError("Unsupported NVPTX architecture");
  }

  return Header.e_machine == EMachine;
}

template <class ELFT>
static Expected<const typename ELFT::Sym *>
getSymbolFromGnuHashTable(StringRef Name, const typename ELFT::GnuHash &HashTab,
                          ArrayRef<typename ELFT::Sym> SymTab,
                          StringRef StrTab) {
  const uint32_t NameHash = hashGnu(Name);
  const typename ELFT::Word NBucket = HashTab.nbuckets;
  const typename ELFT::Word SymOffset = HashTab.symndx;
  ArrayRef<typename ELFT::Off> Filter = HashTab.filter();
  ArrayRef<typename ELFT::Word> Bucket = HashTab.buckets();
  ArrayRef<typename ELFT::Word> Chain = HashTab.values(SymTab.size());

  // Check the bloom filter and exit early if the symbol is not present.
  uint64_t ElfClassBits = ELFT::Is64Bits ? 64 : 32;
  typename ELFT::Off Word =
      Filter[(NameHash / ElfClassBits) % HashTab.maskwords];
  uint64_t Mask = (0x1ull << (NameHash % ElfClassBits)) |
                  (0x1ull << ((NameHash >> HashTab.shift2) % ElfClassBits));
  if ((Word & Mask) != Mask)
    return nullptr;

  // The symbol may or may not be present, check the hash values.
  for (typename ELFT::Word I = Bucket[NameHash % NBucket];
       I >= SymOffset && I < SymTab.size(); I = I + 1) {
    const uint32_t ChainHash = Chain[I - SymOffset];

    if ((NameHash | 0x1) != (ChainHash | 0x1))
      continue;

    if (SymTab[I].st_name >= StrTab.size())
      return createError("symbol [index " + Twine(I) +
                         "] has invalid st_name: " + Twine(SymTab[I].st_name));
    if (StrTab.drop_front(SymTab[I].st_name).data() == Name)
      return &SymTab[I];

    if (ChainHash & 0x1)
      return nullptr;
  }
  return nullptr;
}

template <class ELFT>
static Expected<const typename ELFT::Sym *>
getSymbolFromSysVHashTable(StringRef Name, const typename ELFT::Hash &HashTab,
                           ArrayRef<typename ELFT::Sym> SymTab,
                           StringRef StrTab) {
  const uint32_t Hash = hashSysV(Name);
  const typename ELFT::Word NBucket = HashTab.nbucket;
  ArrayRef<typename ELFT::Word> Bucket = HashTab.buckets();
  ArrayRef<typename ELFT::Word> Chain = HashTab.chains();
  for (typename ELFT::Word I = Bucket[Hash % NBucket]; I != ELF::STN_UNDEF;
       I = Chain[I]) {
    if (I >= SymTab.size())
      return createError(
          "symbol [index " + Twine(I) +
          "] is greater than the number of symbols: " + Twine(SymTab.size()));
    if (SymTab[I].st_name >= StrTab.size())
      return createError("symbol [index " + Twine(I) +
                         "] has invalid st_name: " + Twine(SymTab[I].st_name));

    if (StrTab.drop_front(SymTab[I].st_name).data() == Name)
      return &SymTab[I];
  }
  return nullptr;
}

template <class ELFT>
static Expected<const typename ELFT::Sym *>
getHashTableSymbol(const ELFFile<ELFT> &Elf, const typename ELFT::Shdr &Sec,
                   StringRef Name) {
  if (Sec.sh_type != ELF::SHT_HASH && Sec.sh_type != ELF::SHT_GNU_HASH)
    return createError(
        "invalid sh_type for hash table, expected SHT_HASH or SHT_GNU_HASH");
  Expected<typename ELFT::ShdrRange> SectionsOrError = Elf.sections();
  if (!SectionsOrError)
    return SectionsOrError.takeError();

  auto SymTabOrErr = getSection<ELFT>(*SectionsOrError, Sec.sh_link);
  if (!SymTabOrErr)
    return SymTabOrErr.takeError();

  auto StrTabOrErr =
      Elf.getStringTableForSymtab(**SymTabOrErr, *SectionsOrError);
  if (!StrTabOrErr)
    return StrTabOrErr.takeError();
  StringRef StrTab = *StrTabOrErr;

  auto SymsOrErr = Elf.symbols(*SymTabOrErr);
  if (!SymsOrErr)
    return SymsOrErr.takeError();
  ArrayRef<typename ELFT::Sym> SymTab = *SymsOrErr;

  // If this is a GNU hash table we verify its size and search the symbol
  // table using the GNU hash table format.
  if (Sec.sh_type == ELF::SHT_GNU_HASH) {
    const typename ELFT::GnuHash *HashTab =
        reinterpret_cast<const typename ELFT::GnuHash *>(Elf.base() +
                                                         Sec.sh_offset);
    if (Sec.sh_offset + Sec.sh_size >= Elf.getBufSize())
      return createError("section has invalid sh_offset: " +
                         Twine(Sec.sh_offset));
    if (Sec.sh_size < sizeof(typename ELFT::GnuHash) ||
        Sec.sh_size <
            sizeof(typename ELFT::GnuHash) +
                sizeof(typename ELFT::Word) * HashTab->maskwords +
                sizeof(typename ELFT::Word) * HashTab->nbuckets +
                sizeof(typename ELFT::Word) * (SymTab.size() - HashTab->symndx))
      return createError("section has invalid sh_size: " + Twine(Sec.sh_size));
    return getSymbolFromGnuHashTable<ELFT>(Name, *HashTab, SymTab, StrTab);
  }

  // If this is a Sys-V hash table we verify its size and search the symbol
  // table using the Sys-V hash table format.
  if (Sec.sh_type == ELF::SHT_HASH) {
    const typename ELFT::Hash *HashTab =
        reinterpret_cast<const typename ELFT::Hash *>(Elf.base() +
                                                      Sec.sh_offset);
    if (Sec.sh_offset + Sec.sh_size >= Elf.getBufSize())
      return createError("section has invalid sh_offset: " +
                         Twine(Sec.sh_offset));
    if (Sec.sh_size < sizeof(typename ELFT::Hash) ||
        Sec.sh_size < sizeof(typename ELFT::Hash) +
                          sizeof(typename ELFT::Word) * HashTab->nbucket +
                          sizeof(typename ELFT::Word) * HashTab->nchain)
      return createError("section has invalid sh_size: " + Twine(Sec.sh_size));

    return getSymbolFromSysVHashTable<ELFT>(Name, *HashTab, SymTab, StrTab);
  }

  return nullptr;
}

template <class ELFT>
static Expected<const typename ELFT::Sym *>
getSymTableSymbol(const ELFFile<ELFT> &Elf, const typename ELFT::Shdr &Sec,
                  StringRef Name) {
  if (Sec.sh_type != ELF::SHT_SYMTAB && Sec.sh_type != ELF::SHT_DYNSYM)
    return createError(
        "invalid sh_type for hash table, expected SHT_SYMTAB or SHT_DYNSYM");
  Expected<typename ELFT::ShdrRange> SectionsOrError = Elf.sections();
  if (!SectionsOrError)
    return SectionsOrError.takeError();

  auto StrTabOrErr = Elf.getStringTableForSymtab(Sec, *SectionsOrError);
  if (!StrTabOrErr)
    return StrTabOrErr.takeError();
  StringRef StrTab = *StrTabOrErr;

  auto SymsOrErr = Elf.symbols(&Sec);
  if (!SymsOrErr)
    return SymsOrErr.takeError();
  ArrayRef<typename ELFT::Sym> SymTab = *SymsOrErr;

  for (const typename ELFT::Sym &Sym : SymTab)
    if (StrTab.drop_front(Sym.st_name).data() == Name)
      return &Sym;

  return nullptr;
}

Expected<const typename ELF64LE::Sym *>
utils::elf::getSymbol(const ELFObjectFile<ELF64LE> &ELFObj, StringRef Name) {
  // First try to look up the symbol via the hash table.
  for (ELFSectionRef Sec : ELFObj.sections()) {
    if (Sec.getType() != SHT_HASH && Sec.getType() != SHT_GNU_HASH)
      continue;

    auto HashTabOrErr = ELFObj.getELFFile().getSection(Sec.getIndex());
    if (!HashTabOrErr)
      return HashTabOrErr.takeError();
    return getHashTableSymbol<ELF64LE>(ELFObj.getELFFile(), **HashTabOrErr,
                                       Name);
  }

  // If this is an executable file check the entire standard symbol table.
  for (ELFSectionRef Sec : ELFObj.sections()) {
    if (Sec.getType() != SHT_SYMTAB)
      continue;

    auto SymTabOrErr = ELFObj.getELFFile().getSection(Sec.getIndex());
    if (!SymTabOrErr)
      return SymTabOrErr.takeError();
    return getSymTableSymbol<ELF64LE>(ELFObj.getELFFile(), **SymTabOrErr, Name);
  }

  return nullptr;
}

Expected<const void *> utils::elf::getSymbolAddress(
    const object::ELFObjectFile<object::ELF64LE> &ELFObj,
    const object::ELF64LE::Sym &Symbol) {
  const ELFFile<ELF64LE> &ELFFile = ELFObj.getELFFile();

  auto SecOrErr = ELFFile.getSection(Symbol.st_shndx);
  if (!SecOrErr)
    return SecOrErr.takeError();
  const auto &Section = *SecOrErr;

  // A section with SHT_NOBITS occupies no space in the file and has no
  // offset.
  if (Section->sh_type == ELF::SHT_NOBITS)
    return createError(
        "invalid sh_type for symbol lookup, cannot be SHT_NOBITS");

  uint64_t Offset = Section->sh_offset - Section->sh_addr + Symbol.st_value;
  if (Offset > ELFFile.getBufSize())
    return createError("invalid offset [" + Twine(Offset) +
                       "] into ELF file of size [" +
                       Twine(ELFFile.getBufSize()) + "]");

  return ELFFile.base() + Offset;
}
