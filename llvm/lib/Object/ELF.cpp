//===- ELF.cpp - ELF object file implementation ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELF.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;
using namespace object;

#define STRINGIFY_ENUM_CASE(ns, name)                                          \
  case ns::name:                                                               \
    return #name;

#define ELF_RELOC(name, value) STRINGIFY_ENUM_CASE(ELF, name)

StringRef llvm::object::getELFRelocationTypeName(uint32_t Machine,
                                                 uint32_t Type) {
  switch (Machine) {
  case ELF::EM_X86_64:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/x86_64.def"
    default:
      break;
    }
    break;
  case ELF::EM_386:
  case ELF::EM_IAMCU:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/i386.def"
    default:
      break;
    }
    break;
  case ELF::EM_MIPS:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/Mips.def"
    default:
      break;
    }
    break;
  case ELF::EM_AARCH64:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/AArch64.def"
    default:
      break;
    }
    break;
  case ELF::EM_ARM:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/ARM.def"
    default:
      break;
    }
    break;
  case ELF::EM_ARC_COMPACT:
  case ELF::EM_ARC_COMPACT2:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/ARC.def"
    default:
      break;
    }
    break;
  case ELF::EM_AVR:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/AVR.def"
    default:
      break;
    }
    break;
  case ELF::EM_HEXAGON:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/Hexagon.def"
    default:
      break;
    }
    break;
  case ELF::EM_LANAI:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/Lanai.def"
    default:
      break;
    }
    break;
  case ELF::EM_PPC:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/PowerPC.def"
    default:
      break;
    }
    break;
  case ELF::EM_PPC64:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/PowerPC64.def"
    default:
      break;
    }
    break;
  case ELF::EM_RISCV:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/RISCV.def"
    default:
      break;
    }
    break;
  case ELF::EM_S390:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/SystemZ.def"
    default:
      break;
    }
    break;
  case ELF::EM_SPARC:
  case ELF::EM_SPARC32PLUS:
  case ELF::EM_SPARCV9:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/Sparc.def"
    default:
      break;
    }
    break;
  case ELF::EM_WEBASSEMBLY:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/WebAssembly.def"
    default:
      break;
    }
    break;
  case ELF::EM_AMDGPU:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/AMDGPU.def"
    default:
      break;
    }
  case ELF::EM_BPF:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/BPF.def"
    default:
      break;
    }
    break;
  default:
    break;
  }
  return "Unknown";
}

#undef ELF_RELOC

StringRef llvm::object::getELFSectionTypeName(uint32_t Machine, unsigned Type) {
  switch (Machine) {
  case ELF::EM_ARM:
    switch (Type) {
      STRINGIFY_ENUM_CASE(ELF, SHT_ARM_EXIDX);
      STRINGIFY_ENUM_CASE(ELF, SHT_ARM_PREEMPTMAP);
      STRINGIFY_ENUM_CASE(ELF, SHT_ARM_ATTRIBUTES);
      STRINGIFY_ENUM_CASE(ELF, SHT_ARM_DEBUGOVERLAY);
      STRINGIFY_ENUM_CASE(ELF, SHT_ARM_OVERLAYSECTION);
    }
    break;
  case ELF::EM_HEXAGON:
    switch (Type) { STRINGIFY_ENUM_CASE(ELF, SHT_HEX_ORDERED); }
    break;
  case ELF::EM_X86_64:
    switch (Type) { STRINGIFY_ENUM_CASE(ELF, SHT_X86_64_UNWIND); }
    break;
  case ELF::EM_MIPS:
  case ELF::EM_MIPS_RS3_LE:
    switch (Type) {
      STRINGIFY_ENUM_CASE(ELF, SHT_MIPS_REGINFO);
      STRINGIFY_ENUM_CASE(ELF, SHT_MIPS_OPTIONS);
      STRINGIFY_ENUM_CASE(ELF, SHT_MIPS_ABIFLAGS);
      STRINGIFY_ENUM_CASE(ELF, SHT_MIPS_DWARF);
    }
    break;
  default:
    break;
  }

  switch (Type) {
    STRINGIFY_ENUM_CASE(ELF, SHT_NULL);
    STRINGIFY_ENUM_CASE(ELF, SHT_PROGBITS);
    STRINGIFY_ENUM_CASE(ELF, SHT_SYMTAB);
    STRINGIFY_ENUM_CASE(ELF, SHT_STRTAB);
    STRINGIFY_ENUM_CASE(ELF, SHT_RELA);
    STRINGIFY_ENUM_CASE(ELF, SHT_HASH);
    STRINGIFY_ENUM_CASE(ELF, SHT_DYNAMIC);
    STRINGIFY_ENUM_CASE(ELF, SHT_NOTE);
    STRINGIFY_ENUM_CASE(ELF, SHT_NOBITS);
    STRINGIFY_ENUM_CASE(ELF, SHT_REL);
    STRINGIFY_ENUM_CASE(ELF, SHT_SHLIB);
    STRINGIFY_ENUM_CASE(ELF, SHT_DYNSYM);
    STRINGIFY_ENUM_CASE(ELF, SHT_INIT_ARRAY);
    STRINGIFY_ENUM_CASE(ELF, SHT_FINI_ARRAY);
    STRINGIFY_ENUM_CASE(ELF, SHT_PREINIT_ARRAY);
    STRINGIFY_ENUM_CASE(ELF, SHT_GROUP);
    STRINGIFY_ENUM_CASE(ELF, SHT_SYMTAB_SHNDX);
    STRINGIFY_ENUM_CASE(ELF, SHT_ANDROID_REL);
    STRINGIFY_ENUM_CASE(ELF, SHT_ANDROID_RELA);
    STRINGIFY_ENUM_CASE(ELF, SHT_LLVM_ODRTAB);
    STRINGIFY_ENUM_CASE(ELF, SHT_GNU_ATTRIBUTES);
    STRINGIFY_ENUM_CASE(ELF, SHT_GNU_HASH);
    STRINGIFY_ENUM_CASE(ELF, SHT_GNU_verdef);
    STRINGIFY_ENUM_CASE(ELF, SHT_GNU_verneed);
    STRINGIFY_ENUM_CASE(ELF, SHT_GNU_versym);
  default:
    return "Unknown";
  }
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
  auto SymtabOrErr = symbols(Sec);
  if (!SymtabOrErr)
    return SymtabOrErr.takeError();
  return object::getSymbol<ELFT>(*SymtabOrErr, Index);
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

template <class ELFT> ELFFile<ELFT>::ELFFile(StringRef Object) : Buf(Object) {}

template <class ELFT>
Expected<ELFFile<ELFT>> ELFFile<ELFT>::create(StringRef Object) {
  if (sizeof(Elf_Ehdr) > Object.size())
    return createError("Invalid buffer");
  return ELFFile(Object);
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

template <class ELFT>
Expected<std::vector<typename ELFT::Rela>>
ELFFile<ELFT>::android_relas(const Elf_Shdr *Sec) const {
  // This function reads relocations in Android's packed relocation format,
  // which is based on SLEB128 and delta encoding.
  Expected<ArrayRef<uint8_t>> ContentsOrErr = getSectionContents(Sec);
  if (!ContentsOrErr)
    return ContentsOrErr.takeError();
  const uint8_t *Cur = ContentsOrErr->begin();
  const uint8_t *End = ContentsOrErr->end();
  if (ContentsOrErr->size() < 4 || Cur[0] != 'A' || Cur[1] != 'P' ||
      Cur[2] != 'S' || Cur[3] != '2')
    return createError("invalid packed relocation header");
  Cur += 4;

  const char *ErrStr = nullptr;
  auto ReadSLEB = [&]() -> int64_t {
    if (ErrStr)
      return 0;
    unsigned Len;
    int64_t Result = decodeSLEB128(Cur, &Len, End, &ErrStr);
    Cur += Len;
    return Result;
  };

  uint64_t NumRelocs = ReadSLEB();
  uint64_t Offset = ReadSLEB();
  uint64_t Addend = 0;

  if (ErrStr)
    return createError(ErrStr);

  std::vector<Elf_Rela> Relocs;
  Relocs.reserve(NumRelocs);
  while (NumRelocs) {
    uint64_t NumRelocsInGroup = ReadSLEB();
    if (NumRelocsInGroup > NumRelocs)
      return createError("relocation group unexpectedly large");
    NumRelocs -= NumRelocsInGroup;

    uint64_t GroupFlags = ReadSLEB();
    bool GroupedByInfo = GroupFlags & ELF::RELOCATION_GROUPED_BY_INFO_FLAG;
    bool GroupedByOffsetDelta = GroupFlags & ELF::RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG;
    bool GroupedByAddend = GroupFlags & ELF::RELOCATION_GROUPED_BY_ADDEND_FLAG;
    bool GroupHasAddend = GroupFlags & ELF::RELOCATION_GROUP_HAS_ADDEND_FLAG;

    uint64_t GroupOffsetDelta;
    if (GroupedByOffsetDelta)
      GroupOffsetDelta = ReadSLEB();

    uint64_t GroupRInfo;
    if (GroupedByInfo)
      GroupRInfo = ReadSLEB();

    if (GroupedByAddend && GroupHasAddend)
      Addend += ReadSLEB();

    for (uint64_t I = 0; I != NumRelocsInGroup; ++I) {
      Elf_Rela R;
      Offset += GroupedByOffsetDelta ? GroupOffsetDelta : ReadSLEB();
      R.r_offset = Offset;
      R.r_info = GroupedByInfo ? GroupRInfo : ReadSLEB();

      if (GroupHasAddend) {
        if (!GroupedByAddend)
          Addend += ReadSLEB();
        R.r_addend = Addend;
      } else {
        R.r_addend = 0;
      }

      Relocs.push_back(R);

      if (ErrStr)
        return createError(ErrStr);
    }

    if (ErrStr)
      return createError(ErrStr);
  }

  return Relocs;
}

template class llvm::object::ELFFile<ELF32LE>;
template class llvm::object::ELFFile<ELF32BE>;
template class llvm::object::ELFFile<ELF64LE>;
template class llvm::object::ELFFile<ELF64BE>;
