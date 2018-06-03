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
    break;
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
    STRINGIFY_ENUM_CASE(ELF, SHT_LLVM_LINKER_OPTIONS);
    STRINGIFY_ENUM_CASE(ELF, SHT_LLVM_CALL_GRAPH_PROFILE);
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
