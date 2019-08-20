//===- RelocationResolver.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities to resolve relocations in object files.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/RelocationResolver.h"

namespace llvm {
namespace object {

static int64_t getELFAddend(RelocationRef R) {
  Expected<int64_t> AddendOrErr = ELFRelocationRef(R).getAddend();
  handleAllErrors(AddendOrErr.takeError(), [](const ErrorInfoBase &EI) {
    report_fatal_error(EI.message());
  });
  return *AddendOrErr;
}

static bool supportsX86_64(uint64_t Type) {
  switch (Type) {
  case ELF::R_X86_64_NONE:
  case ELF::R_X86_64_64:
  case ELF::R_X86_64_DTPOFF32:
  case ELF::R_X86_64_DTPOFF64:
  case ELF::R_X86_64_PC32:
  case ELF::R_X86_64_32:
  case ELF::R_X86_64_32S:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveX86_64(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case ELF::R_X86_64_NONE:
    return A;
  case ELF::R_X86_64_64:
  case ELF::R_X86_64_DTPOFF32:
  case ELF::R_X86_64_DTPOFF64:
    return S + getELFAddend(R);
  case ELF::R_X86_64_PC32:
    return S + getELFAddend(R) - R.getOffset();
  case ELF::R_X86_64_32:
  case ELF::R_X86_64_32S:
    return (S + getELFAddend(R)) & 0xFFFFFFFF;
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsAArch64(uint64_t Type) {
  switch (Type) {
  case ELF::R_AARCH64_ABS32:
  case ELF::R_AARCH64_ABS64:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveAArch64(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case ELF::R_AARCH64_ABS32:
    return (S + getELFAddend(R)) & 0xFFFFFFFF;
  case ELF::R_AARCH64_ABS64:
    return S + getELFAddend(R);
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsBPF(uint64_t Type) {
  switch (Type) {
  case ELF::R_BPF_64_32:
  case ELF::R_BPF_64_64:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveBPF(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case ELF::R_BPF_64_32:
    return (S + A) & 0xFFFFFFFF;
  case ELF::R_BPF_64_64:
    return S + A;
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsMips64(uint64_t Type) {
  switch (Type) {
  case ELF::R_MIPS_32:
  case ELF::R_MIPS_64:
  case ELF::R_MIPS_TLS_DTPREL64:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveMips64(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case ELF::R_MIPS_32:
    return (S + getELFAddend(R)) & 0xFFFFFFFF;
  case ELF::R_MIPS_64:
    return S + getELFAddend(R);
  case ELF::R_MIPS_TLS_DTPREL64:
    return S + getELFAddend(R) - 0x8000;
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsPPC64(uint64_t Type) {
  switch (Type) {
  case ELF::R_PPC64_ADDR32:
  case ELF::R_PPC64_ADDR64:
    return true;
  default:
    return false;
  }
}

static uint64_t resolvePPC64(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case ELF::R_PPC64_ADDR32:
    return (S + getELFAddend(R)) & 0xFFFFFFFF;
  case ELF::R_PPC64_ADDR64:
    return S + getELFAddend(R);
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsSystemZ(uint64_t Type) {
  switch (Type) {
  case ELF::R_390_32:
  case ELF::R_390_64:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveSystemZ(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case ELF::R_390_32:
    return (S + getELFAddend(R)) & 0xFFFFFFFF;
  case ELF::R_390_64:
    return S + getELFAddend(R);
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsSparc64(uint64_t Type) {
  switch (Type) {
  case ELF::R_SPARC_32:
  case ELF::R_SPARC_64:
  case ELF::R_SPARC_UA32:
  case ELF::R_SPARC_UA64:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveSparc64(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case ELF::R_SPARC_32:
  case ELF::R_SPARC_64:
  case ELF::R_SPARC_UA32:
  case ELF::R_SPARC_UA64:
    return S + getELFAddend(R);
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsAmdgpu(uint64_t Type) {
  switch (Type) {
  case ELF::R_AMDGPU_ABS32:
  case ELF::R_AMDGPU_ABS64:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveAmdgpu(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case ELF::R_AMDGPU_ABS32:
  case ELF::R_AMDGPU_ABS64:
    return S + getELFAddend(R);
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsX86(uint64_t Type) {
  switch (Type) {
  case ELF::R_386_NONE:
  case ELF::R_386_32:
  case ELF::R_386_PC32:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveX86(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case ELF::R_386_NONE:
    return A;
  case ELF::R_386_32:
    return S + A;
  case ELF::R_386_PC32:
    return S - R.getOffset() + A;
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsPPC32(uint64_t Type) {
  return Type == ELF::R_PPC_ADDR32;
}

static uint64_t resolvePPC32(RelocationRef R, uint64_t S, uint64_t A) {
  if (R.getType() == ELF::R_PPC_ADDR32)
    return (S + getELFAddend(R)) & 0xFFFFFFFF;
  llvm_unreachable("Invalid relocation type");
}

static bool supportsARM(uint64_t Type) {
  return Type == ELF::R_ARM_ABS32;
}

static uint64_t resolveARM(RelocationRef R, uint64_t S, uint64_t A) {
  if (R.getType() == ELF::R_ARM_ABS32)
    return (S + A) & 0xFFFFFFFF;
  llvm_unreachable("Invalid relocation type");
}

static bool supportsAVR(uint64_t Type) {
  switch (Type) {
  case ELF::R_AVR_16:
  case ELF::R_AVR_32:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveAVR(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case ELF::R_AVR_16:
    return (S + getELFAddend(R)) & 0xFFFF;
  case ELF::R_AVR_32:
    return (S + getELFAddend(R)) & 0xFFFFFFFF;
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsLanai(uint64_t Type) {
  return Type == ELF::R_LANAI_32;
}

static uint64_t resolveLanai(RelocationRef R, uint64_t S, uint64_t A) {
  if (R.getType() == ELF::R_LANAI_32)
    return (S + getELFAddend(R)) & 0xFFFFFFFF;
  llvm_unreachable("Invalid relocation type");
}

static bool supportsMips32(uint64_t Type) {
  switch (Type) {
  case ELF::R_MIPS_32:
  case ELF::R_MIPS_TLS_DTPREL32:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveMips32(RelocationRef R, uint64_t S, uint64_t A) {
  // FIXME: Take in account implicit addends to get correct results.
  uint32_t Rel = R.getType();
  if (Rel == ELF::R_MIPS_32)
    return (S + A) & 0xFFFFFFFF;
  if (Rel == ELF::R_MIPS_TLS_DTPREL32)
    return (S + A) & 0xFFFFFFFF;
  llvm_unreachable("Invalid relocation type");
}

static bool supportsSparc32(uint64_t Type) {
  switch (Type) {
  case ELF::R_SPARC_32:
  case ELF::R_SPARC_UA32:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveSparc32(RelocationRef R, uint64_t S, uint64_t A) {
  uint32_t Rel = R.getType();
  if (Rel == ELF::R_SPARC_32 || Rel == ELF::R_SPARC_UA32)
    return S + getELFAddend(R);
  return A;
}

static bool supportsHexagon(uint64_t Type) {
  return Type == ELF::R_HEX_32;
}

static uint64_t resolveHexagon(RelocationRef R, uint64_t S, uint64_t A) {
  if (R.getType() == ELF::R_HEX_32)
    return S + getELFAddend(R);
  llvm_unreachable("Invalid relocation type");
}

static bool supportsRISCV(uint64_t Type) {
  switch (Type) {
  case ELF::R_RISCV_NONE:
  case ELF::R_RISCV_32:
  case ELF::R_RISCV_64:
  case ELF::R_RISCV_ADD8:
  case ELF::R_RISCV_SUB8:
  case ELF::R_RISCV_ADD16:
  case ELF::R_RISCV_SUB16:
  case ELF::R_RISCV_ADD32:
  case ELF::R_RISCV_SUB32:
  case ELF::R_RISCV_ADD64:
  case ELF::R_RISCV_SUB64:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveRISCV(RelocationRef R, uint64_t S, uint64_t A) {
  int64_t RA = getELFAddend(R);
  switch (R.getType()) {
  case ELF::R_RISCV_NONE:
    return A;
  case ELF::R_RISCV_32:
    return (S + RA) & 0xFFFFFFFF;
  case ELF::R_RISCV_64:
    return S + RA;
  case ELF::R_RISCV_ADD8:
    return (A + (S + RA)) & 0xFF;
  case ELF::R_RISCV_SUB8:
    return (A - (S + RA)) & 0xFF;
  case ELF::R_RISCV_ADD16:
    return (A + (S + RA)) & 0xFFFF;
  case ELF::R_RISCV_SUB16:
    return (A - (S + RA)) & 0xFFFF;
  case ELF::R_RISCV_ADD32:
    return (A + (S + RA)) & 0xFFFFFFFF;
  case ELF::R_RISCV_SUB32:
    return (A - (S + RA)) & 0xFFFFFFFF;
  case ELF::R_RISCV_ADD64:
    return (A + (S + RA));
  case ELF::R_RISCV_SUB64:
    return (A - (S + RA));
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsCOFFX86(uint64_t Type) {
  switch (Type) {
  case COFF::IMAGE_REL_I386_SECREL:
  case COFF::IMAGE_REL_I386_DIR32:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveCOFFX86(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case COFF::IMAGE_REL_I386_SECREL:
  case COFF::IMAGE_REL_I386_DIR32:
    return (S + A) & 0xFFFFFFFF;
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsCOFFX86_64(uint64_t Type) {
  switch (Type) {
  case COFF::IMAGE_REL_AMD64_SECREL:
  case COFF::IMAGE_REL_AMD64_ADDR64:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveCOFFX86_64(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case COFF::IMAGE_REL_AMD64_SECREL:
    return (S + A) & 0xFFFFFFFF;
  case COFF::IMAGE_REL_AMD64_ADDR64:
    return S + A;
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

static bool supportsMachOX86_64(uint64_t Type) {
  return Type == MachO::X86_64_RELOC_UNSIGNED;
}

static uint64_t resolveMachOX86_64(RelocationRef R, uint64_t S, uint64_t A) {
  if (R.getType() == MachO::X86_64_RELOC_UNSIGNED)
    return S;
  llvm_unreachable("Invalid relocation type");
}

static bool supportsWasm32(uint64_t Type) {
  switch (Type) {
  case wasm::R_WASM_FUNCTION_INDEX_LEB:
  case wasm::R_WASM_TABLE_INDEX_SLEB:
  case wasm::R_WASM_TABLE_INDEX_I32:
  case wasm::R_WASM_MEMORY_ADDR_LEB:
  case wasm::R_WASM_MEMORY_ADDR_SLEB:
  case wasm::R_WASM_MEMORY_ADDR_I32:
  case wasm::R_WASM_TYPE_INDEX_LEB:
  case wasm::R_WASM_GLOBAL_INDEX_LEB:
  case wasm::R_WASM_FUNCTION_OFFSET_I32:
  case wasm::R_WASM_SECTION_OFFSET_I32:
  case wasm::R_WASM_EVENT_INDEX_LEB:
    return true;
  default:
    return false;
  }
}

static uint64_t resolveWasm32(RelocationRef R, uint64_t S, uint64_t A) {
  switch (R.getType()) {
  case wasm::R_WASM_FUNCTION_INDEX_LEB:
  case wasm::R_WASM_TABLE_INDEX_SLEB:
  case wasm::R_WASM_TABLE_INDEX_I32:
  case wasm::R_WASM_MEMORY_ADDR_LEB:
  case wasm::R_WASM_MEMORY_ADDR_SLEB:
  case wasm::R_WASM_MEMORY_ADDR_I32:
  case wasm::R_WASM_TYPE_INDEX_LEB:
  case wasm::R_WASM_GLOBAL_INDEX_LEB:
  case wasm::R_WASM_FUNCTION_OFFSET_I32:
  case wasm::R_WASM_SECTION_OFFSET_I32:
  case wasm::R_WASM_EVENT_INDEX_LEB:
    // For wasm section, its offset at 0 -- ignoring Value
    return A;
  default:
    llvm_unreachable("Invalid relocation type");
  }
}

std::pair<bool (*)(uint64_t), RelocationResolver>
getRelocationResolver(const ObjectFile &Obj) {
  if (Obj.isCOFF()) {
    if (Obj.getBytesInAddress() == 8)
      return {supportsCOFFX86_64, resolveCOFFX86_64};
    return {supportsCOFFX86, resolveCOFFX86};
  } else if (Obj.isELF()) {
    if (Obj.getBytesInAddress() == 8) {
      switch (Obj.getArch()) {
      case Triple::x86_64:
        return {supportsX86_64, resolveX86_64};
      case Triple::aarch64:
      case Triple::aarch64_be:
        return {supportsAArch64, resolveAArch64};
      case Triple::bpfel:
      case Triple::bpfeb:
        return {supportsBPF, resolveBPF};
      case Triple::mips64el:
      case Triple::mips64:
        return {supportsMips64, resolveMips64};
      case Triple::ppc64le:
      case Triple::ppc64:
        return {supportsPPC64, resolvePPC64};
      case Triple::systemz:
        return {supportsSystemZ, resolveSystemZ};
      case Triple::sparcv9:
        return {supportsSparc64, resolveSparc64};
      case Triple::amdgcn:
        return {supportsAmdgpu, resolveAmdgpu};
      case Triple::riscv64:
        return {supportsRISCV, resolveRISCV};
      default:
        return {nullptr, nullptr};
      }
    }

    // 32-bit object file
    assert(Obj.getBytesInAddress() == 4 &&
           "Invalid word size in object file");

    switch (Obj.getArch()) {
    case Triple::x86:
      return {supportsX86, resolveX86};
    case Triple::ppc:
      return {supportsPPC32, resolvePPC32};
    case Triple::arm:
    case Triple::armeb:
      return {supportsARM, resolveARM};
    case Triple::avr:
      return {supportsAVR, resolveAVR};
    case Triple::lanai:
      return {supportsLanai, resolveLanai};
    case Triple::mipsel:
    case Triple::mips:
      return {supportsMips32, resolveMips32};
    case Triple::sparc:
      return {supportsSparc32, resolveSparc32};
    case Triple::hexagon:
      return {supportsHexagon, resolveHexagon};
    case Triple::riscv32:
      return {supportsRISCV, resolveRISCV};
    default:
      return {nullptr, nullptr};
    }
  } else if (Obj.isMachO()) {
    if (Obj.getArch() == Triple::x86_64)
      return {supportsMachOX86_64, resolveMachOX86_64};
    return {nullptr, nullptr};
  } else if (Obj.isWasm()) {
    if (Obj.getArch() == Triple::wasm32)
      return {supportsWasm32, resolveWasm32};
    return {nullptr, nullptr};
  }

  llvm_unreachable("Invalid object file");
}

} // namespace object
} // namespace llvm
