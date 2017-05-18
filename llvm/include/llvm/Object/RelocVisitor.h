//===- RelocVisitor.h - Visitor for object file relocations -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a wrapper around all the different types of relocations
// in different file formats, such that a client can handle them in a unified
// manner by only implementing a minimal number of functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_RELOCVISITOR_H
#define LLVM_OBJECT_RELOCVISITOR_H

#include "llvm/ADT/Triple.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MachO.h"
#include <cstdint>
#include <system_error>

namespace llvm {
namespace object {

/// @brief Base class for object file relocation visitors.
class RelocVisitor {
public:
  explicit RelocVisitor(const ObjectFile &Obj) : ObjToVisit(Obj) {}

  // TODO: Should handle multiple applied relocations via either passing in the
  // previously computed value or just count paired relocations as a single
  // visit.
  uint64_t visit(uint32_t RelocType, RelocationRef R, uint64_t Value = 0) {
    if (isa<ELFObjectFileBase>(ObjToVisit))
      return visitELF(RelocType, R, Value);
    if (isa<COFFObjectFile>(ObjToVisit))
      return visitCOFF(RelocType, R, Value);
    if (isa<MachOObjectFile>(ObjToVisit))
      return visitMachO(RelocType, R, Value);

    HasError = true;
    return 0;
  }

  bool error() { return HasError; }

private:
  const ObjectFile &ObjToVisit;
  bool HasError = false;

  uint64_t visitELF(uint32_t RelocType, RelocationRef R, uint64_t Value) {
    if (ObjToVisit.getBytesInAddress() == 8) { // 64-bit object file
      switch (ObjToVisit.getArch()) {
      case Triple::x86_64:
        switch (RelocType) {
        case ELF::R_X86_64_NONE:
          return visitELF_X86_64_NONE(R);
        case ELF::R_X86_64_64:
          return visitELF_X86_64_64(R, Value);
        case ELF::R_X86_64_PC32:
          return visitELF_X86_64_PC32(R, Value);
        case ELF::R_X86_64_32:
          return visitELF_X86_64_32(R, Value);
        case ELF::R_X86_64_32S:
          return visitELF_X86_64_32S(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::aarch64:
      case Triple::aarch64_be:
        switch (RelocType) {
        case ELF::R_AARCH64_ABS32:
          return visitELF_AARCH64_ABS32(R, Value);
        case ELF::R_AARCH64_ABS64:
          return visitELF_AARCH64_ABS64(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::bpfel:
      case Triple::bpfeb:
        switch (RelocType) {
        case ELF::R_BPF_64_64:
          return visitELF_BPF_64_64(R, Value);
        case ELF::R_BPF_64_32:
          return visitELF_BPF_64_32(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::mips64el:
      case Triple::mips64:
        switch (RelocType) {
        case ELF::R_MIPS_32:
          return visitELF_MIPS64_32(R, Value);
        case ELF::R_MIPS_64:
          return visitELF_MIPS64_64(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::ppc64le:
      case Triple::ppc64:
        switch (RelocType) {
        case ELF::R_PPC64_ADDR32:
          return visitELF_PPC64_ADDR32(R, Value);
        case ELF::R_PPC64_ADDR64:
          return visitELF_PPC64_ADDR64(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::systemz:
        switch (RelocType) {
        case ELF::R_390_32:
          return visitELF_390_32(R, Value);
        case ELF::R_390_64:
          return visitELF_390_64(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::sparcv9:
        switch (RelocType) {
        case ELF::R_SPARC_32:
        case ELF::R_SPARC_UA32:
          return visitELF_SPARCV9_32(R, Value);
        case ELF::R_SPARC_64:
        case ELF::R_SPARC_UA64:
          return visitELF_SPARCV9_64(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::amdgcn:
        switch (RelocType) {
        case ELF::R_AMDGPU_ABS32:
          return visitELF_AMDGPU_ABS32(R, Value);
        case ELF::R_AMDGPU_ABS64:
          return visitELF_AMDGPU_ABS64(R, Value);
        default:
          HasError = true;
          return 0;
        }
      default:
        HasError = true;
        return 0;
      }
    } else if (ObjToVisit.getBytesInAddress() == 4) { // 32-bit object file
      switch (ObjToVisit.getArch()) {
      case Triple::x86:
        switch (RelocType) {
        case ELF::R_386_NONE:
          return visitELF_386_NONE(R);
        case ELF::R_386_32:
          return visitELF_386_32(R, Value);
        case ELF::R_386_PC32:
          return visitELF_386_PC32(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::ppc:
        switch (RelocType) {
        case ELF::R_PPC_ADDR32:
          return visitELF_PPC_ADDR32(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::arm:
      case Triple::armeb:
        switch (RelocType) {
        default:
          HasError = true;
          return 0;
        case ELF::R_ARM_ABS32:
          return visitELF_ARM_ABS32(R, Value);
        }
      case Triple::lanai:
        switch (RelocType) {
        case ELF::R_LANAI_32:
          return visitELF_Lanai_32(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::mipsel:
      case Triple::mips:
        switch (RelocType) {
        case ELF::R_MIPS_32:
          return visitELF_MIPS_32(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::sparc:
        switch (RelocType) {
        case ELF::R_SPARC_32:
        case ELF::R_SPARC_UA32:
          return visitELF_SPARC_32(R, Value);
        default:
          HasError = true;
          return 0;
        }
      case Triple::hexagon:
        switch (RelocType) {
        case ELF::R_HEX_32:
          return visitELF_HEX_32(R, Value);
        default:
          HasError = true;
          return 0;
        }
      default:
        HasError = true;
        return 0;
      }
    } else {
      report_fatal_error("Invalid word size in object file");
    }
  }

  uint64_t visitCOFF(uint32_t RelocType, RelocationRef R, uint64_t Value) {
    switch (ObjToVisit.getArch()) {
    case Triple::x86:
      switch (RelocType) {
      case COFF::IMAGE_REL_I386_SECREL:
        return visitCOFF_I386_SECREL(R, Value);
      case COFF::IMAGE_REL_I386_DIR32:
        return visitCOFF_I386_DIR32(R, Value);
      }
      break;
    case Triple::x86_64:
      switch (RelocType) {
      case COFF::IMAGE_REL_AMD64_SECREL:
        return visitCOFF_AMD64_SECREL(R, Value);
      case COFF::IMAGE_REL_AMD64_ADDR64:
        return visitCOFF_AMD64_ADDR64(R, Value);
      }
      break;
    }
    HasError = true;
    return 0;
  }

  uint64_t visitMachO(uint32_t RelocType, RelocationRef R, uint64_t Value) {
    switch (ObjToVisit.getArch()) {
    default: break;
    case Triple::x86_64:
      switch (RelocType) {
        default: break;
        case MachO::X86_64_RELOC_UNSIGNED:
          return visitMACHO_X86_64_UNSIGNED(R, Value);
      }
    }
    HasError = true;
    return 0;
  }

  int64_t getELFAddend(RelocationRef R) {
    ErrorOr<int64_t> AddendOrErr = ELFRelocationRef(R).getAddend();
    if (std::error_code EC = AddendOrErr.getError())
      report_fatal_error(EC.message());
    return *AddendOrErr;
  }

  /// Operations

  /// 386-ELF
  uint64_t visitELF_386_NONE(RelocationRef R) {
    return 0;
  }

  // Ideally the Addend here will be the addend in the data for
  // the relocation. It's not actually the case for Rel relocations.
  uint64_t visitELF_386_32(RelocationRef R, uint64_t Value) {
    return Value;
  }

  uint64_t visitELF_386_PC32(RelocationRef R, uint64_t Value) {
    return Value - R.getOffset();
  }

  /// X86-64 ELF
  uint64_t visitELF_X86_64_NONE(RelocationRef R) {
    return 0;
  }

  uint64_t visitELF_X86_64_64(RelocationRef R, uint64_t Value) {
    return Value + getELFAddend(R);
  }

  uint64_t visitELF_X86_64_PC32(RelocationRef R, uint64_t Value) {
    return Value + getELFAddend(R) - R.getOffset();
  }

  uint64_t visitELF_X86_64_32(RelocationRef R, uint64_t Value) {
    return (Value + getELFAddend(R)) & 0xFFFFFFFF;
  }

  uint64_t visitELF_X86_64_32S(RelocationRef R, uint64_t Value) {
    return (Value + getELFAddend(R)) & 0xFFFFFFFF;
  }

  /// BPF ELF
  uint64_t visitELF_BPF_64_32(RelocationRef R, uint64_t Value) {
    return Value & 0xFFFFFFFF;
  }

  uint64_t visitELF_BPF_64_64(RelocationRef R, uint64_t Value) {
    return Value;
  }

  /// PPC64 ELF
  uint64_t visitELF_PPC64_ADDR32(RelocationRef R, uint64_t Value) {
    return (Value + getELFAddend(R)) & 0xFFFFFFFF;
  }

  uint64_t visitELF_PPC64_ADDR64(RelocationRef R, uint64_t Value) {
    return Value + getELFAddend(R);
  }

  /// PPC32 ELF
  uint64_t visitELF_PPC_ADDR32(RelocationRef R, uint64_t Value) {
    return (Value + getELFAddend(R)) & 0xFFFFFFFF;
  }

  /// Lanai ELF
  uint64_t visitELF_Lanai_32(RelocationRef R, uint64_t Value) {
    return (Value + getELFAddend(R)) & 0xFFFFFFFF;
  }

  /// MIPS ELF
  uint64_t visitELF_MIPS_32(RelocationRef R, uint64_t Value) {
    return Value & 0xFFFFFFFF;
  }

  /// MIPS64 ELF
  uint64_t visitELF_MIPS64_32(RelocationRef R, uint64_t Value) {
    return (Value + getELFAddend(R)) & 0xFFFFFFFF;
  }

  uint64_t visitELF_MIPS64_64(RelocationRef R, uint64_t Value) {
    return Value + getELFAddend(R);
  }

  // AArch64 ELF
  uint64_t visitELF_AARCH64_ABS32(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    int64_t Res =  Value + Addend;

    // Overflow check allows for both signed and unsigned interpretation.
    if (Res < INT32_MIN || Res > UINT32_MAX)
      HasError = true;

    return static_cast<uint32_t>(Res);
  }

  uint64_t visitELF_AARCH64_ABS64(RelocationRef R, uint64_t Value) {
    return Value + getELFAddend(R);
  }

  // SystemZ ELF
  uint64_t visitELF_390_32(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    int64_t Res = Value + Addend;

    // Overflow check allows for both signed and unsigned interpretation.
    if (Res < INT32_MIN || Res > UINT32_MAX)
      HasError = true;

    return static_cast<uint32_t>(Res);
  }

  uint64_t visitELF_390_64(RelocationRef R, uint64_t Value) {
    return Value + getELFAddend(R);
  }

  uint64_t visitELF_SPARC_32(RelocationRef R, uint32_t Value) {
    return Value + getELFAddend(R);
  }

  uint64_t visitELF_SPARCV9_32(RelocationRef R, uint64_t Value) {
    return Value + getELFAddend(R);
  }

  uint64_t visitELF_SPARCV9_64(RelocationRef R, uint64_t Value) {
    return Value + getELFAddend(R);
  }

  uint64_t visitELF_ARM_ABS32(RelocationRef R, uint64_t Value) {
    int64_t Res = Value;

    // Overflow check allows for both signed and unsigned interpretation.
    if (Res < INT32_MIN || Res > UINT32_MAX)
      HasError = true;

    return static_cast<uint32_t>(Res);
  }

  uint64_t visitELF_HEX_32(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    return Value + Addend;
  }

  uint64_t visitELF_AMDGPU_ABS32(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    return Value + Addend;
  }

  uint64_t visitELF_AMDGPU_ABS64(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    return Value + Addend;
  }

  /// I386 COFF
  uint64_t visitCOFF_I386_SECREL(RelocationRef R, uint64_t Value) {
    return static_cast<uint32_t>(Value);
  }

  uint64_t visitCOFF_I386_DIR32(RelocationRef R, uint64_t Value) {
    return static_cast<uint32_t>(Value);
  }

  /// AMD64 COFF
  uint64_t visitCOFF_AMD64_SECREL(RelocationRef R, uint64_t Value) {
    return static_cast<uint32_t>(Value);
  }

  uint64_t visitCOFF_AMD64_ADDR64(RelocationRef R, uint64_t Value) {
    return Value;
  }

  // X86_64 MachO
  uint64_t visitMACHO_X86_64_UNSIGNED(RelocationRef R, uint64_t Value) {
    return Value;
  }
};

} // end namespace object
} // end namespace llvm

#endif // LLVM_OBJECT_RELOCVISITOR_H
