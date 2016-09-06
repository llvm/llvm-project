//===- ELFObjectFile.cpp - ELF object file implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Part of the ELFObjectFile class implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {
using namespace object;

ELFObjectFileBase::ELFObjectFileBase(unsigned int Type, MemoryBufferRef Source)
    : ObjectFile(Type, Source) {}

ErrorOr<std::unique_ptr<ObjectFile>>
ObjectFile::createELFObjectFile(MemoryBufferRef Obj) {
  std::pair<unsigned char, unsigned char> Ident =
      getElfArchType(Obj.getBuffer());
  std::size_t MaxAlignment =
      1ULL << countTrailingZeros(uintptr_t(Obj.getBufferStart()));

  if (MaxAlignment < 2)
    return object_error::parse_failed;

  std::error_code EC;
  std::unique_ptr<ObjectFile> R;
  if (Ident.first == ELF::ELFCLASS32) {
    if (Ident.second == ELF::ELFDATA2LSB)
      R.reset(new ELFObjectFile<ELFType<support::little, false>>(Obj, EC));
    else if (Ident.second == ELF::ELFDATA2MSB)
      R.reset(new ELFObjectFile<ELFType<support::big, false>>(Obj, EC));
    else
      return object_error::parse_failed;
  } else if (Ident.first == ELF::ELFCLASS64) {
    if (Ident.second == ELF::ELFDATA2LSB)
      R.reset(new ELFObjectFile<ELFType<support::little, true>>(Obj, EC));
    else if (Ident.second == ELF::ELFDATA2MSB)
      R.reset(new ELFObjectFile<ELFType<support::big, true>>(Obj, EC));
    else
      return object_error::parse_failed;
  } else {
    return object_error::parse_failed;
  }

  if (EC)
    return EC;
  return std::move(R);
}

SubtargetFeatures ELFObjectFileBase::getFeatures() const {
  switch (getEMachine()) {
  case ELF::EM_MIPS: {
    SubtargetFeatures Features;
    unsigned PlatformFlags;
    getPlatformFlags(PlatformFlags);

    switch (PlatformFlags & ELF::EF_MIPS_ARCH) {
    case ELF::EF_MIPS_ARCH_1:
      break;
    case ELF::EF_MIPS_ARCH_2:
      Features.AddFeature("mips2");
      break;
    case ELF::EF_MIPS_ARCH_3:
      Features.AddFeature("mips3");
      break;
    case ELF::EF_MIPS_ARCH_4:
      Features.AddFeature("mips4");
      break;
    case ELF::EF_MIPS_ARCH_5:
      Features.AddFeature("mips5");
      break;
    case ELF::EF_MIPS_ARCH_32:
      Features.AddFeature("mips32");
      break;
    case ELF::EF_MIPS_ARCH_64:
      Features.AddFeature("mips64");
      break;
    case ELF::EF_MIPS_ARCH_32R2:
      Features.AddFeature("mips32r2");
      break;
    case ELF::EF_MIPS_ARCH_64R2:
      Features.AddFeature("mips64r2");
      break;
    case ELF::EF_MIPS_ARCH_32R6:
      Features.AddFeature("mips32r6");
      break;
    case ELF::EF_MIPS_ARCH_64R6:
      Features.AddFeature("mips64r6");
      break;
    default:
      llvm_unreachable("Unknown EF_MIPS_ARCH value");
    }

    switch (PlatformFlags & ELF::EF_MIPS_MACH) {
    case ELF::EF_MIPS_MACH_NONE:
      // No feature associated with this value.
      break;
    case ELF::EF_MIPS_MACH_OCTEON:
      Features.AddFeature("cnmips");
      break;
    default:
      llvm_unreachable("Unknown EF_MIPS_ARCH value");
    }

    if (PlatformFlags & ELF::EF_MIPS_ARCH_ASE_M16)
      Features.AddFeature("mips16");
    if (PlatformFlags & ELF::EF_MIPS_MICROMIPS)
      Features.AddFeature("micromips");

    return Features;
  }
  default:
    return SubtargetFeatures();
  }
}

} // end namespace llvm
