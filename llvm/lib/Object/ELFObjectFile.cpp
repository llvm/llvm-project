//===- ELFObjectFile.cpp - ELF object file implementation -----------------===//
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
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/ARMAttributeParser.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
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

SubtargetFeatures ELFObjectFileBase::getMIPSFeatures() const {
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

SubtargetFeatures ELFObjectFileBase::getARMFeatures() const {
  SubtargetFeatures Features;
  ARMAttributeParser Attributes;
  std::error_code EC = getBuildAttributes(Attributes);
  if (EC)
    return SubtargetFeatures();

  // both ARMv7-M and R have to support thumb hardware div
  bool isV7 = false;
  if (Attributes.hasAttribute(ARMBuildAttrs::CPU_arch))
    isV7 = Attributes.getAttributeValue(ARMBuildAttrs::CPU_arch)
      == ARMBuildAttrs::v7;

  if (Attributes.hasAttribute(ARMBuildAttrs::CPU_arch_profile)) {
    switch(Attributes.getAttributeValue(ARMBuildAttrs::CPU_arch_profile)) {
    case ARMBuildAttrs::ApplicationProfile:
      Features.AddFeature("aclass");
      break;
    case ARMBuildAttrs::RealTimeProfile:
      Features.AddFeature("rclass");
      if (isV7)
        Features.AddFeature("hwdiv");
      break;
    case ARMBuildAttrs::MicroControllerProfile:
      Features.AddFeature("mclass");
      if (isV7)
        Features.AddFeature("hwdiv");
      break;
    }
  }

  if (Attributes.hasAttribute(ARMBuildAttrs::THUMB_ISA_use)) {
    switch(Attributes.getAttributeValue(ARMBuildAttrs::THUMB_ISA_use)) {
    default:
      break;
    case ARMBuildAttrs::Not_Allowed:
      Features.AddFeature("thumb", false);
      Features.AddFeature("thumb2", false);
      break;
    case ARMBuildAttrs::AllowThumb32:
      Features.AddFeature("thumb2");
      break;
    }
  }

  if (Attributes.hasAttribute(ARMBuildAttrs::FP_arch)) {
    switch(Attributes.getAttributeValue(ARMBuildAttrs::FP_arch)) {
    default:
      break;
    case ARMBuildAttrs::Not_Allowed:
      Features.AddFeature("vfp2", false);
      Features.AddFeature("vfp3", false);
      Features.AddFeature("vfp4", false);
      break;
    case ARMBuildAttrs::AllowFPv2:
      Features.AddFeature("vfp2");
      break;
    case ARMBuildAttrs::AllowFPv3A:
    case ARMBuildAttrs::AllowFPv3B:
      Features.AddFeature("vfp3");
      break;
    case ARMBuildAttrs::AllowFPv4A:
    case ARMBuildAttrs::AllowFPv4B:
      Features.AddFeature("vfp4");
      break;
    }
  }

  if (Attributes.hasAttribute(ARMBuildAttrs::Advanced_SIMD_arch)) {
    switch(Attributes.getAttributeValue(ARMBuildAttrs::Advanced_SIMD_arch)) {
    default:
      break;
    case ARMBuildAttrs::Not_Allowed:
      Features.AddFeature("neon", false);
      Features.AddFeature("fp16", false);
      break;
    case ARMBuildAttrs::AllowNeon:
      Features.AddFeature("neon");
      break;
    case ARMBuildAttrs::AllowNeon2:
      Features.AddFeature("neon");
      Features.AddFeature("fp16");
      break;
    }
  }

  if (Attributes.hasAttribute(ARMBuildAttrs::DIV_use)) {
    switch(Attributes.getAttributeValue(ARMBuildAttrs::DIV_use)) {
    default:
      break;
    case ARMBuildAttrs::DisallowDIV:
      Features.AddFeature("hwdiv", false);
      Features.AddFeature("hwdiv-arm", false);
      break;
    case ARMBuildAttrs::AllowDIVExt:
      Features.AddFeature("hwdiv");
      Features.AddFeature("hwdiv-arm");
      break;
    }
  }

  return Features;
}

SubtargetFeatures ELFObjectFileBase::getFeatures() const {
  switch (getEMachine()) {
  case ELF::EM_MIPS:
    return getMIPSFeatures();
  case ELF::EM_ARM:
    return getARMFeatures();
  default:
    return SubtargetFeatures();
  }
}

// FIXME Encode from a tablegen description or target parser.
void ELFObjectFileBase::setARMSubArch(Triple &TheTriple) const {
  if (TheTriple.getSubArch() != Triple::NoSubArch)
    return;

  ARMAttributeParser Attributes;
  std::error_code EC = getBuildAttributes(Attributes);
  if (EC)
    return;

  std::string Triple;
  // Default to ARM, but use the triple if it's been set.
  if (TheTriple.getArch() == Triple::thumb ||
      TheTriple.getArch() == Triple::thumbeb)
    Triple = "thumb";
  else
    Triple = "arm";

  if (Attributes.hasAttribute(ARMBuildAttrs::CPU_arch)) {
    switch(Attributes.getAttributeValue(ARMBuildAttrs::CPU_arch)) {
    case ARMBuildAttrs::v4:
      Triple += "v4";
      break;
    case ARMBuildAttrs::v4T:
      Triple += "v4t";
      break;
    case ARMBuildAttrs::v5T:
      Triple += "v5t";
      break;
    case ARMBuildAttrs::v5TE:
      Triple += "v5te";
      break;
    case ARMBuildAttrs::v5TEJ:
      Triple += "v5tej";
      break;
    case ARMBuildAttrs::v6:
      Triple += "v6";
      break;
    case ARMBuildAttrs::v6KZ:
      Triple += "v6kz";
      break;
    case ARMBuildAttrs::v6T2:
      Triple += "v6t2";
      break;
    case ARMBuildAttrs::v6K:
      Triple += "v6k";
      break;
    case ARMBuildAttrs::v7:
      Triple += "v7";
      break;
    case ARMBuildAttrs::v6_M:
      Triple += "v6m";
      break;
    case ARMBuildAttrs::v6S_M:
      Triple += "v6sm";
      break;
    case ARMBuildAttrs::v7E_M:
      Triple += "v7em";
      break;
    }
  }
  if (!isLittleEndian())
    Triple += "eb";

  TheTriple.setArchName(Triple);
}
