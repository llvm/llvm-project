//===--- Triple.cpp - Target triple helper class --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/Triple.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/ARMTargetParserCommon.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/TargetParser.h"
#include <cassert>
#include <cstring>
using namespace llvm;

bool Triple::operator==(const Triple &Other) const {
  return Arch == Other.Arch && SubArch == Other.SubArch &&
         Vendor == Other.Vendor && OS == Other.OS &&
         Environment == Other.Environment && ObjectFormat == Other.ObjectFormat;
}

bool Triple::operator<(const Triple &Other) const {
  return std::tie(Arch, SubArch, Vendor, OS, Environment, ObjectFormat, Data) <
         std::tie(Other.Arch, Other.SubArch, Other.Vendor, Other.OS,
                  Other.Environment, Other.ObjectFormat, Other.Data);
}

StringRef Triple::getArchTypeName(ArchType Kind) {
  switch (Kind) {
  case UnknownArch:
    return "unknown";

  case aarch64:
    return "aarch64";
  case aarch64_32:
    return "aarch64_32";
  case aarch64_be:
    return "aarch64_be";
  case amdgpu:
    return "amdgpu";
  case amdil64:
    return "amdil64";
  case amdil:
    return "amdil";
  case arc:
    return "arc";
  case arm:
    return "arm";
  case armeb:
    return "armeb";
  case avr:
    return "avr";
  case bpfeb:
    return "bpfeb";
  case bpfel:
    return "bpfel";
  case csky:
    return "csky";
  case dxil:
    return "dxil";
  case hexagon:
    return "hexagon";
  case hsail64:
    return "hsail64";
  case hsail:
    return "hsail";
  case kalimba:
    return "kalimba";
  case lanai:
    return "lanai";
  case loongarch32:
    return "loongarch32";
  case loongarch64:
    return "loongarch64";
  case m68k:
    return "m68k";
  case mips64:
    return "mips64";
  case mips64el:
    return "mips64el";
  case mips:
    return "mips";
  case mipsel:
    return "mipsel";
  case msp430:
    return "msp430";
  case nvptx64:
    return "nvptx64";
  case nvptx:
    return "nvptx";
  case ppc64:
    return "powerpc64";
  case ppc64le:
    return "powerpc64le";
  case ppc:
    return "powerpc";
  case ppcle:
    return "powerpcle";
  case r600:
    return "r600";
  case renderscript32:
    return "renderscript32";
  case renderscript64:
    return "renderscript64";
  case riscv32:
    return "riscv32";
  case riscv64:
    return "riscv64";
  case riscv32be:
    return "riscv32be";
  case riscv64be:
    return "riscv64be";
  case shave:
    return "shave";
  case sparc:
    return "sparc";
  case sparcel:
    return "sparcel";
  case sparcv9:
    return "sparcv9";
  case spir64:
    return "spir64";
  case spir:
    return "spir";
  case spirv:
    return "spirv";
  case spirv32:
    return "spirv32";
  case spirv64:
    return "spirv64";
  case systemz:
    return "s390x";
  case tce:
    return "tce";
  case tcele:
    return "tcele";
  case tcele64:
    return "tcele64";
  case thumb:
    return "thumb";
  case thumbeb:
    return "thumbeb";
  case ve:
    return "ve";
  case wasm32:
    return "wasm32";
  case wasm64:
    return "wasm64";
  case x86:
    return "i386";
  case x86_64:
    return "x86_64";
  case xcore:
    return "xcore";
  case xtensa:
    return "xtensa";
  }

  llvm_unreachable("Invalid ArchType!");
}

StringRef Triple::getArchName(ArchType Kind, SubArchType SubArch) {
  switch (Kind) {
  case Triple::mips:
    if (SubArch == MipsSubArch_r6)
      return "mipsisa32r6";
    break;
  case Triple::mipsel:
    if (SubArch == MipsSubArch_r6)
      return "mipsisa32r6el";
    break;
  case Triple::mips64:
    if (SubArch == MipsSubArch_r6)
      return "mipsisa64r6";
    break;
  case Triple::mips64el:
    if (SubArch == MipsSubArch_r6)
      return "mipsisa64r6el";
    break;
  case Triple::aarch64:
    if (SubArch == AArch64SubArch_arm64ec)
      return "arm64ec";
    if (SubArch == AArch64SubArch_arm64e)
      return "arm64e";
    if (SubArch == AArch64SubArch_lfi)
      return "aarch64_lfi";
    break;
  case Triple::x86_64:
    if (SubArch == X86_64SubArch_lfi)
      return "x86_64_lfi";
    break;
  case Triple::spirv:
    switch (SubArch) {
    case Triple::SPIRVSubArch_v10:
      return "spirv1.0";
    case Triple::SPIRVSubArch_v11:
      return "spirv1.1";
    case Triple::SPIRVSubArch_v12:
      return "spirv1.2";
    case Triple::SPIRVSubArch_v13:
      return "spirv1.3";
    case Triple::SPIRVSubArch_v14:
      return "spirv1.4";
    case Triple::SPIRVSubArch_v15:
      return "spirv1.5";
    case Triple::SPIRVSubArch_v16:
      return "spirv1.6";
    default:
      break;
    }
    break;
  case Triple::dxil:
    switch (SubArch) {
    case Triple::NoSubArch:
    case Triple::DXILSubArch_v1_0:
      return "dxilv1.0";
    case Triple::DXILSubArch_v1_1:
      return "dxilv1.1";
    case Triple::DXILSubArch_v1_2:
      return "dxilv1.2";
    case Triple::DXILSubArch_v1_3:
      return "dxilv1.3";
    case Triple::DXILSubArch_v1_4:
      return "dxilv1.4";
    case Triple::DXILSubArch_v1_5:
      return "dxilv1.5";
    case Triple::DXILSubArch_v1_6:
      return "dxilv1.6";
    case Triple::DXILSubArch_v1_7:
      return "dxilv1.7";
    case Triple::DXILSubArch_v1_8:
      return "dxilv1.8";
    case Triple::DXILSubArch_v1_9:
      return "dxilv1.9";
    default:
      break;
    }
    break;
  case Triple::amdgpu: {
    if (SubArch < Triple::FirstAMDGPUSubArch ||
        SubArch > Triple::LastAMDGPUSubArch)
      break;

    static const StringLiteral AMDGPUSubArchNames[Triple::LastAMDGPUSubArch -
                                                  Triple::FirstAMDGPUSubArch +
                                                  1] = {
        "amdgpu6",     "amdgpu6.00",  "amdgpu6.01",  "amdgpu6.02",

        "amdgpu7",     "amdgpu7.00",  "amdgpu7.01",  "amdgpu7.02",
        "amdgpu7.03",  "amdgpu7.04",  "amdgpu7.05",

        "amdgpu8",     "amdgpu8.01",  "amdgpu8.02",  "amdgpu8.03",
        "amdgpu8.05",

        "amdgpu8.10",

        "amdgpu9",     "amdgpu9.00",  "amdgpu9.02",  "amdgpu9.04",
        "amdgpu9.06",  "amdgpu9.09",  "amdgpu9.0c",

        "amdgpu9.08",  "amdgpu9.0a",

        "amdgpu9.4",   "amdgpu9.42",  "amdgpu9.50",

        "amdgpu10.1",  "amdgpu10.10", "amdgpu10.11", "amdgpu10.12",
        "amdgpu10.13",

        "amdgpu10.3",  "amdgpu10.30", "amdgpu10.31", "amdgpu10.32",
        "amdgpu10.33", "amdgpu10.34", "amdgpu10.35", "amdgpu10.36",

        "amdgpu11",    "amdgpu11.00", "amdgpu11.01", "amdgpu11.02",
        "amdgpu11.03", "amdgpu11.50", "amdgpu11.51", "amdgpu11.52",
        "amdgpu11.53", "amdgpu11.54",

        "amdgpu11.7",  "amdgpu11.70", "amdgpu11.71", "amdgpu11.72",

        "amdgpu12",    "amdgpu12.00", "amdgpu12.01",

        "amdgpu12.5",  "amdgpu12.50", "amdgpu12.51",

        "amdgpu13",    "amdgpu13.10"};

    return AMDGPUSubArchNames[SubArch - Triple::FirstAMDGPUSubArch];
  }
  default:
    break;
  }
  return getArchTypeName(Kind);
}

StringRef Triple::getArchTypePrefix(ArchType Kind) {
  switch (Kind) {
  default:
    return StringRef();

  case aarch64:
  case aarch64_be:
  case aarch64_32:
    return "aarch64";

  case arc:
    return "arc";

  case arm:
  case armeb:
  case thumb:
  case thumbeb:
    return "arm";

  case avr:
    return "avr";

  case ppc64:
  case ppc64le:
  case ppc:
  case ppcle:
    return "ppc";

  case m68k:
    return "m68k";

  case mips:
  case mipsel:
  case mips64:
  case mips64el:
    return "mips";

  case hexagon:
    return "hexagon";

  // Intrinsics use amdgcn prefix.
  case amdgpu:
    return "amdgcn";
  case r600:
    return "r600";

  case bpfel:
  case bpfeb:
    return "bpf";

  case sparcv9:
  case sparcel:
  case sparc:
    return "sparc";

  case systemz:
    return "s390";

  case x86:
  case x86_64:
    return "x86";

  case xcore:
    return "xcore";

  // NVPTX intrinsics are namespaced under nvvm.
  case nvptx:
    return "nvvm";
  case nvptx64:
    return "nvvm";

  case amdil:
  case amdil64:
    return "amdil";

  case hsail:
  case hsail64:
    return "hsail";

  case spir:
  case spir64:
    return "spir";

  case spirv:
  case spirv32:
  case spirv64:
    return "spv";

  case kalimba:
    return "kalimba";
  case lanai:
    return "lanai";
  case shave:
    return "shave";
  case wasm32:
  case wasm64:
    return "wasm";

  case riscv32:
  case riscv64:
  case riscv32be:
  case riscv64be:
    return "riscv";

  case ve:
    return "ve";
  case csky:
    return "csky";

  case loongarch32:
  case loongarch64:
    return "loongarch";

  case dxil:
    return "dx";

  case xtensa:
    return "xtensa";
  }
}

StringRef Triple::getVendorTypeName(VendorType Kind) {
  switch (Kind) {
  case UnknownVendor:
    return "unknown";
#define TRIPLE_VENDOR(Enum, Name)                                              \
  case Enum:                                                                   \
    return Name;
#include "llvm/TargetParser/TripleName.def"
  }

  llvm_unreachable("Invalid VendorType!");
}

StringRef Triple::getOSTypeName(OSType Kind) {
  switch (Kind) {
  case UnknownOS:
    return "unknown";
#define TRIPLE_OS(Enum, Name)                                                  \
  case Enum:                                                                   \
    return Name;
#include "llvm/TargetParser/TripleName.def"
  }

  llvm_unreachable("Invalid OSType");
}

StringRef Triple::getEnvironmentTypeName(EnvironmentType Kind) {
  switch (Kind) {
  case UnknownEnvironment:
    return "unknown";
#define TRIPLE_ENV(Enum, Name)                                                 \
  case Enum:                                                                   \
    return Name;
#include "llvm/TargetParser/TripleName.def"
  }

  llvm_unreachable("Invalid EnvironmentType!");
}

StringRef Triple::getObjectFormatTypeName(ObjectFormatType Kind) {
  switch (Kind) {
  case UnknownObjectFormat:
    return "";
  case COFF:
    return "coff";
  case ELF:
    return "elf";
  case GOFF:
    return "goff";
  case MachO:
    return "macho";
  case Wasm:
    return "wasm";
  case XCOFF:
    return "xcoff";
  case DXContainer:
    return "dxcontainer";
  case SPIRV:
    return "spirv";
  }
  llvm_unreachable("unknown object format type");
}

static Triple::ArchType parseBPFArch(StringRef ArchName) {
  if (ArchName == "bpf") {
    if (sys::IsLittleEndianHost)
      return Triple::bpfel;
    else
      return Triple::bpfeb;
  } else if (ArchName == "bpf_be" || ArchName == "bpfeb") {
    return Triple::bpfeb;
  } else if (ArchName == "bpf_le" || ArchName == "bpfel") {
    return Triple::bpfel;
  } else {
    return Triple::UnknownArch;
  }
}

Triple::ArchType Triple::getArchTypeForLLVMName(StringRef Name) {
  Triple::ArchType BPFArch(parseBPFArch(Name));
  return StringSwitch<Triple::ArchType>(Name)
      .Case("aarch64", aarch64)
      .Case("aarch64_be", aarch64_be)
      .Case("aarch64_32", aarch64_32)
      .Case("arc", arc)
      .Case("arm64", aarch64) // "arm64" is an alias for "aarch64"
      .Case("arm64_32", aarch64_32)
      .Case("arm", arm)
      .Case("armeb", armeb)
      .Case("avr", avr)
      .StartsWith("bpf", BPFArch)
      .Case("m68k", m68k)
      .Case("mips", mips)
      .Case("mipsel", mipsel)
      .Case("mips64", mips64)
      .Case("mips64el", mips64el)
      .Case("msp430", msp430)
      .Case("ppc64", ppc64)
      .Case("ppc32", ppc)
      .Case("ppc", ppc)
      .Case("ppc32le", ppcle)
      .Case("ppcle", ppcle)
      .Case("ppc64le", ppc64le)
      .Case("amdgpu", amdgpu)
      .Case("amdgcn", amdgpu) // Legacy name
      .Case("r600", r600)
      .Case("riscv32", riscv32)
      .Case("riscv64", riscv64)
      .Case("riscv32be", riscv32be)
      .Case("riscv64be", riscv64be)
      .Case("hexagon", hexagon)
      .Case("sparc", sparc)
      .Case("sparcel", sparcel)
      .Case("sparcv9", sparcv9)
      .Case("s390x", systemz)
      .Case("systemz", systemz)
      .Case("tce", tce)
      .Case("tcele", tcele)
      .Case("tcele64", tcele64)
      .Case("thumb", thumb)
      .Case("thumbeb", thumbeb)
      .Case("x86", x86)
      .Case("i386", x86)
      .Case("x86-64", x86_64)
      .Case("xcore", xcore)
      .Case("nvptx", nvptx)
      .Case("nvptx64", nvptx64)
      .Case("amdil", amdil)
      .Case("amdil64", amdil64)
      .Case("hsail", hsail)
      .Case("hsail64", hsail64)
      .Case("spir", spir)
      .Case("spir64", spir64)
      .Case("spirv", spirv)
      .Case("spirv32", spirv32)
      .Case("spirv64", spirv64)
      .Case("kalimba", kalimba)
      .Case("lanai", lanai)
      .Case("shave", shave)
      .Case("wasm32", wasm32)
      .Case("wasm64", wasm64)
      .Case("renderscript32", renderscript32)
      .Case("renderscript64", renderscript64)
      .Case("ve", ve)
      .Case("csky", csky)
      .Case("loongarch32", loongarch32)
      .Case("loongarch64", loongarch64)
      .Case("dxil", dxil)
      .Case("xtensa", xtensa)
      .Default(UnknownArch);
}

static Triple::ArchType parseARMArch(StringRef ArchName) {
  ARM::ISAKind ISA = ARM::parseArchISA(ArchName);
  ARM::EndianKind ENDIAN = ARM::parseArchEndian(ArchName);

  Triple::ArchType arch = Triple::UnknownArch;
  switch (ENDIAN) {
  case ARM::EndianKind::LITTLE: {
    switch (ISA) {
    case ARM::ISAKind::ARM:
      arch = Triple::arm;
      break;
    case ARM::ISAKind::THUMB:
      arch = Triple::thumb;
      break;
    case ARM::ISAKind::AARCH64:
      arch = Triple::aarch64;
      break;
    case ARM::ISAKind::INVALID:
      break;
    }
    break;
  }
  case ARM::EndianKind::BIG: {
    switch (ISA) {
    case ARM::ISAKind::ARM:
      arch = Triple::armeb;
      break;
    case ARM::ISAKind::THUMB:
      arch = Triple::thumbeb;
      break;
    case ARM::ISAKind::AARCH64:
      arch = Triple::aarch64_be;
      break;
    case ARM::ISAKind::INVALID:
      break;
    }
    break;
  }
  case ARM::EndianKind::INVALID: {
    break;
  }
  }

  ArchName = ARM::getCanonicalArchName(ArchName);
  if (ArchName.empty())
    return Triple::UnknownArch;

  // Thumb only exists in v4+
  if (ISA == ARM::ISAKind::THUMB &&
      (ArchName.starts_with("v2") || ArchName.starts_with("v3")))
    return Triple::UnknownArch;

  // Thumb only for v6m
  ARM::ProfileKind Profile = ARM::parseArchProfile(ArchName);
  unsigned Version = ARM::parseArchVersion(ArchName);
  if (Profile == ARM::ProfileKind::M && Version == 6) {
    if (ENDIAN == ARM::EndianKind::BIG)
      return Triple::thumbeb;
    else
      return Triple::thumb;
  }

  return arch;
}

Triple::ArchType Triple::parseArch(StringRef ArchName) {
  auto AT =
      StringSwitch<Triple::ArchType>(ArchName)
          .Cases({"i386", "i486", "i586", "i686"}, Triple::x86)
          // FIXME: Do we need to support these?
          .Cases({"i786", "i886", "i986"}, Triple::x86)
          .Cases({"amd64", "x86_64", "x86_64h", "x86_64_lfi"}, Triple::x86_64)
          .Cases({"powerpc", "powerpcspe", "ppc", "ppc32"}, Triple::ppc)
          .Cases({"powerpcle", "ppcle", "ppc32le"}, Triple::ppcle)
          .Cases({"powerpc64", "ppu", "ppc64"}, Triple::ppc64)
          .Cases({"powerpc64le", "ppc64le"}, Triple::ppc64le)
          .Case("xscale", Triple::arm)
          .Case("xscaleeb", Triple::armeb)
          .Case("aarch64", Triple::aarch64)
          .Case("aarch64_be", Triple::aarch64_be)
          .Case("aarch64_32", Triple::aarch64_32)
          .Case("aarch64_lfi", Triple::aarch64)
          .Case("arc", Triple::arc)
          .Case("arm64", Triple::aarch64)
          .Case("arm64_32", Triple::aarch64_32)
          .Case("arm64e", Triple::aarch64)
          .Case("arm64ec", Triple::aarch64)
          .Case("arm", Triple::arm)
          .Case("armeb", Triple::armeb)
          .Case("thumb", Triple::thumb)
          .Case("thumbeb", Triple::thumbeb)
          .Case("avr", Triple::avr)
          .Case("m68k", Triple::m68k)
          .Case("msp430", Triple::msp430)
          .Cases({"mips", "mipseb", "mipsallegrex", "mipsisa32r6", "mipsr6"},
                 Triple::mips)
          .Cases({"mipsel", "mipsallegrexel", "mipsisa32r6el", "mipsr6el"},
                 Triple::mipsel)
          .Cases({"mips64", "mips64eb", "mipsn32", "mipsisa64r6", "mips64r6",
                  "mipsn32r6"},
                 Triple::mips64)
          .Cases({"mips64el", "mipsn32el", "mipsisa64r6el", "mips64r6el",
                  "mipsn32r6el"},
                 Triple::mips64el)
          .StartsWith("amdgpu", Triple::amdgpu)
          .Case("amdgcn", Triple::amdgpu)
          .Case("r600", Triple::r600)
          .Case("riscv32", Triple::riscv32)
          .Case("riscv64", Triple::riscv64)
          .Case("riscv32be", Triple::riscv32be)
          .Case("riscv64be", Triple::riscv64be)
          .Case("hexagon", Triple::hexagon)
          .Cases({"s390x", "systemz"}, Triple::systemz)
          .Case("sparc", Triple::sparc)
          .Case("sparcel", Triple::sparcel)
          .Cases({"sparcv9", "sparc64"}, Triple::sparcv9)
          .Case("tce", Triple::tce)
          .Case("tcele", Triple::tcele)
          .Case("tcele64", Triple::tcele64)
          .Case("xcore", Triple::xcore)
          .Case("nvptx", Triple::nvptx)
          .Case("nvptx64", Triple::nvptx64)
          .Case("amdil", Triple::amdil)
          .Case("amdil64", Triple::amdil64)
          .Case("hsail", Triple::hsail)
          .Case("hsail64", Triple::hsail64)
          .Case("spir", Triple::spir)
          .Case("spir64", Triple::spir64)
          .Cases({"spirv", "spirv1.5", "spirv1.6"}, Triple::spirv)
          .Cases({"spirv32", "spirv32v1.0", "spirv32v1.1", "spirv32v1.2",
                  "spirv32v1.3", "spirv32v1.4", "spirv32v1.5", "spirv32v1.6"},
                 Triple::spirv32)
          .Cases({"spirv64", "spirv64v1.0", "spirv64v1.1", "spirv64v1.2",
                  "spirv64v1.3", "spirv64v1.4", "spirv64v1.5", "spirv64v1.6"},
                 Triple::spirv64)
          .StartsWith("kalimba", Triple::kalimba)
          .Case("lanai", Triple::lanai)
          .Case("renderscript32", Triple::renderscript32)
          .Case("renderscript64", Triple::renderscript64)
          .Case("shave", Triple::shave)
          .Case("ve", Triple::ve)
          .Case("wasm32", Triple::wasm32)
          .Case("wasm64", Triple::wasm64)
          .Case("csky", Triple::csky)
          .Case("loongarch32", Triple::loongarch32)
          .Case("loongarch64", Triple::loongarch64)
          .Cases({"dxil", "dxilv1.0", "dxilv1.1", "dxilv1.2", "dxilv1.3",
                  "dxilv1.4", "dxilv1.5", "dxilv1.6", "dxilv1.7", "dxilv1.8",
                  "dxilv1.9"},
                 Triple::dxil)
          .Case("xtensa", Triple::xtensa)
          .Default(Triple::UnknownArch);

  // Some architectures require special parsing logic just to compute the
  // ArchType result.
  if (AT == Triple::UnknownArch) {
    if (ArchName.starts_with("arm") || ArchName.starts_with("thumb") ||
        ArchName.starts_with("aarch64"))
      return parseARMArch(ArchName);
    if (ArchName.starts_with("bpf"))
      return parseBPFArch(ArchName);
  }

  return AT;
}

static Triple::VendorType parseVendor(StringRef VendorName) {
  return StringSwitch<Triple::VendorType>(VendorName)
#define TRIPLE_VENDOR(Enum, Name) .Case(Name, Triple::Enum)
#define TRIPLE_VENDOR_ALIAS(Enum, AliasName) .Case(AliasName, Triple::Enum)
#include "llvm/TargetParser/TripleName.def"
      .Default(Triple::UnknownVendor);
}

static Triple::OSType parseOS(StringRef OSName) {
  return StringSwitch<Triple::OSType>(OSName)
#define TRIPLE_OS(Enum, Name) .StartsWith(Name, Triple::Enum)
#define TRIPLE_OS_ALIAS(Enum, AliasName) .StartsWith(AliasName, Triple::Enum)
#include "llvm/TargetParser/TripleName.def"
      .Default(Triple::UnknownOS);
}

static Triple::EnvironmentType parseEnvironment(StringRef EnvironmentName) {
  return StringSwitch<Triple::EnvironmentType>(EnvironmentName)
#define TRIPLE_ENV(Enum, Name) .StartsWith(Name, Triple::Enum)
#include "llvm/TargetParser/TripleName.def"
      .Default(Triple::UnknownEnvironment);
}

static Triple::ObjectFormatType parseFormat(StringRef EnvironmentName) {
  return StringSwitch<Triple::ObjectFormatType>(EnvironmentName)
      // "xcoff" must come before "coff" because of the order-dependendent
      // pattern matching.
      .EndsWith("xcoff", Triple::XCOFF)
      .EndsWith("coff", Triple::COFF)
      .EndsWith("elf", Triple::ELF)
      .EndsWith("goff", Triple::GOFF)
      .EndsWith("macho", Triple::MachO)
      .EndsWith("wasm", Triple::Wasm)
      .EndsWith("spirv", Triple::SPIRV)
      .Default(Triple::UnknownObjectFormat);
}

Triple::SubArchType Triple::parseSubArch(StringRef SubArchName) {
  if (SubArchName.starts_with("mips") &&
      (SubArchName.ends_with("r6el") || SubArchName.ends_with("r6")))
    return Triple::MipsSubArch_r6;

  if (SubArchName == "powerpcspe")
    return Triple::PPCSubArch_spe;

  if (SubArchName == "arm64e")
    return Triple::AArch64SubArch_arm64e;

  if (SubArchName == "arm64ec")
    return Triple::AArch64SubArch_arm64ec;

  if (SubArchName == "aarch64_lfi")
    return Triple::AArch64SubArch_lfi;

  if (SubArchName == "x86_64_lfi")
    return Triple::X86_64SubArch_lfi;

  if (SubArchName.starts_with("spirv"))
    return StringSwitch<Triple::SubArchType>(SubArchName)
        .EndsWith("v1.0", Triple::SPIRVSubArch_v10)
        .EndsWith("v1.1", Triple::SPIRVSubArch_v11)
        .EndsWith("v1.2", Triple::SPIRVSubArch_v12)
        .EndsWith("v1.3", Triple::SPIRVSubArch_v13)
        .EndsWith("v1.4", Triple::SPIRVSubArch_v14)
        .EndsWith("v1.5", Triple::SPIRVSubArch_v15)
        .EndsWith("v1.6", Triple::SPIRVSubArch_v16)
        .Default(Triple::NoSubArch);

  if (SubArchName.starts_with("dxil"))
    return StringSwitch<Triple::SubArchType>(SubArchName)
        .EndsWith("v1.0", Triple::DXILSubArch_v1_0)
        .EndsWith("v1.1", Triple::DXILSubArch_v1_1)
        .EndsWith("v1.2", Triple::DXILSubArch_v1_2)
        .EndsWith("v1.3", Triple::DXILSubArch_v1_3)
        .EndsWith("v1.4", Triple::DXILSubArch_v1_4)
        .EndsWith("v1.5", Triple::DXILSubArch_v1_5)
        .EndsWith("v1.6", Triple::DXILSubArch_v1_6)
        .EndsWith("v1.7", Triple::DXILSubArch_v1_7)
        .EndsWith("v1.8", Triple::DXILSubArch_v1_8)
        .EndsWith("v1.9", Triple::DXILSubArch_v1_9)
        .Default(Triple::NoSubArch);

  if (SubArchName.consume_front("amdgpu")) {
    return StringSwitch<Triple::SubArchType>(SubArchName)
        .Case("6", Triple::AMDGPUSubArch6)
        .Case("6.00", Triple::AMDGPUSubArch600)
        .Case("6.01", Triple::AMDGPUSubArch601)
        .Case("6.02", Triple::AMDGPUSubArch602)
        .Case("7", Triple::AMDGPUSubArch7)
        .Case("7.00", Triple::AMDGPUSubArch700)
        .Case("7.01", Triple::AMDGPUSubArch701)
        .Case("7.02", Triple::AMDGPUSubArch702)
        .Case("7.03", Triple::AMDGPUSubArch703)
        .Case("7.04", Triple::AMDGPUSubArch704)
        .Case("7.05", Triple::AMDGPUSubArch705)
        .Case("8", Triple::AMDGPUSubArch8)
        .Case("8.01", Triple::AMDGPUSubArch801)
        .Case("8.02", Triple::AMDGPUSubArch802)
        .Case("8.03", Triple::AMDGPUSubArch803)
        .Case("8.05", Triple::AMDGPUSubArch805)
        .Case("8.10", Triple::AMDGPUSubArch810)
        .Case("9", Triple::AMDGPUSubArch9)
        .Case("9.00", Triple::AMDGPUSubArch900)
        .Case("9.02", Triple::AMDGPUSubArch902)
        .Case("9.04", Triple::AMDGPUSubArch904)
        .Case("9.06", Triple::AMDGPUSubArch906)
        .Case("9.08", Triple::AMDGPUSubArch908)
        .Case("9.09", Triple::AMDGPUSubArch909)
        .Case("9.0a", Triple::AMDGPUSubArch90A)
        .Case("9.0c", Triple::AMDGPUSubArch90C)
        .Case("9.4", Triple::AMDGPUSubArch9_4)
        .Case("9.5", Triple::AMDGPUSubArch9_4)
        .Case("9.42", Triple::AMDGPUSubArch942)
        .Case("9.50", Triple::AMDGPUSubArch950)
        .Case("10", Triple::AMDGPUSubArch10_1)
        .Case("10.1", Triple::AMDGPUSubArch10_1)
        .Case("10.10", Triple::AMDGPUSubArch1010)
        .Case("10.11", Triple::AMDGPUSubArch1011)
        .Case("10.12", Triple::AMDGPUSubArch1012)
        .Case("10.13", Triple::AMDGPUSubArch1013)
        .Case("10.3", Triple::AMDGPUSubArch10_3)
        .Case("10.30", Triple::AMDGPUSubArch1030)
        .Case("10.31", Triple::AMDGPUSubArch1031)
        .Case("10.32", Triple::AMDGPUSubArch1032)
        .Case("10.33", Triple::AMDGPUSubArch1033)
        .Case("10.34", Triple::AMDGPUSubArch1034)
        .Case("10.35", Triple::AMDGPUSubArch1035)
        .Case("10.36", Triple::AMDGPUSubArch1036)
        .Case("11", Triple::AMDGPUSubArch11)
        .Case("11.00", Triple::AMDGPUSubArch1100)
        .Case("11.01", Triple::AMDGPUSubArch1101)
        .Case("11.02", Triple::AMDGPUSubArch1102)
        .Case("11.03", Triple::AMDGPUSubArch1103)
        .Case("11.50", Triple::AMDGPUSubArch1150)
        .Case("11.51", Triple::AMDGPUSubArch1151)
        .Case("11.52", Triple::AMDGPUSubArch1152)
        .Case("11.53", Triple::AMDGPUSubArch1153)
        .Case("11.54", Triple::AMDGPUSubArch1154)
        .Case("11.7", Triple::AMDGPUSubArch11_7)
        .Case("11.70", Triple::AMDGPUSubArch1170)
        .Case("11.71", Triple::AMDGPUSubArch1171)
        .Case("11.72", Triple::AMDGPUSubArch1172)
        .Case("12", Triple::AMDGPUSubArch12)
        .Case("12.00", Triple::AMDGPUSubArch1200)
        .Case("12.01", Triple::AMDGPUSubArch1201)
        .Case("12.5", Triple::AMDGPUSubArch12_5)
        .Case("12.50", Triple::AMDGPUSubArch1250)
        .Case("12.51", Triple::AMDGPUSubArch1251)
        .Case("13", Triple::AMDGPUSubArch13)
        .Case("13.10", Triple::AMDGPUSubArch1310)
        .Default(Triple::NoSubArch);
  }

  StringRef ARMSubArch = ARM::getCanonicalArchName(SubArchName);

  // For now, this is the small part. Early return.
  if (ARMSubArch.empty())
    return StringSwitch<Triple::SubArchType>(SubArchName)
        .EndsWith("kalimba3", Triple::KalimbaSubArch_v3)
        .EndsWith("kalimba4", Triple::KalimbaSubArch_v4)
        .EndsWith("kalimba5", Triple::KalimbaSubArch_v5)
        .Default(Triple::NoSubArch);

  // ARM sub arch.
  switch (ARM::parseArch(ARMSubArch)) {
  case ARM::ArchKind::ARMV4:
    return Triple::NoSubArch;
  case ARM::ArchKind::ARMV4T:
    return Triple::ARMSubArch_v4t;
  case ARM::ArchKind::ARMV5T:
    return Triple::ARMSubArch_v5;
  case ARM::ArchKind::ARMV5TE:
  case ARM::ArchKind::IWMMXT:
  case ARM::ArchKind::IWMMXT2:
  case ARM::ArchKind::XSCALE:
  case ARM::ArchKind::ARMV5TEJ:
    return Triple::ARMSubArch_v5te;
  case ARM::ArchKind::ARMV6:
    return Triple::ARMSubArch_v6;
  case ARM::ArchKind::ARMV6K:
  case ARM::ArchKind::ARMV6KZ:
    return Triple::ARMSubArch_v6k;
  case ARM::ArchKind::ARMV6T2:
    return Triple::ARMSubArch_v6t2;
  case ARM::ArchKind::ARMV6M:
    return Triple::ARMSubArch_v6m;
  case ARM::ArchKind::ARMV7A:
  case ARM::ArchKind::ARMV7R:
    return Triple::ARMSubArch_v7;
  case ARM::ArchKind::ARMV7VE:
    return Triple::ARMSubArch_v7ve;
  case ARM::ArchKind::ARMV7K:
    return Triple::ARMSubArch_v7k;
  case ARM::ArchKind::ARMV7M:
    return Triple::ARMSubArch_v7m;
  case ARM::ArchKind::ARMV7S:
    return Triple::ARMSubArch_v7s;
  case ARM::ArchKind::ARMV7EM:
    return Triple::ARMSubArch_v7em;
  case ARM::ArchKind::ARMV8A:
    return Triple::ARMSubArch_v8;
  case ARM::ArchKind::ARMV8_1A:
    return Triple::ARMSubArch_v8_1a;
  case ARM::ArchKind::ARMV8_2A:
    return Triple::ARMSubArch_v8_2a;
  case ARM::ArchKind::ARMV8_3A:
    return Triple::ARMSubArch_v8_3a;
  case ARM::ArchKind::ARMV8_4A:
    return Triple::ARMSubArch_v8_4a;
  case ARM::ArchKind::ARMV8_5A:
    return Triple::ARMSubArch_v8_5a;
  case ARM::ArchKind::ARMV8_6A:
    return Triple::ARMSubArch_v8_6a;
  case ARM::ArchKind::ARMV8_7A:
    return Triple::ARMSubArch_v8_7a;
  case ARM::ArchKind::ARMV8_8A:
    return Triple::ARMSubArch_v8_8a;
  case ARM::ArchKind::ARMV8_9A:
    return Triple::ARMSubArch_v8_9a;
  case ARM::ArchKind::ARMV9A:
    return Triple::ARMSubArch_v9;
  case ARM::ArchKind::ARMV9_1A:
    return Triple::ARMSubArch_v9_1a;
  case ARM::ArchKind::ARMV9_2A:
    return Triple::ARMSubArch_v9_2a;
  case ARM::ArchKind::ARMV9_3A:
    return Triple::ARMSubArch_v9_3a;
  case ARM::ArchKind::ARMV9_4A:
    return Triple::ARMSubArch_v9_4a;
  case ARM::ArchKind::ARMV9_5A:
    return Triple::ARMSubArch_v9_5a;
  case ARM::ArchKind::ARMV9_6A:
    return Triple::ARMSubArch_v9_6a;
  case ARM::ArchKind::ARMV9_7A:
    return Triple::ARMSubArch_v9_7a;
  case ARM::ArchKind::ARMV8R:
    return Triple::ARMSubArch_v8r;
  case ARM::ArchKind::ARMV8MBaseline:
    return Triple::ARMSubArch_v8m_baseline;
  case ARM::ArchKind::ARMV8MMainline:
    return Triple::ARMSubArch_v8m_mainline;
  case ARM::ArchKind::ARMV8_1MMainline:
    return Triple::ARMSubArch_v8_1m_mainline;
  default:
    return Triple::NoSubArch;
  }
}

static Triple::ObjectFormatType getDefaultFormat(const Triple &T) {
  switch (T.getArch()) {
  case Triple::UnknownArch:
  case Triple::aarch64:
  case Triple::aarch64_32:
  case Triple::arm:
  case Triple::thumb:
  case Triple::x86:
  case Triple::x86_64:
    switch (T.getOS()) {
    case Triple::Win32:
    case Triple::UEFI:
      return Triple::COFF;
    default:
      return T.isOSDarwin() ? Triple::MachO : Triple::ELF;
    }
  case Triple::aarch64_be:
  case Triple::amdgpu:
  case Triple::amdil64:
  case Triple::amdil:
  case Triple::arc:
  case Triple::armeb:
  case Triple::avr:
  case Triple::bpfeb:
  case Triple::bpfel:
  case Triple::csky:
  case Triple::hexagon:
  case Triple::hsail64:
  case Triple::hsail:
  case Triple::kalimba:
  case Triple::lanai:
  case Triple::loongarch32:
  case Triple::loongarch64:
  case Triple::m68k:
  case Triple::mips64:
  case Triple::mips64el:
  case Triple::mips:
  case Triple::msp430:
  case Triple::nvptx64:
  case Triple::nvptx:
  case Triple::ppc64le:
  case Triple::ppcle:
  case Triple::r600:
  case Triple::renderscript32:
  case Triple::renderscript64:
  case Triple::riscv32:
  case Triple::riscv64:
  case Triple::riscv32be:
  case Triple::riscv64be:
  case Triple::shave:
  case Triple::sparc:
  case Triple::sparcel:
  case Triple::sparcv9:
  case Triple::spir64:
  case Triple::spir:
  case Triple::tce:
  case Triple::tcele:
  case Triple::tcele64:
  case Triple::thumbeb:
  case Triple::ve:
  case Triple::xcore:
  case Triple::xtensa:
    return Triple::ELF;

  case Triple::mipsel:
    if (T.isOSWindows())
      return Triple::COFF;
    return Triple::ELF;

  case Triple::ppc64:
  case Triple::ppc:
    if (T.isOSAIX())
      return Triple::XCOFF;
    if (T.isOSDarwin())
      return Triple::MachO;
    return Triple::ELF;

  case Triple::systemz:
    if (T.isOSzOS())
      return Triple::GOFF;
    return Triple::ELF;

  case Triple::wasm32:
  case Triple::wasm64:
    return Triple::Wasm;

  case Triple::spirv:
  case Triple::spirv32:
  case Triple::spirv64:
    return Triple::SPIRV;

  case Triple::dxil:
    return Triple::DXContainer;
  }
  llvm_unreachable("unknown architecture");
}

/// Construct a triple from the string representation provided.
///
/// This stores the string representation and parses the various pieces into
/// enum members.
Triple::Triple(std::string &&Str) : Data(std::move(Str)) {
  // Do minimal parsing by hand here.
  SmallVector<StringRef, 4> Components;
  StringRef(Data).split(Components, '-', /*MaxSplit*/ 3);
  if (Components.size() > 0) {
    Arch = parseArch(Components[0]);
    SubArch = parseSubArch(Components[0]);
    if (Components.size() > 1) {
      Vendor = parseVendor(Components[1]);
      if (Components.size() > 2) {
        OS = parseOS(Components[2]);
        if (Components.size() > 3) {
          Environment = parseEnvironment(Components[3]);
          ObjectFormat = parseFormat(Components[3]);
        }
      }
    } else {
      Environment =
          StringSwitch<Triple::EnvironmentType>(Components[0])
              .StartsWith("mipsn32", Triple::GNUABIN32)
              .StartsWith("mips64", Triple::GNUABI64)
              .StartsWith("mipsisa64", Triple::GNUABI64)
              .StartsWith("mipsisa32", Triple::GNU)
              .Cases({"mips", "mipsel", "mipsr6", "mipsr6el"}, Triple::GNU)
              .Default(UnknownEnvironment);
    }
  }
  if (ObjectFormat == UnknownObjectFormat)
    ObjectFormat = getDefaultFormat(*this);
}

Triple::Triple(const Twine &Str) : Triple(Str.str()) {}

/// Construct a triple from string representations of the architecture,
/// vendor, and OS.
///
/// This joins each argument into a canonical string representation and parses
/// them into enum members. It leaves the environment unknown and omits it from
/// the string representation.
Triple::Triple(const Twine &ArchStr, const Twine &VendorStr, const Twine &OSStr)
    : Data((ArchStr + Twine('-') + VendorStr + Twine('-') + OSStr).str()),
      Arch(parseArch(ArchStr.str())), SubArch(parseSubArch(ArchStr.str())),
      Vendor(parseVendor(VendorStr.str())), OS(parseOS(OSStr.str())),
      Environment(), ObjectFormat(Triple::UnknownObjectFormat) {
  ObjectFormat = getDefaultFormat(*this);
}

/// Construct a triple from string representations of the architecture,
/// vendor, OS, and environment.
///
/// This joins each argument into a canonical string representation and parses
/// them into enum members.
Triple::Triple(const Twine &ArchStr, const Twine &VendorStr, const Twine &OSStr,
               const Twine &EnvironmentStr)
    : Data((ArchStr + Twine('-') + VendorStr + Twine('-') + OSStr + Twine('-') +
            EnvironmentStr)
               .str()),
      Arch(parseArch(ArchStr.str())), SubArch(parseSubArch(ArchStr.str())),
      Vendor(parseVendor(VendorStr.str())), OS(parseOS(OSStr.str())),
      Environment(parseEnvironment(EnvironmentStr.str())),
      ObjectFormat(parseFormat(EnvironmentStr.str())) {
  if (ObjectFormat == Triple::UnknownObjectFormat)
    ObjectFormat = getDefaultFormat(*this);
}

Triple::Triple(ArchType A, SubArchType SA, VendorType V, OSType OS)
    : Data((getArchName(A, SA) + Twine('-') + getVendorTypeName(V) +
            Twine('-') + getOSTypeName(OS))
               .str()),
      Arch(A), SubArch(SA), Vendor(V), OS(OS),
      ObjectFormat(getDefaultFormat(*this)) {}

Triple::Triple(ArchType A, SubArchType SA, VendorType V, OSType OS,
               EnvironmentType E)
    : Data((getArchName(A, SA) + Twine('-') + getVendorTypeName(V) +
            Twine('-') + getOSTypeName(OS) + Twine('-') +
            getEnvironmentTypeName(E))
               .str()),
      Arch(A), SubArch(SA), Vendor(V), OS(OS), Environment(E),
      ObjectFormat(getDefaultFormat(*this)) {}

Triple::Triple(ArchType A, SubArchType SA, VendorType V, OSType OS,
               EnvironmentType E, ObjectFormatType OF)
    : Data((getArchName(A, SA) + Twine('-') + getVendorTypeName(V) +
            Twine('-') + getOSTypeName(OS) + Twine('-') +
            getEnvironmentTypeName(E) + Twine('-') +
            getObjectFormatTypeName(OF))
               .str()),
      Arch(A), SubArch(SA), Vendor(V), OS(OS), Environment(E),
      ObjectFormat(OF) {}

static VersionTuple parseVersionFromName(StringRef Name);

static StringRef getDXILArchNameFromShaderModel(StringRef ShaderModelStr) {
  VersionTuple Ver =
      parseVersionFromName(ShaderModelStr.drop_front(strlen("shadermodel")));
  // Default DXIL minor version when Shader Model version is anything other
  // than 6.[0...9] or 6.x (which translates to latest current SM version)
  const unsigned SMMajor = 6;
  if (!Ver.empty()) {
    if (Ver.getMajor() == SMMajor) {
      if (std::optional<unsigned> SMMinor = Ver.getMinor()) {
        switch (*SMMinor) {
        case 0:
          return Triple::getArchName(Triple::dxil, Triple::DXILSubArch_v1_0);
        case 1:
          return Triple::getArchName(Triple::dxil, Triple::DXILSubArch_v1_1);
        case 2:
          return Triple::getArchName(Triple::dxil, Triple::DXILSubArch_v1_2);
        case 3:
          return Triple::getArchName(Triple::dxil, Triple::DXILSubArch_v1_3);
        case 4:
          return Triple::getArchName(Triple::dxil, Triple::DXILSubArch_v1_4);
        case 5:
          return Triple::getArchName(Triple::dxil, Triple::DXILSubArch_v1_5);
        case 6:
          return Triple::getArchName(Triple::dxil, Triple::DXILSubArch_v1_6);
        case 7:
          return Triple::getArchName(Triple::dxil, Triple::DXILSubArch_v1_7);
        case 8:
          return Triple::getArchName(Triple::dxil, Triple::DXILSubArch_v1_8);
        case 9:
          return Triple::getArchName(Triple::dxil, Triple::DXILSubArch_v1_9);
        default:
          report_fatal_error("Unsupported Shader Model version", false);
        }
      }
    }
  } else {
    // Special case: DXIL minor version is set to LatestCurrentDXILMinor for
    // shadermodel6.x is
    if (ShaderModelStr == "shadermodel6.x") {
      return Triple::getArchName(Triple::dxil, Triple::LatestDXILSubArch);
    }
  }
  // DXIL version corresponding to Shader Model version other than 6.Minor
  // is 1.0
  return Triple::getArchName(Triple::dxil, Triple::DXILSubArch_v1_0);
}

std::string Triple::normalize(StringRef Str, CanonicalForm Form) {
  bool IsMinGW32 = false;
  bool IsCygwin = false;

  // Parse into components.
  SmallVector<StringRef, 4> Components;
  Str.split(Components, '-');

  // If the first component corresponds to a known architecture, preferentially
  // use it for the architecture.  If the second component corresponds to a
  // known vendor, preferentially use it for the vendor, etc.  This avoids silly
  // component movement when a component parses as (eg) both a valid arch and a
  // valid os.
  ArchType Arch = UnknownArch;
  if (Components.size() > 0)
    Arch = parseArch(Components[0]);
  VendorType Vendor = UnknownVendor;
  if (Components.size() > 1)
    Vendor = parseVendor(Components[1]);
  OSType OS = UnknownOS;
  if (Components.size() > 2) {
    OS = parseOS(Components[2]);
    IsCygwin = Components[2].starts_with("cygwin") ||
               Components[2].starts_with("msys");
    IsMinGW32 = Components[2].starts_with("mingw");
  }
  EnvironmentType Environment = UnknownEnvironment;
  if (Components.size() > 3)
    Environment = parseEnvironment(Components[3]);
  ObjectFormatType ObjectFormat = UnknownObjectFormat;
  if (Components.size() > 4)
    ObjectFormat = parseFormat(Components[4]);

  // Note which components are already in their final position.  These will not
  // be moved.
  bool Found[4];
  Found[0] = Arch != UnknownArch;
  Found[1] = Vendor != UnknownVendor;
  Found[2] = OS != UnknownOS;
  Found[3] = Environment != UnknownEnvironment;

  // If they are not there already, permute the components into their canonical
  // positions by seeing if they parse as a valid architecture, and if so moving
  // the component to the architecture position etc.
  for (unsigned Pos = 0; Pos != std::size(Found); ++Pos) {
    if (Found[Pos])
      continue; // Already in the canonical position.

    for (unsigned Idx = 0; Idx != Components.size(); ++Idx) {
      // Do not reparse any components that already matched.
      if (Idx < std::size(Found) && Found[Idx])
        continue;

      // Does this component parse as valid for the target position?
      bool Valid = false;
      StringRef Comp = Components[Idx];
      switch (Pos) {
      default:
        llvm_unreachable("unexpected component type!");
      case 0:
        Arch = parseArch(Comp);
        Valid = Arch != UnknownArch;
        break;
      case 1:
        Vendor = parseVendor(Comp);
        Valid = Vendor != UnknownVendor;
        break;
      case 2:
        OS = parseOS(Comp);
        IsCygwin = Comp.starts_with("cygwin") || Comp.starts_with("msys");
        IsMinGW32 = Comp.starts_with("mingw");
        Valid = OS != UnknownOS || IsCygwin || IsMinGW32;
        break;
      case 3:
        Environment = parseEnvironment(Comp);
        Valid = Environment != UnknownEnvironment;
        if (!Valid) {
          ObjectFormat = parseFormat(Comp);
          Valid = ObjectFormat != UnknownObjectFormat;
        }
        break;
      }
      if (!Valid)
        continue; // Nope, try the next component.

      // Move the component to the target position, pushing any non-fixed
      // components that are in the way to the right.  This tends to give
      // good results in the common cases of a forgotten vendor component
      // or a wrongly positioned environment.
      if (Pos < Idx) {
        // Insert left, pushing the existing components to the right.  For
        // example, a-b-i386 -> i386-a-b when moving i386 to the front.
        StringRef CurrentComponent(""); // The empty component.
        // Replace the component we are moving with an empty component.
        std::swap(CurrentComponent, Components[Idx]);
        // Insert the component being moved at Pos, displacing any existing
        // components to the right.
        for (unsigned i = Pos; !CurrentComponent.empty(); ++i) {
          // Skip over any fixed components.
          while (i < std::size(Found) && Found[i])
            ++i;
          // Place the component at the new position, getting the component
          // that was at this position - it will be moved right.
          std::swap(CurrentComponent, Components[i]);
        }
      } else if (Pos > Idx) {
        // Push right by inserting empty components until the component at Idx
        // reaches the target position Pos.  For example, pc-a -> -pc-a when
        // moving pc to the second position.
        do {
          // Insert one empty component at Idx.
          StringRef CurrentComponent(""); // The empty component.
          for (unsigned i = Idx; i < Components.size();) {
            // Place the component at the new position, getting the component
            // that was at this position - it will be moved right.
            std::swap(CurrentComponent, Components[i]);
            // If it was placed on top of an empty component then we are done.
            if (CurrentComponent.empty())
              break;
            // Advance to the next component, skipping any fixed components.
            while (++i < std::size(Found) && Found[i])
              ;
          }
          // The last component was pushed off the end - append it.
          if (!CurrentComponent.empty())
            Components.push_back(CurrentComponent);

          // Advance Idx to the component's new position.
          while (++Idx < std::size(Found) && Found[Idx])
            ;
        } while (Idx < Pos); // Add more until the final position is reached.
      }
      assert(Pos < Components.size() && Components[Pos] == Comp &&
             "Component moved wrong!");
      Found[Pos] = true;
      break;
    }
  }

  // If "none" is in the middle component in a three-component triple, treat it
  // as the OS (Components[2]) instead of the vendor (Components[1]).
  if (Found[0] && !Found[1] && !Found[2] && Found[3] &&
      Components[1] == "none" && Components[2].empty())
    std::swap(Components[1], Components[2]);

  // Replace empty components with "unknown" value.
  for (StringRef &C : Components)
    if (C.empty())
      C = "unknown";

  // Special case logic goes here.  At this point Arch, Vendor and OS have the
  // correct values for the computed components.
  std::string NormalizedEnvironment;
  if (Environment == Triple::Android &&
      Components[3].starts_with("androideabi")) {
    StringRef AndroidVersion = Components[3].drop_front(strlen("androideabi"));
    if (AndroidVersion.empty()) {
      Components[3] = "android";
    } else {
      NormalizedEnvironment = Twine("android", AndroidVersion).str();
      Components[3] = NormalizedEnvironment;
    }
  }

  // SUSE uses "gnueabi" to mean "gnueabihf"
  if (Vendor == Triple::SUSE && Environment == llvm::Triple::GNUEABI)
    Components[3] = "gnueabihf";

  if (OS == Triple::Win32) {
    Components.resize(4);
    Components[2] = "windows";
    if (Environment == UnknownEnvironment) {
      if (ObjectFormat == UnknownObjectFormat || ObjectFormat == Triple::COFF)
        Components[3] = "msvc";
      else
        Components[3] = getObjectFormatTypeName(ObjectFormat);
    }
  } else if (IsMinGW32) {
    Components.resize(4);
    Components[2] = "windows";
    Components[3] = "gnu";
  } else if (IsCygwin) {
    Components.resize(4);
    Components[2] = "windows";
    Components[3] = "cygnus";
  }
  if (IsMinGW32 || IsCygwin ||
      (OS == Triple::Win32 && Environment != UnknownEnvironment)) {
    if (ObjectFormat != UnknownObjectFormat && ObjectFormat != Triple::COFF) {
      Components.resize(5);
      Components[4] = getObjectFormatTypeName(ObjectFormat);
    }
  }

  // Normalize DXIL triple if it does not include DXIL version number.
  // Determine DXIL version number using the minor version number of Shader
  // Model version specified in target triple, if any. Prior to decoupling DXIL
  // version numbering from that of Shader Model DXIL version 1.Y corresponds to
  // SM 6.Y. E.g., dxilv1.Y-unknown-shadermodelX.Y-hull
  if (Components[0] == "dxil") {
    if (Components.size() > 4) {
      Components.resize(4);
    }
    // Add DXIL version only if shadermodel is specified in the triple
    if (OS == Triple::ShaderModel) {
      Components[0] = getDXILArchNameFromShaderModel(Components[2]);
    }
  }

  // Currently the firmware OS is an Apple specific concept.
  if ((Components.size() > 2) && (Components[2] == "firmware") &&
      (Components[1] != "apple"))
    llvm::reportFatalUsageError(
        "the firmware target os is only supported for the apple vendor");

  // Canonicalize the components if necessary.
  switch (Form) {
  case CanonicalForm::ANY:
    break;
  case CanonicalForm::THREE_IDENT:
  case CanonicalForm::FOUR_IDENT:
  case CanonicalForm::FIVE_IDENT: {
    Components.resize(static_cast<unsigned>(Form), "unknown");
    break;
  }
  }

  // Stick the corrected components back together to form the normalized string.
  return join(Components, "-");
}

StringRef Triple::getArchName() const {
  return StringRef(Data).split('-').first; // Isolate first component
}

StringRef Triple::getVendorName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  return Tmp.split('-').first;                       // Isolate second component
}

StringRef Triple::getOSName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  Tmp = Tmp.split('-').second;                       // Strip second component
  return Tmp.split('-').first;                       // Isolate third component
}

StringRef Triple::getEnvironmentName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  Tmp = Tmp.split('-').second;                       // Strip second component
  return Tmp.split('-').second;                      // Strip third component
}

StringRef Triple::getOSAndEnvironmentName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  return Tmp.split('-').second;                      // Strip second component
}

static VersionTuple parseVersionFromName(StringRef Name) {
  VersionTuple Version;
  Version.tryParse(Name);
  return Version.withoutBuild();
}

VersionTuple Triple::getEnvironmentVersion() const {
  return parseVersionFromName(getEnvironmentVersionString());
}

StringRef Triple::getEnvironmentVersionString() const {
  StringRef EnvironmentName = getEnvironmentName();

  // none is a valid environment type - it basically amounts to a freestanding
  // environment.
  if (EnvironmentName == "none")
    return "";

  StringRef EnvironmentTypeName = getEnvironmentTypeName(getEnvironment());
  EnvironmentName.consume_front(EnvironmentTypeName);

  if (EnvironmentName.contains("-")) {
    // -obj is the suffix
    if (getObjectFormat() != Triple::UnknownObjectFormat) {
      StringRef ObjectFormatTypeName =
          getObjectFormatTypeName(getObjectFormat());
      const std::string tmp = (Twine("-") + ObjectFormatTypeName).str();
      EnvironmentName.consume_back(tmp);
    }
  }
  return EnvironmentName;
}

VersionTuple Triple::getOSVersion() const {
  StringRef OSName = getOSName();
  // Assume that the OS portion of the triple starts with the canonical name.
  StringRef OSTypeName = getOSTypeName(getOS());
  if (OSName.starts_with(OSTypeName))
    OSName = OSName.substr(OSTypeName.size());
  else if (getOS() == MacOSX)
    OSName.consume_front("macos");
  else if (OSName.starts_with("visionos"))
    OSName.consume_front("visionos");

  return parseVersionFromName(OSName);
}

bool Triple::getMacOSXVersion(VersionTuple &Version) const {
  Version = getOSVersion();

  switch (getOS()) {
  default:
    llvm_unreachable("unexpected OS for Darwin triple");
  case Darwin:
    // Default to darwin8, i.e., MacOSX 10.4.
    if (Version.getMajor() == 0)
      Version = VersionTuple(8);
    // Darwin version numbers are skewed from OS X versions.
    if (Version.getMajor() < 4) {
      return false;
    }
    if (Version.getMajor() <= 19) {
      Version = VersionTuple(10, Version.getMajor() - 4);
    } else if (Version.getMajor() < 25) {
      // darwin20-24 corresponds to macOS 11-15.
      Version = VersionTuple(11 + Version.getMajor() - 20);
    } else if ((Version.getMajor() == 25) || (Version.getMajor() == 26)) {
      // darwin25-26 corresponds to macOS 26-27.
      Version = VersionTuple(Version.getMajor() + 1);
    } else {
      // Starting with darwin27, it naturally corresponds to the same macOS
      // version.
    }
    break;
  case MacOSX:
    // Default to 10.4.
    if (Version.getMajor() == 0) {
      Version = VersionTuple(10, 4);
    } else if (Version.getMajor() < 10) {
      return false;
    }
    break;
  case IOS:
  case TvOS:
  case WatchOS:
    // Ignore the version from the triple.  This is only handled because the
    // the clang driver combines OS X and IOS support into a common Darwin
    // toolchain that wants to know the OS X version number even when targeting
    // IOS.
    Version = VersionTuple(10, 4);
    break;
  case XROS:
    llvm_unreachable("OSX version isn't relevant for xrOS");
  case DriverKit:
    llvm_unreachable("OSX version isn't relevant for DriverKit");
  case Firmware:
    llvm_unreachable("OSX version isn't relevant for Firmware");
  }
  return true;
}

VersionTuple Triple::getiOSVersion() const {
  switch (getOS()) {
  default:
    llvm_unreachable("unexpected OS for Darwin triple");
  case Darwin:
  case MacOSX:
    // Ignore the version from the triple.  This is only handled because the
    // the clang driver combines OS X and IOS support into a common Darwin
    // toolchain that wants to know the iOS version number even when targeting
    // OS X.
    return VersionTuple(5);
  case IOS:
  case TvOS: {
    VersionTuple Version = getOSVersion();
    // Default to 5.0 (or 7.0 for arm64).
    if (Version.getMajor() == 0)
      return (getArch() == aarch64) ? VersionTuple(7) : VersionTuple(5);
    if (Version.getMajor() == 19)
      // tvOS 19 corresponds to ios26.
      return VersionTuple(26);
    return getCanonicalVersionForOS(OSType::IOS, Version,
                                    isValidVersionForOS(OSType::IOS, Version));
  }
  case XROS: {
    VersionTuple Version = getOSVersion();
    // xrOS 1 is aligned with iOS 17.
    if (Version.getMajor() < 3)
      return Version.withMajorReplaced(Version.getMajor() + 16);
    // visionOS 3 corresponds to ios 26+.
    if (Version.getMajor() == 3)
      return VersionTuple(26);
    return getCanonicalVersionForOS(OSType::XROS, Version,
                                    isValidVersionForOS(OSType::XROS, Version));
  }
  case WatchOS: {
    VersionTuple Version = getOSVersion();
    // watchOS 12 corresponds to ios 26.
    if (Version.getMajor() == 12)
      return VersionTuple(26);
    return getCanonicalVersionForOS(
        OSType::WatchOS, Version,
        isValidVersionForOS(OSType::WatchOS, Version));
  }
  case BridgeOS:
    llvm_unreachable("conflicting triple info");
  case DriverKit:
    llvm_unreachable("DriverKit doesn't have an iOS version");
  case Firmware:
    llvm_unreachable("iOS version isn't relevant for Firmware");
  }
}

VersionTuple Triple::getWatchOSVersion() const {
  switch (getOS()) {
  default:
    llvm_unreachable("unexpected OS for Darwin triple");
  case Darwin:
  case MacOSX:
    // Ignore the version from the triple.  This is only handled because the
    // the clang driver combines OS X and IOS support into a common Darwin
    // toolchain that wants to know the iOS version number even when targeting
    // OS X.
    return VersionTuple(2);
  case WatchOS: {
    VersionTuple Version = getOSVersion();
    if (Version.getMajor() == 0)
      return VersionTuple(2);
    return Version;
  }
  case IOS:
    llvm_unreachable("conflicting triple info");
  case XROS:
    llvm_unreachable("watchOS version isn't relevant for xrOS");
  case DriverKit:
    llvm_unreachable("DriverKit doesn't have a WatchOS version");
  case Firmware:
    llvm_unreachable("watchOS version isn't relevant for Firmware");
  }
}

VersionTuple Triple::getDriverKitVersion() const {
  switch (getOS()) {
  default:
    llvm_unreachable("unexpected OS for Darwin triple");
  case DriverKit:
    VersionTuple Version = getOSVersion();
    if (Version.getMajor() == 0)
      return Version.withMajorReplaced(19);
    return Version;
  }
}

VersionTuple Triple::getVulkanVersion() const {
  if (getArch() != spirv || getOS() != Vulkan)
    llvm_unreachable("invalid Vulkan SPIR-V triple");

  VersionTuple VulkanVersion = getOSVersion();
  SubArchType SpirvVersion = getSubArch();

  llvm::DenseMap<VersionTuple, SubArchType> ValidVersionMap = {
      // Vulkan 1.2 -> SPIR-V 1.5.
      {VersionTuple(1, 2), SPIRVSubArch_v15},
      // Vulkan 1.3 -> SPIR-V 1.6.
      {VersionTuple(1, 3), SPIRVSubArch_v16}};

  // If Vulkan version is unset, default to 1.2.
  if (VulkanVersion == VersionTuple(0))
    VulkanVersion = VersionTuple(1, 2);

  if (ValidVersionMap.contains(VulkanVersion) &&
      (ValidVersionMap.lookup(VulkanVersion) == SpirvVersion ||
       SpirvVersion == NoSubArch))
    return VulkanVersion;

  return VersionTuple(0);
}

VersionTuple Triple::getDXILVersion() const {
  if (getArch() != dxil || getOS() != ShaderModel)
    llvm_unreachable("invalid DXIL triple");
  StringRef Arch = getArchName();
  if (getSubArch() == NoSubArch)
    Arch = getDXILArchNameFromShaderModel(getOSName());
  Arch.consume_front("dxilv");
  VersionTuple DXILVersion = parseVersionFromName(Arch);
  // FIXME: validate DXIL version against Shader Model version.
  // Tracked by https://github.com/llvm/llvm-project/issues/91388
  return DXILVersion;
}

void Triple::setTriple(const Twine &Str) { *this = Triple(Str); }

void Triple::setArch(ArchType Kind, SubArchType SubArch) {
  setArchName(getArchName(Kind, SubArch));
}

void Triple::setVendor(VendorType Kind) {
  setVendorName(getVendorTypeName(Kind));
}

void Triple::setOS(OSType Kind) { setOSName(getOSTypeName(Kind)); }

void Triple::setEnvironment(EnvironmentType Kind) {
  if (ObjectFormat == getDefaultFormat(*this))
    return setEnvironmentName(getEnvironmentTypeName(Kind));

  setEnvironmentName((getEnvironmentTypeName(Kind) + Twine("-") +
                      getObjectFormatTypeName(ObjectFormat))
                         .str());
}

void Triple::setObjectFormat(ObjectFormatType Kind) {
  if (Environment == UnknownEnvironment)
    return setEnvironmentName(getObjectFormatTypeName(Kind));

  setEnvironmentName((getEnvironmentTypeName(Environment) + Twine("-") +
                      getObjectFormatTypeName(Kind))
                         .str());
}

void Triple::setArchName(StringRef Str) {
  setTriple(Str + "-" + getVendorName() + "-" + getOSAndEnvironmentName());
}

void Triple::setVendorName(StringRef Str) {
  setTriple(getArchName() + "-" + Str + "-" + getOSAndEnvironmentName());
}

void Triple::setOSName(StringRef Str) {
  if (hasEnvironment())
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str + "-" +
              getEnvironmentName());
  else
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

void Triple::setEnvironmentName(StringRef Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + getOSName() + "-" +
            Str);
}

void Triple::setOSAndEnvironmentName(StringRef Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

unsigned Triple::getArchPointerBitWidth(llvm::Triple::ArchType Arch) {
  switch (Arch) {
  case llvm::Triple::UnknownArch:
    return 0;

  case llvm::Triple::avr:
  case llvm::Triple::msp430:
    return 16;

  case llvm::Triple::aarch64_32:
  case llvm::Triple::amdil:
  case llvm::Triple::arc:
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::csky:
  case llvm::Triple::dxil:
  case llvm::Triple::hexagon:
  case llvm::Triple::hsail:
  case llvm::Triple::kalimba:
  case llvm::Triple::lanai:
  case llvm::Triple::loongarch32:
  case llvm::Triple::m68k:
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::nvptx:
  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::r600:
  case llvm::Triple::renderscript32:
  case llvm::Triple::riscv32:
  case llvm::Triple::riscv32be:
  case llvm::Triple::shave:
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
  case llvm::Triple::spir:
  case llvm::Triple::spirv32:
  case llvm::Triple::tce:
  case llvm::Triple::tcele:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
  case llvm::Triple::wasm32:
  case llvm::Triple::x86:
  case llvm::Triple::xcore:
  case llvm::Triple::xtensa:
    return 32;

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::amdgpu:
  case llvm::Triple::amdil64:
  case llvm::Triple::bpfeb:
  case llvm::Triple::bpfel:
  case llvm::Triple::hsail64:
  case llvm::Triple::loongarch64:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
  case llvm::Triple::nvptx64:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
  case llvm::Triple::renderscript64:
  case llvm::Triple::riscv64:
  case llvm::Triple::riscv64be:
  case llvm::Triple::sparcv9:
  case llvm::Triple::spirv:
  case llvm::Triple::spir64:
  case llvm::Triple::spirv64:
  case llvm::Triple::tcele64:
  case llvm::Triple::systemz:
  case llvm::Triple::ve:
  case llvm::Triple::wasm64:
  case llvm::Triple::x86_64:
    return 64;
  }
  llvm_unreachable("Invalid architecture value");
}

unsigned Triple::getTrampolineSize() const {
  switch (getArch()) {
  default:
    break;
  case Triple::ppc:
  case Triple::ppcle:
    if (isOSLinux())
      return 40;
    break;
  case Triple::ppc64:
  case Triple::ppc64le:
    if (isOSLinux())
      return 48;
    break;
  }
  return 32;
}

bool Triple::isArch64Bit() const {
  return getArchPointerBitWidth(getArch()) == 64;
}

bool Triple::isArch32Bit() const {
  return getArchPointerBitWidth(getArch()) == 32;
}

bool Triple::isArch16Bit() const {
  return getArchPointerBitWidth(getArch()) == 16;
}

Triple Triple::get32BitArchVariant() const {
  Triple T(*this);
  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::amdgpu:
  case Triple::avr:
  case Triple::bpfeb:
  case Triple::bpfel:
  case Triple::msp430:
  case Triple::systemz:
  case Triple::ve:
    T.setArch(UnknownArch);
    break;

  case Triple::aarch64_32:
  case Triple::amdil:
  case Triple::arc:
  case Triple::arm:
  case Triple::armeb:
  case Triple::csky:
  case Triple::dxil:
  case Triple::hexagon:
  case Triple::hsail:
  case Triple::kalimba:
  case Triple::lanai:
  case Triple::loongarch32:
  case Triple::m68k:
  case Triple::mips:
  case Triple::mipsel:
  case Triple::nvptx:
  case Triple::ppc:
  case Triple::ppcle:
  case Triple::r600:
  case Triple::renderscript32:
  case Triple::riscv32:
  case Triple::riscv32be:
  case Triple::shave:
  case Triple::sparc:
  case Triple::sparcel:
  case Triple::spir:
  case Triple::spirv32:
  case Triple::tce:
  case Triple::tcele:
  case Triple::thumb:
  case Triple::thumbeb:
  case Triple::wasm32:
  case Triple::x86:
  case Triple::xcore:
  case Triple::xtensa:
    // Already 32-bit.
    break;

  case Triple::aarch64:
    T.setArch(Triple::arm);
    break;
  case Triple::aarch64_be:
    T.setArch(Triple::armeb);
    break;
  case Triple::amdil64:
    T.setArch(Triple::amdil);
    break;
  case Triple::hsail64:
    T.setArch(Triple::hsail);
    break;
  case Triple::loongarch64:
    T.setArch(Triple::loongarch32);
    break;
  case Triple::mips64:
    T.setArch(Triple::mips, getSubArch());
    break;
  case Triple::mips64el:
    T.setArch(Triple::mipsel, getSubArch());
    break;
  case Triple::nvptx64:
    T.setArch(Triple::nvptx);
    break;
  case Triple::ppc64:
    T.setArch(Triple::ppc);
    break;
  case Triple::ppc64le:
    T.setArch(Triple::ppcle);
    break;
  case Triple::renderscript64:
    T.setArch(Triple::renderscript32);
    break;
  case Triple::riscv64:
    T.setArch(Triple::riscv32);
    break;
  case Triple::riscv64be:
    T.setArch(Triple::riscv32be);
    break;
  case Triple::sparcv9:
    T.setArch(Triple::sparc);
    break;
  case Triple::spir64:
    T.setArch(Triple::spir);
    break;
  case Triple::spirv:
  case Triple::spirv64:
    T.setArch(Triple::spirv32, getSubArch());
    break;
  case Triple::tcele64:
    T.setArch(Triple::tcele);
    break;
  case Triple::wasm64:
    T.setArch(Triple::wasm32);
    break;
  case Triple::x86_64:
    T.setArch(Triple::x86);
    break;
  }
  return T;
}

Triple Triple::get64BitArchVariant() const {
  Triple T(*this);
  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::arc:
  case Triple::avr:
  case Triple::csky:
  case Triple::dxil:
  case Triple::hexagon:
  case Triple::kalimba:
  case Triple::lanai:
  case Triple::m68k:
  case Triple::msp430:
  case Triple::r600:
  case Triple::shave:
  case Triple::sparcel:
  case Triple::tce:
  case Triple::xcore:
  case Triple::xtensa:
    T.setArch(UnknownArch);
    break;

  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::amdgpu:
  case Triple::amdil64:
  case Triple::bpfeb:
  case Triple::bpfel:
  case Triple::hsail64:
  case Triple::loongarch64:
  case Triple::mips64:
  case Triple::mips64el:
  case Triple::nvptx64:
  case Triple::ppc64:
  case Triple::ppc64le:
  case Triple::renderscript64:
  case Triple::riscv64:
  case Triple::riscv64be:
  case Triple::sparcv9:
  case Triple::spir64:
  case Triple::spirv64:
  case Triple::systemz:
  case Triple::tcele64:
  case Triple::ve:
  case Triple::wasm64:
  case Triple::x86_64:
    // Already 64-bit.
    break;

  case Triple::aarch64_32:
    T.setArch(Triple::aarch64);
    break;
  case Triple::amdil:
    T.setArch(Triple::amdil64);
    break;
  case Triple::arm:
    T.setArch(Triple::aarch64);
    break;
  case Triple::armeb:
    T.setArch(Triple::aarch64_be);
    break;
  case Triple::hsail:
    T.setArch(Triple::hsail64);
    break;
  case Triple::loongarch32:
    T.setArch(Triple::loongarch64);
    break;
  case Triple::mips:
    T.setArch(Triple::mips64, getSubArch());
    break;
  case Triple::mipsel:
    T.setArch(Triple::mips64el, getSubArch());
    break;
  case Triple::nvptx:
    T.setArch(Triple::nvptx64);
    break;
  case Triple::ppc:
    T.setArch(Triple::ppc64);
    break;
  case Triple::ppcle:
    T.setArch(Triple::ppc64le);
    break;
  case Triple::renderscript32:
    T.setArch(Triple::renderscript64);
    break;
  case Triple::riscv32:
    T.setArch(Triple::riscv64);
    break;
  case Triple::riscv32be:
    T.setArch(Triple::riscv64be);
    break;
  case Triple::sparc:
    T.setArch(Triple::sparcv9);
    break;
  case Triple::spir:
    T.setArch(Triple::spir64);
    break;
  case Triple::spirv:
  case Triple::spirv32:
    T.setArch(Triple::spirv64, getSubArch());
    break;
  case Triple::tcele:
    T.setArch(Triple::tcele64);
    break;
  case Triple::thumb:
    T.setArch(Triple::aarch64);
    break;
  case Triple::thumbeb:
    T.setArch(Triple::aarch64_be);
    break;
  case Triple::wasm32:
    T.setArch(Triple::wasm64);
    break;
  case Triple::x86:
    T.setArch(Triple::x86_64);
    break;
  }
  return T;
}

Triple Triple::getBigEndianArchVariant() const {
  Triple T(*this);
  // Already big endian.
  if (!isLittleEndian())
    return T;
  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::amdgpu:
  case Triple::amdil64:
  case Triple::amdil:
  case Triple::avr:
  case Triple::dxil:
  case Triple::hexagon:
  case Triple::hsail64:
  case Triple::hsail:
  case Triple::kalimba:
  case Triple::loongarch32:
  case Triple::loongarch64:
  case Triple::msp430:
  case Triple::nvptx64:
  case Triple::nvptx:
  case Triple::r600:
  case Triple::renderscript32:
  case Triple::renderscript64:
  case Triple::shave:
  case Triple::spir64:
  case Triple::spir:
  case Triple::spirv:
  case Triple::spirv32:
  case Triple::spirv64:
  case Triple::tcele64:
  case Triple::wasm32:
  case Triple::wasm64:
  case Triple::x86:
  case Triple::x86_64:
  case Triple::xcore:
  case Triple::ve:
  case Triple::csky:
  case Triple::xtensa:

  // ARM is intentionally unsupported here, changing the architecture would
  // drop any arch suffixes.
  case Triple::arm:
  case Triple::thumb:
    T.setArch(UnknownArch);
    break;

  case Triple::aarch64:
    T.setArch(Triple::aarch64_be);
    break;
  case Triple::bpfel:
    T.setArch(Triple::bpfeb);
    break;
  case Triple::mips64el:
    T.setArch(Triple::mips64, getSubArch());
    break;
  case Triple::mipsel:
    T.setArch(Triple::mips, getSubArch());
    break;
  case Triple::ppcle:
    T.setArch(Triple::ppc);
    break;
  case Triple::ppc64le:
    T.setArch(Triple::ppc64);
    break;
  case Triple::riscv32:
    T.setArch(Triple::riscv32be);
    break;
  case Triple::riscv64:
    T.setArch(Triple::riscv64be);
    break;
  case Triple::sparcel:
    T.setArch(Triple::sparc);
    break;
  case Triple::tcele:
    T.setArch(Triple::tce);
    break;
  default:
    llvm_unreachable("getBigEndianArchVariant: unknown triple.");
  }
  return T;
}

Triple Triple::getLittleEndianArchVariant() const {
  Triple T(*this);
  if (isLittleEndian())
    return T;

  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::lanai:
  case Triple::sparcv9:
  case Triple::systemz:
  case Triple::m68k:

  // ARM is intentionally unsupported here, changing the architecture would
  // drop any arch suffixes.
  case Triple::armeb:
  case Triple::thumbeb:
    T.setArch(UnknownArch);
    break;

  case Triple::aarch64_be:
    T.setArch(Triple::aarch64);
    break;
  case Triple::bpfeb:
    T.setArch(Triple::bpfel);
    break;
  case Triple::mips64:
    T.setArch(Triple::mips64el, getSubArch());
    break;
  case Triple::mips:
    T.setArch(Triple::mipsel, getSubArch());
    break;
  case Triple::ppc:
    T.setArch(Triple::ppcle);
    break;
  case Triple::ppc64:
    T.setArch(Triple::ppc64le);
    break;
  case Triple::riscv32be:
    T.setArch(Triple::riscv32);
    break;
  case Triple::riscv64be:
    T.setArch(Triple::riscv64);
    break;
  case Triple::sparc:
    T.setArch(Triple::sparcel);
    break;
  case Triple::tce:
    T.setArch(Triple::tcele);
    break;
  default:
    llvm_unreachable("getLittleEndianArchVariant: unknown triple.");
  }
  return T;
}

bool Triple::isLittleEndian() const {
  switch (getArch()) {
  case Triple::aarch64:
  case Triple::aarch64_32:
  case Triple::amdgpu:
  case Triple::amdil64:
  case Triple::amdil:
  case Triple::arm:
  case Triple::avr:
  case Triple::bpfel:
  case Triple::csky:
  case Triple::dxil:
  case Triple::hexagon:
  case Triple::hsail64:
  case Triple::hsail:
  case Triple::kalimba:
  case Triple::loongarch32:
  case Triple::loongarch64:
  case Triple::mips64el:
  case Triple::mipsel:
  case Triple::msp430:
  case Triple::nvptx64:
  case Triple::nvptx:
  case Triple::ppcle:
  case Triple::ppc64le:
  case Triple::r600:
  case Triple::renderscript32:
  case Triple::renderscript64:
  case Triple::riscv32:
  case Triple::riscv64:
  case Triple::shave:
  case Triple::sparcel:
  case Triple::spir64:
  case Triple::spir:
  case Triple::spirv:
  case Triple::spirv32:
  case Triple::spirv64:
  case Triple::tcele:
  case Triple::tcele64:
  case Triple::thumb:
  case Triple::ve:
  case Triple::wasm32:
  case Triple::wasm64:
  case Triple::x86:
  case Triple::x86_64:
  case Triple::xcore:
  case Triple::xtensa:
    return true;
  default:
    return false;
  }
}

unsigned Triple::getDefaultWCharSize() const {
  if (getArch() == Triple::xcore)
    return 1;
  if (isOSWindowsOrUEFI() || isPS())
    return 2;
  if (isOSAIX() && isArch32Bit())
    return 2;
  return 4;
}

bool Triple::isCompatibleWith(const Triple &Other) const {
  // On MinGW, C code is usually built with a "w64" vendor, while Rust
  // often uses a "pc" vendor.
  bool IgnoreVendor = isWindowsGNUEnvironment();

  // ARM and Thumb triples are compatible, if subarch, vendor and OS match.
  if ((getArch() == Triple::thumb && Other.getArch() == Triple::arm) ||
      (getArch() == Triple::arm && Other.getArch() == Triple::thumb) ||
      (getArch() == Triple::thumbeb && Other.getArch() == Triple::armeb) ||
      (getArch() == Triple::armeb && Other.getArch() == Triple::thumbeb)) {
    if (getVendor() == Triple::Apple)
      return getSubArch() == Other.getSubArch() &&
             getVendor() == Other.getVendor() && getOS() == Other.getOS();
    else
      return getSubArch() == Other.getSubArch() &&
             (getVendor() == Other.getVendor() || IgnoreVendor) &&
             getOS() == Other.getOS() &&
             getEnvironment() == Other.getEnvironment() &&
             getObjectFormat() == Other.getObjectFormat();
  }

  if (getArch() == Triple::amdgpu && Other.getArch() == Triple::amdgpu) {
    if (getOS() != Other.getOS() || getVendor() != Other.getVendor() ||
        getEnvironment() != Other.getEnvironment() ||
        getObjectFormat() != Other.getObjectFormat())
      return false;
    return AMDGPU::isSubArchCompatible(*this, Other);
  }

  // If vendor is apple, ignore the version number (the environment field)
  // and the object format.
  if (getVendor() == Triple::Apple)
    return getArch() == Other.getArch() && getSubArch() == Other.getSubArch() &&
           (getVendor() == Other.getVendor() || IgnoreVendor) &&
           getOS() == Other.getOS();

  return getArch() == Other.getArch() && getSubArch() == Other.getSubArch() &&
         (getVendor() == Other.getVendor() || IgnoreVendor) &&
         getOS() == Other.getOS() &&
         getEnvironment() == Other.getEnvironment() &&
         getObjectFormat() == Other.getObjectFormat();
}

std::string Triple::merge(const Triple &Other) const {
  // If vendor is apple, pick the triple with the larger version number.
  if (getVendor() == Triple::Apple && Other.isOSVersionLT(*this))
    return str();

  if (isAMDGCN() && Other.isAMDGCN() && getOS() == Other.getOS() &&
      getVendor() == Other.getVendor())
    return AMDGPU::mergeSubArch(*this, Other);

  return Other.str();
}

bool Triple::isMacOSXVersionLT(unsigned Major, unsigned Minor,
                               unsigned Micro) const {
  assert(isMacOSX() && "Not an OS X triple!");

  // If this is OS X, expect a sane version number.
  if (getOS() == Triple::MacOSX)
    return isOSVersionLT(Major, Minor, Micro);

  // Otherwise, compare to the "Darwin" number.
  if (Major == 10)
    return isOSVersionLT(Minor + 4, Micro, 0);
  assert(Major >= 11 && "Unexpected major version");
  if (Major < 25)
    return isOSVersionLT(Major - 11 + 20, Minor, Micro);
  return isOSVersionLT(Major + 1, Minor, Micro);
}

VersionTuple Triple::getMinimumSupportedOSVersion() const {
  if (getVendor() != Triple::Apple || getArch() != Triple::aarch64)
    return VersionTuple();
  switch (getOS()) {
  case Triple::MacOSX:
    // ARM64 slice is supported starting from macOS 11.0+.
    return VersionTuple(11, 0, 0);
  case Triple::IOS:
    // ARM64 slice is supported starting from Mac Catalyst 14 (macOS 11).
    // ARM64 simulators are supported for iOS 14+.
    if (isMacCatalystEnvironment() || isSimulatorEnvironment())
      return VersionTuple(14, 0, 0);
    // ARM64e slice is supported starting from iOS 14.
    if (isArm64e())
      return VersionTuple(14, 0, 0);
    break;
  case Triple::TvOS:
    // ARM64 simulators are supported for tvOS 14+.
    if (isSimulatorEnvironment())
      return VersionTuple(14, 0, 0);
    break;
  case Triple::WatchOS:
    // ARM64 simulators are supported for watchOS 7+.
    if (isSimulatorEnvironment())
      return VersionTuple(7, 0, 0);
    // ARM64/ARM64e slices are supported starting from watchOS 26.
    // ARM64_32 is older though.
    assert(getArch() != Triple::aarch64_32);
    return VersionTuple(26, 0, 0);
  case Triple::DriverKit:
    return VersionTuple(20, 0, 0);
  default:
    break;
  }
  return VersionTuple();
}

VersionTuple Triple::getCanonicalVersionForOS(OSType OSKind,
                                              const VersionTuple &Version,
                                              bool IsInValidRange) {
  const unsigned MacOSRangeBump = 10;
  const unsigned IOSRangeBump = 7;
  const unsigned XROSRangeBump = 23;
  const unsigned WatchOSRangeBump = 14;
  switch (OSKind) {
  case MacOSX: {
    // macOS 10.16 is canonicalized to macOS 11.
    if (Version == VersionTuple(10, 16))
      return VersionTuple(11, 0);
    // macOS 16 is canonicalized to macOS 26.
    if (Version == VersionTuple(16, 0))
      return VersionTuple(26, 0);
    if (!IsInValidRange)
      return Version.withMajorReplaced(Version.getMajor() + MacOSRangeBump);
    break;
  }
  case IOS:
  case TvOS: {
    // Both iOS & tvOS 19.0 canonicalize to 26.
    if (Version == VersionTuple(19, 0))
      return VersionTuple(26, 0);
    if (!IsInValidRange)
      return Version.withMajorReplaced(Version.getMajor() + IOSRangeBump);
    break;
  }
  case XROS: {
    // visionOS3 is canonicalized to 26.
    if (Version == VersionTuple(3, 0))
      return VersionTuple(26, 0);
    if (!IsInValidRange)
      return Version.withMajorReplaced(Version.getMajor() + XROSRangeBump);
    break;
  }
  case WatchOS: {
    // watchOS 12 is canonicalized to 26.
    if (Version == VersionTuple(12, 0))
      return VersionTuple(26, 0);
    if (!IsInValidRange)
      return Version.withMajorReplaced(Version.getMajor() + WatchOSRangeBump);
    break;
  }
  case DriverKit: {
    // DriverKit26 is canonicalized to 27.
    if (Version.getMajor() == 26U)
      return Version.withMajorReplaced(27);
    break;
  }
  default:
    return Version;
  }

  return Version;
}

bool Triple::isValidVersionForOS(OSType OSKind, const VersionTuple &Version) {
  /// This constant is used to capture gaps in versioning.
  const VersionTuple CommonVersion(26);
  auto IsValid = [&](const VersionTuple &StartingVersion) {
    return !((Version > StartingVersion) && (Version < CommonVersion));
  };
  switch (OSKind) {
  case WatchOS: {
    const VersionTuple StartingWatchOS(12);
    return IsValid(StartingWatchOS);
  }
  case IOS:
  case TvOS: {
    const VersionTuple StartingIOS(19);
    return IsValid(StartingIOS);
  }
  case MacOSX: {
    const VersionTuple StartingMacOS(16);
    return IsValid(StartingMacOS);
  }
  case XROS: {
    const VersionTuple StartingXROS(3);
    return IsValid(StartingXROS);
  }
  default:
    return true;
  }

  llvm_unreachable("unexpected or invalid os version");
}

ExceptionHandling Triple::getDefaultExceptionHandling() const {
  if (isOSBinFormatCOFF()) {
    if (getArch() == Triple::x86 &&
        (isOSCygMing() || isWindowsItaniumEnvironment()))
      return ExceptionHandling::DwarfCFI;
    return ExceptionHandling::WinEH;
  }

  if (isOSBinFormatXCOFF())
    return ExceptionHandling::AIX;
  if (isOSBinFormatGOFF())
    return ExceptionHandling::ZOS;

  if (isARM() || isThumb()) {
    if (isOSBinFormatELF()) {
      return getOS() == Triple::NetBSD ? ExceptionHandling::DwarfCFI
                                       : ExceptionHandling::ARM;
    }

    return isOSDarwin() && !isWatchABI() ? ExceptionHandling::SjLj
                                         : ExceptionHandling::DwarfCFI;
  }

  if (isAArch64() || isX86() || isPPC() || isMIPS() || isSPARC() || isBPF() ||
      isRISCV() || isLoongArch())
    return ExceptionHandling::DwarfCFI;

  switch (getArch()) {
  case Triple::arc:
  case Triple::csky:
  case Triple::hexagon:
  case Triple::lanai:
  case Triple::m68k:
  case Triple::msp430:
  case Triple::systemz:
  case Triple::xcore:
  case Triple::xtensa:
    return ExceptionHandling::DwarfCFI;
  default:
    break;
  }

  // Explicitly none targets.
  if (isWasm() || isAMDGPU() || isNVPTX() || isSPIROrSPIRV())
    return ExceptionHandling::None;

  // Default to none.
  return ExceptionHandling::None;
}

// HLSL triple environment orders are relied on in the front end
static_assert(Triple::Vertex - Triple::Pixel == 1,
              "incorrect HLSL stage order");
static_assert(Triple::Geometry - Triple::Pixel == 2,
              "incorrect HLSL stage order");
static_assert(Triple::Hull - Triple::Pixel == 3, "incorrect HLSL stage order");
static_assert(Triple::Domain - Triple::Pixel == 4,
              "incorrect HLSL stage order");
static_assert(Triple::Compute - Triple::Pixel == 5,
              "incorrect HLSL stage order");
static_assert(Triple::Library - Triple::Pixel == 6,
              "incorrect HLSL stage order");
static_assert(Triple::RayGeneration - Triple::Pixel == 7,
              "incorrect HLSL stage order");
static_assert(Triple::Intersection - Triple::Pixel == 8,
              "incorrect HLSL stage order");
static_assert(Triple::AnyHit - Triple::Pixel == 9,
              "incorrect HLSL stage order");
static_assert(Triple::ClosestHit - Triple::Pixel == 10,
              "incorrect HLSL stage order");
static_assert(Triple::Miss - Triple::Pixel == 11, "incorrect HLSL stage order");
static_assert(Triple::Callable - Triple::Pixel == 12,
              "incorrect HLSL stage order");
static_assert(Triple::Mesh - Triple::Pixel == 13, "incorrect HLSL stage order");
static_assert(Triple::Amplification - Triple::Pixel == 14,
              "incorrect HLSL stage order");
