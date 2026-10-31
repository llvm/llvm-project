//===- llvm/BinaryFormat/ELF.cpp - The ELF format ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ADT/Enum.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;
using namespace ELF;

constexpr EnumStringDef<uint16_t> ElfMachineArchDefs[] = {
#define ELF_MACHINE(em, num, arch, gnuname) {{arch}, em},
#include "llvm/BinaryFormat/ELF.def"
};
constexpr auto ElfMachineArchs = BUILD_ENUM_STRINGS(ElfMachineArchDefs);

/// Convert an architecture name into ELF's e_machine value.
uint16_t ELF::convertArchNameToEMachine(StringRef Arch) {
  std::string LowerArch = Arch.lower();
  if (const auto *E = EnumStrings(ElfMachineArchs).fromString(LowerArch))
    return E->value();
  return EM_NONE;
}

uint16_t ELF::convertTripleArchTypeToEMachine(Triple::ArchType ArchType) {
  switch (ArchType) {
  case Triple::UnknownArch:
  default:
    return EM_NONE;

  case Triple::arm:
  case Triple::armeb:
  case Triple::thumb:
  case Triple::thumbeb:
    return EM_ARM;
  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::aarch64_32:
    return EM_AARCH64;
  case Triple::arc:
    return EM_ARC;
  case Triple::avr:
    return EM_AVR;
  case Triple::bpfel:
  case Triple::bpfeb:
    return EM_BPF;
  case Triple::csky:
    return EM_CSKY;
  case Triple::hexagon:
    return EM_HEXAGON;
  case Triple::loongarch32:
  case Triple::loongarch64:
    return EM_LOONGARCH;
  case Triple::m68k:
    return EM_68K;
  case Triple::mips:
  case Triple::mipsel:
  case Triple::mips64:
  case Triple::mips64el:
    return EM_MIPS;
  case Triple::msp430:
    return EM_MSP430;
  case Triple::ppc:
  case Triple::ppcle:
    return EM_PPC;
  case Triple::ppc64:
  case Triple::ppc64le:
    return EM_PPC;
  case Triple::riscv32:
  case Triple::riscv64:
    return EM_RISCV;
  case Triple::sparc:
  case Triple::sparcel:
    return EM_SPARC;
  case Triple::sparcv9:
    return EM_SPARCV9;
  case Triple::systemz:
    return EM_S390;
  case Triple::x86:
    return EM_386;
  case Triple::x86_64:
    return EM_X86_64;
  case Triple::xcore:
    return EM_XCORE;
  case Triple::xtensa:
    return EM_XTENSA;
  }
}

/// Convert an ELF's e_machine value into an architecture name.
StringRef ELF::convertEMachineToArchName(uint16_t EMachine) {
  if (StringRef S = EnumStrings(ElfMachineArchs).toString(EMachine); !S.empty())
    return S;
  return "None";
}

uint8_t ELF::convertNameToOSABI(StringRef Name) {
  return StringSwitch<uint16_t>(Name)
      .StartsWith("hpux", ELFOSABI_HPUX)
      .StartsWith("netbsd", ELFOSABI_NETBSD)
      .StartsWith("gnu", ELFOSABI_GNU)
      .StartsWith("hurd", ELFOSABI_HURD)
      .StartsWith("solaris", ELFOSABI_SOLARIS)
      .StartsWith("aix", ELFOSABI_AIX)
      .StartsWith("irix", ELFOSABI_IRIX)
      .StartsWith("freebsd", ELFOSABI_FREEBSD)
      .StartsWith("tru64", ELFOSABI_TRU64)
      .StartsWith("modesto", ELFOSABI_MODESTO)
      .StartsWith("openbsd", ELFOSABI_OPENBSD)
      .StartsWith("openvms", ELFOSABI_OPENVMS)
      .StartsWith("nsk", ELFOSABI_NSK)
      .StartsWith("aros", ELFOSABI_AROS)
      .StartsWith("fenixos", ELFOSABI_FENIXOS)
      .StartsWith("cloudabi", ELFOSABI_CLOUDABI)
      .StartsWith("cuda", ELFOSABI_CUDA)
      .StartsWith("amdhsa", ELFOSABI_AMDGPU_HSA)
      .StartsWith("amdpal", ELFOSABI_AMDGPU_PAL)
      .StartsWith("mesa3d", ELFOSABI_AMDGPU_MESA3D)
      .StartsWith("arm", ELFOSABI_ARM)
      .StartsWith("standalone", ELFOSABI_STANDALONE)
      .StartsWith("none", ELFOSABI_NONE)
      .Default(ELFOSABI_NONE);
}

StringRef ELF::convertOSABIToName(uint8_t OSABI) {
  switch (OSABI) {
  case ELFOSABI_HPUX:
    return "hpux";
  case ELFOSABI_NETBSD:
    return "netbsd";
  case ELFOSABI_GNU:
    return "gnu";
  case ELFOSABI_HURD:
    return "hurd";
  case ELFOSABI_SOLARIS:
    return "solaris";
  case ELFOSABI_AIX:
    return "aix";
  case ELFOSABI_IRIX:
    return "irix";
  case ELFOSABI_FREEBSD:
    return "freebsd";
  case ELFOSABI_TRU64:
    return "tru64";
  case ELFOSABI_MODESTO:
    return "modesto";
  case ELFOSABI_OPENBSD:
    return "openbsd";
  case ELFOSABI_OPENVMS:
    return "openvms";
  case ELFOSABI_NSK:
    return "nsk";
  case ELFOSABI_AROS:
    return "aros";
  case ELFOSABI_FENIXOS:
    return "fenixos";
  case ELFOSABI_CLOUDABI:
    return "cloudabi";
  case ELFOSABI_CUDA:
    return "cuda";
  case ELFOSABI_AMDGPU_HSA:
    return "amdhsa";
  case ELFOSABI_AMDGPU_PAL:
    return "amdpal";
  case ELFOSABI_AMDGPU_MESA3D:
    return "mesa3d";
  case ELFOSABI_ARM:
    return "arm";
  case ELFOSABI_STANDALONE:
    return "standalone";
  default:
    return "none";
  }
}
