//===-- llvm/BinaryFormat/MachO.cpp - The MachO file format -----*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/MachO.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

static MachO::CPUSubTypeX86 getX86SubType(const Triple &T) {
  assert(T.isX86());
  if (T.isArch32Bit())
    return MachO::CPU_SUBTYPE_I386_ALL;

  assert(T.isArch64Bit());
  if (T.getArchName() == "x86_64h")
    return MachO::CPU_SUBTYPE_X86_64_H;
  return MachO::CPU_SUBTYPE_X86_64_ALL;
}

static MachO::CPUSubTypeARM getARMSubType(const Triple &T) {
  assert(T.isARM() || T.isThumb());
  StringRef Arch = T.getArchName();
  ARM::ArchKind AK = ARM::parseArch(Arch);
  switch (AK) {
  default:
    return MachO::CPU_SUBTYPE_ARM_V7;
  case ARM::ArchKind::ARMV4T:
    return MachO::CPU_SUBTYPE_ARM_V4T;
  case ARM::ArchKind::ARMV5T:
  case ARM::ArchKind::ARMV5TE:
  case ARM::ArchKind::ARMV5TEJ:
    return MachO::CPU_SUBTYPE_ARM_V5;
  case ARM::ArchKind::ARMV6:
  case ARM::ArchKind::ARMV6K:
    return MachO::CPU_SUBTYPE_ARM_V6;
  case ARM::ArchKind::ARMV7A:
    return MachO::CPU_SUBTYPE_ARM_V7;
  case ARM::ArchKind::ARMV7S:
    return MachO::CPU_SUBTYPE_ARM_V7S;
  case ARM::ArchKind::ARMV7K:
    return MachO::CPU_SUBTYPE_ARM_V7K;
  case ARM::ArchKind::ARMV6M:
    return MachO::CPU_SUBTYPE_ARM_V6M;
  case ARM::ArchKind::ARMV7M:
    return MachO::CPU_SUBTYPE_ARM_V7M;
  case ARM::ArchKind::ARMV7EM:
    return MachO::CPU_SUBTYPE_ARM_V7EM;
  }
}

static MachO::CPUSubTypeARM64 getARM64SubType(const Triple &T) {
  assert(T.isAArch64());
  if (T.isArch32Bit())
    return (MachO::CPUSubTypeARM64)MachO::CPU_SUBTYPE_ARM64_32_V8;
  if (T.isArm64e())
    return MachO::CPU_SUBTYPE_ARM64E;

  return MachO::CPU_SUBTYPE_ARM64_ALL;
}

static MachO::CPUSubTypePowerPC getPowerPCSubType(const Triple &T) {
  return MachO::CPU_SUBTYPE_POWERPC_ALL;
}

static Error unsupported(const char *Str, const Triple &T) {
  return createStringError(std::errc::invalid_argument,
                           "Unsupported triple for mach-o cpu %s: %s", Str,
                           T.str().c_str());
}

Expected<uint32_t> MachO::getCPUType(const Triple &T) {
  if (!T.isOSBinFormatMachO())
    return unsupported("type", T);
  if (T.isX86() && T.isArch32Bit())
    return MachO::CPU_TYPE_X86;
  if (T.isX86() && T.isArch64Bit())
    return MachO::CPU_TYPE_X86_64;
  if (T.isARM() || T.isThumb())
    return MachO::CPU_TYPE_ARM;
  if (T.isAArch64())
    return T.isArch32Bit() ? MachO::CPU_TYPE_ARM64_32 : MachO::CPU_TYPE_ARM64;
  if (T.getArch() == Triple::ppc)
    return MachO::CPU_TYPE_POWERPC;
  if (T.getArch() == Triple::ppc64)
    return MachO::CPU_TYPE_POWERPC64;
  return unsupported("type", T);
}

Expected<uint32_t> MachO::getCPUSubType(const Triple &T) {
  if (!T.isOSBinFormatMachO())
    return unsupported("subtype", T);
  if (T.isX86())
    return getX86SubType(T);
  if (T.isARM() || T.isThumb())
    return getARMSubType(T);
  if (T.isAArch64() || T.getArch() == Triple::aarch64_32)
    return getARM64SubType(T);
  if (T.getArch() == Triple::ppc || T.getArch() == Triple::ppc64)
    return getPowerPCSubType(T);
  return unsupported("subtype", T);
}

Expected<uint32_t> MachO::getCPUSubType(const Triple &T,
                                        unsigned PtrAuthABIVersion,
                                        bool PtrAuthKernelABIVersion) {
  Expected<uint32_t> Result = MachO::getCPUSubType(T);
  if (!Result)
    return Result.takeError();
  if (*Result != MachO::CPU_SUBTYPE_ARM64E)
    return createStringError(
        std::errc::invalid_argument,
        "ptrauth ABI version is only supported on arm64e.");
  if (PtrAuthABIVersion > 0xF)
    return createStringError(
        std::errc::invalid_argument,
        "The ptrauth ABI version needs to fit within 4 bits.");
  return CPU_SUBTYPE_ARM64E_WITH_PTRAUTH_VERSION(PtrAuthABIVersion,
                                                 PtrAuthKernelABIVersion);
}

Triple::ArchType MachO::getArch(uint32_t CPUType, uint32_t CPUSubType) {
  switch (CPUType) {
  case MachO::CPU_TYPE_I386:
    return Triple::x86;
  case MachO::CPU_TYPE_X86_64:
    return Triple::x86_64;
  case MachO::CPU_TYPE_ARM:
    return Triple::arm;
  case MachO::CPU_TYPE_ARM64:
    return Triple::aarch64;
  case MachO::CPU_TYPE_ARM64_32:
    return Triple::aarch64_32;
  case MachO::CPU_TYPE_POWERPC:
    return Triple::ppc;
  case MachO::CPU_TYPE_POWERPC64:
    return Triple::ppc64;
  default:
    return Triple::UnknownArch;
  }
}

Triple MachO::getArchTriple(uint32_t CPUType, uint32_t CPUSubType,
                            const char **McpuDefault, const char **ArchFlag) {
  if (McpuDefault)
    *McpuDefault = nullptr;
  if (ArchFlag)
    *ArchFlag = nullptr;

  switch (CPUType) {
  case MachO::CPU_TYPE_I386:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_I386_ALL:
      if (ArchFlag)
        *ArchFlag = "i386";
      return Triple("i386-apple-darwin");
    default:
      return Triple();
    }
  case MachO::CPU_TYPE_X86_64:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_X86_64_ALL:
      if (ArchFlag)
        *ArchFlag = "x86_64";
      return Triple("x86_64-apple-darwin");
    case MachO::CPU_SUBTYPE_X86_64_H:
      if (ArchFlag)
        *ArchFlag = "x86_64h";
      return Triple("x86_64h-apple-darwin");
    default:
      return Triple();
    }
  case MachO::CPU_TYPE_ARM:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_ARM_V4T:
      if (ArchFlag)
        *ArchFlag = "armv4t";
      return Triple("armv4t-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V5TEJ:
      if (ArchFlag)
        *ArchFlag = "armv5e";
      return Triple("armv5e-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_XSCALE:
      if (ArchFlag)
        *ArchFlag = "xscale";
      return Triple("xscale-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V6:
      if (ArchFlag)
        *ArchFlag = "armv6";
      return Triple("armv6-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V6M:
      if (McpuDefault)
        *McpuDefault = "cortex-m0";
      if (ArchFlag)
        *ArchFlag = "armv6m";
      return Triple("armv6m-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7:
      if (ArchFlag)
        *ArchFlag = "armv7";
      return Triple("armv7-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7EM:
      if (McpuDefault)
        *McpuDefault = "cortex-m4";
      if (ArchFlag)
        *ArchFlag = "armv7em";
      return Triple("thumbv7em-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7K:
      if (McpuDefault)
        *McpuDefault = "cortex-a7";
      if (ArchFlag)
        *ArchFlag = "armv7k";
      return Triple("armv7k-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7M:
      if (McpuDefault)
        *McpuDefault = "cortex-m3";
      if (ArchFlag)
        *ArchFlag = "armv7m";
      return Triple("thumbv7m-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7S:
      if (McpuDefault)
        *McpuDefault = "cortex-a7";
      if (ArchFlag)
        *ArchFlag = "armv7s";
      return Triple("armv7s-apple-darwin");
    default:
      return Triple();
    }
  case MachO::CPU_TYPE_ARM64:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_ARM64_ALL:
      if (McpuDefault)
        *McpuDefault = "cyclone";
      if (ArchFlag)
        *ArchFlag = "arm64";
      return Triple("arm64-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM64E:
      if (McpuDefault)
        *McpuDefault = "apple-a12";
      if (ArchFlag)
        *ArchFlag = "arm64e";
      return Triple("arm64e-apple-darwin");
    default:
      return Triple();
    }
  case MachO::CPU_TYPE_ARM64_32:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_ARM64_32_V8:
      if (McpuDefault)
        *McpuDefault = "cyclone";
      if (ArchFlag)
        *ArchFlag = "arm64_32";
      return Triple("arm64_32-apple-darwin");
    default:
      return Triple();
    }
  case MachO::CPU_TYPE_POWERPC:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_POWERPC_ALL:
      if (ArchFlag)
        *ArchFlag = "ppc";
      return Triple("ppc-apple-darwin");
    default:
      return Triple();
    }
  case MachO::CPU_TYPE_POWERPC64:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_POWERPC_ALL:
      if (ArchFlag)
        *ArchFlag = "ppc64";
      return Triple("ppc64-apple-darwin");
    default:
      return Triple();
    }
  default:
    return Triple();
  }
}
