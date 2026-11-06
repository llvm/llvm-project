//===-- ARMTargetParser - Parser for ARM target features --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise ARM hardware features
// such as FPU/CPU/ARCH/extensions and specific support such as HWDIV.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_ARMTARGETPARSER_H
#define LLVM_TARGETPARSER_ARMTARGETPARSER_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/ARMTargetParserCommon.h"
#include <vector>

namespace llvm {

template <typename, unsigned> class EnumStrings;
class Triple;

namespace ARM {

enum ARMABI {
  ARM_ABI_UNKNOWN,
  ARM_ABI_APCS,
  ARM_ABI_AAPCS, // ARM EABI
  ARM_ABI_AAPCS16
};

// Arch extension modifiers for CPUs.
// Note that this is not the same as the AArch64 list
enum ArchExtKind : uint64_t {
  AEK_INVALID = 0,
  AEK_NONE = 1,
  AEK_CRC = 1 << 1,
  AEK_CRYPTO = 1 << 2,
  AEK_FP = 1 << 3,
  AEK_HWDIVTHUMB = 1 << 4,
  AEK_HWDIVARM = 1 << 5,
  AEK_MP = 1 << 6,
  AEK_SIMD = 1 << 7,
  AEK_SEC = 1 << 8,
  AEK_VIRT = 1 << 9,
  AEK_DSP = 1 << 10,
  AEK_FP16 = 1 << 11,
  AEK_RAS = 1 << 12,
  AEK_DOTPROD = 1 << 13,
  AEK_SHA2 = 1 << 14,
  AEK_AES = 1 << 15,
  AEK_FP16FML = 1 << 16,
  AEK_SB = 1 << 17,
  AEK_FP_DP = 1 << 18,
  AEK_LOB = 1 << 19,
  AEK_BF16 = 1 << 20,
  AEK_I8MM = 1 << 21,
  AEK_CDECP0 = 1 << 22,
  AEK_CDECP1 = 1 << 23,
  AEK_CDECP2 = 1 << 24,
  AEK_CDECP3 = 1 << 25,
  AEK_CDECP4 = 1 << 26,
  AEK_CDECP5 = 1 << 27,
  AEK_CDECP6 = 1 << 28,
  AEK_CDECP7 = 1 << 29,
  AEK_PACBTI = 1 << 30,
  AEK_MVE = 1ULL << 31,
  // Unsupported extensions.
  AEK_OS = 1ULL << 59,
  AEK_IWMMXT = 1ULL << 60,
  AEK_IWMMXT2 = 1ULL << 61,
  AEK_MAVERICK = 1ULL << 62,
  AEK_XSCALE = 1ULL << 63,
};

// Arch names.
enum class ArchKind {
#define ARM_ARCH(NAME, ID, CPU_ATTR, ARCH_FEATURE, ARCH_ATTR, ARCH_FPU,        \
                 ARCH_BASE_EXT)                                                \
  ID,
#include "ARMTargetParser.def"
};

// FPU names.
enum FPUKind {
#define ARM_FPU(NAME, KIND, VERSION, NEON_SUPPORT, RESTRICTION) KIND,
#include "ARMTargetParser.def"
  FK_LAST
};

// FPU Version
enum class FPUVersion {
  NONE,
  VFPV2,
  VFPV3,
  VFPV3_FP16,
  VFPV4,
  VFPV5,
  VFPV5_FULLFP16,
};

// An FPU name restricts the FPU in one of three ways:
enum class FPURestriction {
  None = 0, ///< No restriction
  D16,      ///< Only 16 D registers
  SP_D16    ///< Only single-precision instructions, with 16 D registers
};

inline bool isDoublePrecision(const FPURestriction restriction) {
  return restriction != FPURestriction::SP_D16;
}

inline bool has32Regs(const FPURestriction restriction) {
  return restriction == FPURestriction::None;
}

// An FPU name implies one of three levels of Neon support:
enum class NeonSupportLevel {
  None = 0, ///< No Neon
  Neon,     ///< Neon
  Crypto    ///< Neon with Crypto
};

// v6/v7/v8 Profile
enum class ProfileKind { INVALID = 0, A, R, M };

inline ArchKind &operator--(ArchKind &Kind) {
  assert((Kind >= ArchKind::ARMV8A && Kind <= ArchKind::ARMV9_3A) &&
         "We only expect operator-- to be called with ARMV8/V9");
  if (Kind == ArchKind::INVALID || Kind == ArchKind::ARMV8A ||
      Kind == ArchKind::ARMV8_1A || Kind == ArchKind::ARMV9A ||
      Kind == ArchKind::ARMV8R)
    Kind = ArchKind::INVALID;
  else {
    unsigned KindAsInteger = static_cast<unsigned>(Kind);
    Kind = static_cast<ArchKind>(--KindAsInteger);
  }
  return Kind;
}

// Information by ID
LLVM_ABI StringRef getFPUName(FPUKind FPUKind);
LLVM_ABI FPUVersion getFPUVersion(FPUKind FPUKind);
LLVM_ABI NeonSupportLevel getFPUNeonSupportLevel(FPUKind FPUKind);
LLVM_ABI FPURestriction getFPURestriction(FPUKind FPUKind);

LLVM_ABI bool getFPUFeatures(FPUKind FPUKind, std::vector<StringRef> &Features);
LLVM_ABI bool getHWDivFeatures(uint64_t HWDivKind,
                               std::vector<StringRef> &Features);
LLVM_ABI bool getExtensionFeatures(uint64_t Extensions,
                                   std::vector<StringRef> &Features);

LLVM_ABI StringRef getArchName(ArchKind AK);
LLVM_ABI unsigned getArchAttr(ArchKind AK);
LLVM_ABI StringRef getCPUAttr(ArchKind AK);
LLVM_ABI StringRef getSubArch(ArchKind AK);
/// Get list of architecture extensions. The name strings are: 0=Name,
/// 1=PosFeature (incl. "+"), 2=NegFeature (incl. "-").
LLVM_ABI EnumStrings<ArchExtKind, 3> getArchExts();
LLVM_ABI StringRef getArchExtName(uint64_t ArchExtKind);
LLVM_ABI StringRef getArchExtFeature(StringRef ArchExt);
LLVM_ABI bool appendArchExtFeatures(StringRef CPU, ARM::ArchKind AK,
                                    StringRef ArchExt,
                                    std::vector<StringRef> &Features,
                                    FPUKind &ArgFPUKind);
LLVM_ABI ArchKind convertV9toV8(ArchKind AK);

// Information by Name
LLVM_ABI FPUKind getDefaultFPU(StringRef CPU, ArchKind AK);
LLVM_ABI uint64_t getDefaultExtensions(StringRef CPU, ArchKind AK);
LLVM_ABI StringRef getDefaultCPU(StringRef Arch);
LLVM_ABI StringRef getFPUSynonym(StringRef FPU);

// Parser
LLVM_ABI uint64_t parseHWDiv(StringRef HWDiv);
LLVM_ABI FPUKind parseFPU(StringRef FPU);
LLVM_ABI ArchKind parseArch(StringRef Arch);
LLVM_ABI uint64_t parseArchExt(StringRef ArchExt);
LLVM_ABI ArchKind parseCPUArch(StringRef CPU);
LLVM_ABI ProfileKind parseArchProfile(StringRef Arch);
LLVM_ABI unsigned parseArchVersion(StringRef Arch);

LLVM_ABI void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values);
LLVM_ABI LLVM_READONLY StringRef computeDefaultTargetABI(const Triple &TT);
LLVM_ABI LLVM_READONLY ARMABI computeTargetABI(const Triple &TT,
                                               StringRef ABIName = "");

/// Get the (LLVM) name of the minimum ARM CPU for the arch we are targeting.
///
/// \param Arch the architecture name (e.g., "armv7s"). If it is an empty
/// string then the triple's arch name is used.
LLVM_ABI StringRef getARMCPUForArch(const llvm::Triple &Triple,
                                    StringRef MArch = {});

LLVM_ABI void PrintSupportedExtensions(StringMap<StringRef> DescMap);

} // namespace ARM
} // namespace llvm

#endif
