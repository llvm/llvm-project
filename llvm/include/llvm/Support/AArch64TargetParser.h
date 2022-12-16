//===-- AArch64TargetParser - Parser for AArch64 features -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise AArch64 hardware features
// such as FPU/CPU/ARCH and extension names.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AARCH64TARGETPARSER_H
#define LLVM_SUPPORT_AARCH64TARGETPARSER_H

#include "llvm/ADT/StringRef.h"
#include <vector>

// FIXME:This should be made into class design,to avoid dupplication.
namespace llvm {

class Triple;

namespace AArch64 {

// Arch extension modifiers for CPUs. These are labelled with their Arm ARM
// feature name (though the canonical reference for those is AArch64.td)
// clang-format off
enum ArchExtKind : uint64_t {
  AEK_INVALID =     0,
  AEK_NONE =        1,
  AEK_CRC =         1 << 1,  // FEAT_CRC32
  AEK_CRYPTO =      1 << 2,
  AEK_FP =          1 << 3,  // FEAT_FP
  AEK_SIMD =        1 << 4,  // FEAT_AdvSIMD
  AEK_FP16 =        1 << 5,  // FEAT_FP16
  AEK_PROFILE =     1 << 6,  // FEAT_SPE
  AEK_RAS =         1 << 7,  // FEAT_RAS, FEAT_RASv1p1
  AEK_LSE =         1 << 8,  // FEAT_LSE
  AEK_SVE =         1 << 9,  // FEAT_SVE
  AEK_DOTPROD =     1 << 10, // FEAT_DotProd
  AEK_RCPC =        1 << 11, // FEAT_LRCPC
  AEK_RDM =         1 << 12, // FEAT_RDM
  AEK_SM4 =         1 << 13, // FEAT_SM4, FEAT_SM3
  AEK_SHA3 =        1 << 14, // FEAT_SHA3, FEAT_SHA512
  AEK_SHA2 =        1 << 15, // FEAT_SHA1, FEAT_SHA256
  AEK_AES =         1 << 16, // FEAT_AES, FEAT_PMULL
  AEK_FP16FML =     1 << 17, // FEAT_FHM
  AEK_RAND =        1 << 18, // FEAT_RNG
  AEK_MTE =         1 << 19, // FEAT_MTE, FEAT_MTE2
  AEK_SSBS =        1 << 20, // FEAT_SSBS, FEAT_SSBS2
  AEK_SB =          1 << 21, // FEAT_SB
  AEK_PREDRES =     1 << 22, // FEAT_SPECRES
  AEK_SVE2 =        1 << 23, // FEAT_SVE2
  AEK_SVE2AES =     1 << 24, // FEAT_SVE_AES, FEAT_SVE_PMULL128
  AEK_SVE2SM4 =     1 << 25, // FEAT_SVE_SM4
  AEK_SVE2SHA3 =    1 << 26, // FEAT_SVE_SHA3
  AEK_SVE2BITPERM = 1 << 27, // FEAT_SVE_BitPerm
  AEK_TME =         1 << 28, // FEAT_TME
  AEK_BF16 =        1 << 29, // FEAT_BF16
  AEK_I8MM =        1 << 30, // FEAT_I8MM
  AEK_F32MM =       1ULL << 31, // FEAT_F32MM
  AEK_F64MM =       1ULL << 32, // FEAT_F64MM
  AEK_LS64 =        1ULL << 33, // FEAT_LS64, FEAT_LS64_V, FEAT_LS64_ACCDATA
  AEK_BRBE =        1ULL << 34, // FEAT_BRBE
  AEK_PAUTH =       1ULL << 35, // FEAT_PAuth
  AEK_FLAGM =       1ULL << 36, // FEAT_FlagM
  AEK_SME =         1ULL << 37, // FEAT_SME
  AEK_SMEF64F64 =   1ULL << 38, // FEAT_SME_F64F64
  AEK_SMEI16I64 =   1ULL << 39, // FEAT_SME_I16I64
  AEK_HBC =         1ULL << 40, // FEAT_HBC
  AEK_MOPS =        1ULL << 41, // FEAT_MOPS
  AEK_PERFMON =     1ULL << 42, // FEAT_PMUv3
  AEK_SME2 =        1ULL << 43, // FEAT_SME2
  AEK_SVE2p1 =      1ULL << 44, // FEAT_SVE2p1
  AEK_SME2p1 =      1ULL << 45, // FEAT_SME2p1
  AEK_B16B16 =      1ULL << 46, // FEAT_B16B16
  AEK_SMEF16F16 =   1ULL << 47, // FEAT_SMEF16F16
  AEK_CSSC =        1ULL << 48, // FEAT_CSSC
  AEK_RCPC3 =       1ULL << 49, // FEAT_LRCPC3
  AEK_THE =         1ULL << 50, // FEAT_THE
  AEK_D128 =        1ULL << 51, // FEAT_D128
  AEK_LSE128 =      1ULL << 52, // FEAT_LSE128
  AEK_SPECRES2 =    1ULL << 53, // FEAT_SPECRES2
  AEK_RASv2 =       1ULL << 54, // FEAT_RASv2
};
// clang-format on

enum class ArchKind {
#define AARCH64_ARCH(NAME, ID, ARCH_FEATURE, ARCH_BASE_EXT) ID,
#include "AArch64TargetParser.def"
};

struct ArchNames {
  StringRef Name;
  StringRef ArchFeature;
  uint64_t ArchBaseExtensions;
  ArchKind ID;

  // Return ArchFeature without the leading "+".
  StringRef getSubArch() const { return ArchFeature.substr(1); }
};

const ArchNames AArch64ARCHNames[] = {
#define AARCH64_ARCH(NAME, ID, ARCH_FEATURE, ARCH_BASE_EXT)                    \
  {NAME, ARCH_FEATURE, ARCH_BASE_EXT, AArch64::ArchKind::ID},
#include "AArch64TargetParser.def"
};

// List of Arch Extension names.
struct ExtName {
  StringRef Name;
  uint64_t ID;
  StringRef Feature;
  StringRef NegFeature;
};

const ExtName AArch64ARCHExtNames[] = {
#define AARCH64_ARCH_EXT_NAME(NAME, ID, FEATURE, NEGFEATURE)                   \
  {NAME, ID, FEATURE, NEGFEATURE},
#include "AArch64TargetParser.def"
};

// List of CPU names and their arches.
// The same CPU can have multiple arches and can be default on multiple arches.
// When finding the Arch for a CPU, first-found prevails. Sort them accordingly.
// When this becomes table-generated, we'd probably need two tables.
struct CpuNames {
  StringRef Name;
  ArchKind ArchID;
  uint64_t DefaultExtensions;
};

const CpuNames AArch64CPUNames[] = {
#define AARCH64_CPU_NAME(NAME, ID, DEFAULT_EXT)                                \
  {NAME, AArch64::ArchKind::ID, DEFAULT_EXT},
#include "AArch64TargetParser.def"
};

const struct {
  StringRef Alias;
  StringRef Name;
} AArch64CPUAliases[] = {
#define AARCH64_CPU_ALIAS(ALIAS, NAME) {ALIAS, NAME},
#include "AArch64TargetParser.def"
};

const ArchKind ArchKinds[] = {
#define AARCH64_ARCH(NAME, ID, ARCH_FEATURE, ARCH_BASE_EXT) ArchKind::ID,
#include "AArch64TargetParser.def"
};

inline ArchKind &operator--(ArchKind &Kind) {
  if ((Kind == ArchKind::INVALID) || (Kind == ArchKind::ARMV8A) ||
      (Kind == ArchKind::ARMV9A) || (Kind == ArchKind::ARMV8R))
    Kind = ArchKind::INVALID;
  else {
    unsigned KindAsInteger = static_cast<unsigned>(Kind);
    Kind = static_cast<ArchKind>(--KindAsInteger);
  }
  return Kind;
}

bool getExtensionFeatures(uint64_t Extensions,
                          std::vector<StringRef> &Features);
StringRef getArchFeature(ArchKind AK);

StringRef getArchName(ArchKind AK);
StringRef getSubArch(ArchKind AK);
StringRef getArchExtName(unsigned ArchExtKind);
StringRef getArchExtFeature(StringRef ArchExt);
ArchKind convertV9toV8(ArchKind AK);
StringRef resolveCPUAlias(StringRef CPU);

// Information by Name
uint64_t getDefaultExtensions(StringRef CPU, ArchKind AK);
ArchKind getCPUArchKind(StringRef CPU);
ArchKind getSubArchArchKind(StringRef SubArch);

// Parser
ArchKind parseArch(StringRef Arch);
ArchExtKind parseArchExt(StringRef ArchExt);
ArchKind parseCPUArch(StringRef CPU);
// Used by target parser tests
void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values);

bool isX18ReservedByDefault(const Triple &TT);

} // namespace AArch64
} // namespace llvm

#endif
