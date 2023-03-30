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

#ifndef LLVM_TARGETPARSER_AARCH64TARGETPARSER_H
#define LLVM_TARGETPARSER_AARCH64TARGETPARSER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VersionTuple.h"
#include <array>
#include <vector>

namespace llvm {

class Triple;

namespace AArch64 {
enum CPUFeatures {
  FEAT_RNG,
  FEAT_FLAGM,
  FEAT_FLAGM2,
  FEAT_FP16FML,
  FEAT_DOTPROD,
  FEAT_SM4,
  FEAT_RDM,
  FEAT_LSE,
  FEAT_FP,
  FEAT_SIMD,
  FEAT_CRC,
  FEAT_SHA1,
  FEAT_SHA2,
  FEAT_SHA3,
  FEAT_AES,
  FEAT_PMULL,
  FEAT_FP16,
  FEAT_DIT,
  FEAT_DPB,
  FEAT_DPB2,
  FEAT_JSCVT,
  FEAT_FCMA,
  FEAT_RCPC,
  FEAT_RCPC2,
  FEAT_FRINTTS,
  FEAT_DGH,
  FEAT_I8MM,
  FEAT_BF16,
  FEAT_EBF16,
  FEAT_RPRES,
  FEAT_SVE,
  FEAT_SVE_BF16,
  FEAT_SVE_EBF16,
  FEAT_SVE_I8MM,
  FEAT_SVE_F32MM,
  FEAT_SVE_F64MM,
  FEAT_SVE2,
  FEAT_SVE_AES,
  FEAT_SVE_PMULL128,
  FEAT_SVE_BITPERM,
  FEAT_SVE_SHA3,
  FEAT_SVE_SM4,
  FEAT_SME,
  FEAT_MEMTAG,
  FEAT_MEMTAG2,
  FEAT_MEMTAG3,
  FEAT_SB,
  FEAT_PREDRES,
  FEAT_SSBS,
  FEAT_SSBS2,
  FEAT_BTI,
  FEAT_LS64,
  FEAT_LS64_V,
  FEAT_LS64_ACCDATA,
  FEAT_WFXT,
  FEAT_SME_F64,
  FEAT_SME_I64,
  FEAT_SME2,
  FEAT_MAX
};

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
  AEK_ITE =         1ULL << 55, // FEAT_ITE
};
// clang-format on

// Represents an extension that can be enabled with -march=<arch>+<extension>.
// Typically these correspond to Arm Architecture extensions, unlike
// SubtargetFeature which may represent either an actual extension or some
// internal LLVM property.
struct ExtensionInfo {
  StringRef Name;       // Human readable name, e.g. "profile".
  ArchExtKind ID;       // Corresponding to the ArchExtKind, this extensions
                        // representation in the bitfield.
  StringRef Feature;    // -mattr enable string, e.g. "+spe"
  StringRef NegFeature; // -mattr disable string, e.g. "-spe"

  // FIXME These were added by D127812 FMV support and need documenting:
  CPUFeatures CPUFeature; // Bitfield value set in __aarch64_cpu_features
  StringRef DependentFeatures;
  unsigned FmvPriority;
  static constexpr unsigned MaxFMVPriority = 1000;
};

// clang-format off
inline constexpr ExtensionInfo Extensions[] = {
    {"aes", AArch64::AEK_AES, "+aes", "-aes", FEAT_AES, "+fp-armv8,+neon", 150},
    {"b16b16", AArch64::AEK_B16B16, "+b16b16", "-b16b16", FEAT_MAX, "", 0},
    {"bf16", AArch64::AEK_BF16, "+bf16", "-bf16", FEAT_BF16, "+bf16", 280},
    {"brbe", AArch64::AEK_BRBE, "+brbe", "-brbe", FEAT_MAX, "", 0},
    {"bti", AArch64::AEK_NONE, {}, {}, FEAT_BTI, "+bti", 510},
    {"crc", AArch64::AEK_CRC, "+crc", "-crc", FEAT_CRC, "+crc", 110},
    {"crypto", AArch64::AEK_CRYPTO, "+crypto", "-crypto", FEAT_MAX, "+aes,+sha2", 0},
    {"cssc", AArch64::AEK_CSSC, "+cssc", "-cssc", FEAT_MAX, "", 0},
    {"d128", AArch64::AEK_D128, "+d128", "-d128", FEAT_MAX, "", 0},
    {"dgh", AArch64::AEK_NONE, {}, {}, FEAT_DGH, "", 260},
    {"dit", AArch64::AEK_NONE, {}, {}, FEAT_DIT, "+dit", 180},
    {"dotprod", AArch64::AEK_DOTPROD, "+dotprod", "-dotprod", FEAT_DOTPROD, "+dotprod,+fp-armv8,+neon", 50},
    {"dpb", AArch64::AEK_NONE, {}, {}, FEAT_DPB, "+ccpp", 190},
    {"dpb2", AArch64::AEK_NONE, {}, {}, FEAT_DPB2, "+ccpp,+ccdp", 200},
    {"ebf16", AArch64::AEK_NONE, {}, {}, FEAT_EBF16, "+bf16", 290},
    {"f32mm", AArch64::AEK_F32MM, "+f32mm", "-f32mm", FEAT_SVE_F32MM, "+sve,+f32mm,+fullfp16,+fp-armv8,+neon", 350},
    {"f64mm", AArch64::AEK_F64MM, "+f64mm", "-f64mm", FEAT_SVE_F64MM, "+sve,+f64mm,+fullfp16,+fp-armv8,+neon", 360},
    {"fcma", AArch64::AEK_NONE, {}, {}, FEAT_FCMA, "+fp-armv8,+neon,+complxnum", 220},
    {"flagm", AArch64::AEK_FLAGM, "+flagm", "-flagm", FEAT_FLAGM, "+flagm", 20},
    {"flagm2", AArch64::AEK_NONE, {}, {}, FEAT_FLAGM2, "+flagm,+altnzcv", 30},
    {"fp", AArch64::AEK_FP, "+fp-armv8", "-fp-armv8", FEAT_FP, "+fp-armv8,+neon", 90},
    {"fp16", AArch64::AEK_FP16, "+fullfp16", "-fullfp16", FEAT_FP16, "+fullfp16,+fp-armv8,+neon", 170},
    {"fp16fml", AArch64::AEK_FP16FML, "+fp16fml", "-fp16fml", FEAT_FP16FML, "+fp16fml,+fullfp16,+fp-armv8,+neon", 40},
    {"frintts", AArch64::AEK_NONE, {}, {}, FEAT_FRINTTS, "+fptoint", 250},
    {"hbc", AArch64::AEK_HBC, "+hbc", "-hbc", FEAT_MAX, "", 0},
    {"i8mm", AArch64::AEK_I8MM, "+i8mm", "-i8mm", FEAT_I8MM, "+i8mm", 270},
    {"ite", AArch64::AEK_ITE, "+ite", "-ite", FEAT_MAX, "", 0},
    {"jscvt", AArch64::AEK_NONE, {}, {}, FEAT_JSCVT, "+fp-armv8,+neon,+jsconv", 210},
    {"ls64_accdata", AArch64::AEK_NONE, {}, {}, FEAT_LS64_ACCDATA, "+ls64", 540},
    {"ls64_v", AArch64::AEK_NONE, {}, {}, FEAT_LS64_V, "", 530},
    {"ls64", AArch64::AEK_LS64, "+ls64", "-ls64", FEAT_LS64, "", 520},
    {"lse", AArch64::AEK_LSE, "+lse", "-lse", FEAT_LSE, "+lse", 80},
    {"lse128", AArch64::AEK_LSE128, "+lse128", "-lse128", FEAT_MAX, "", 0},
    {"memtag", AArch64::AEK_MTE, "+mte", "-mte", FEAT_MEMTAG, "", 440},
    {"memtag2", AArch64::AEK_NONE, {}, {}, FEAT_MEMTAG2, "+mte", 450},
    {"memtag3", AArch64::AEK_NONE, {}, {}, FEAT_MEMTAG3, "+mte", 460},
    {"mops", AArch64::AEK_MOPS, "+mops", "-mops", FEAT_MAX, "", 0},
    {"pauth", AArch64::AEK_PAUTH, "+pauth", "-pauth", FEAT_MAX, "", 0},
    {"pmull", AArch64::AEK_NONE, {}, {}, FEAT_PMULL, "+aes,+fp-armv8,+neon", 160},
    {"pmuv3", AArch64::AEK_PERFMON, "+perfmon", "-perfmon", FEAT_MAX, "", 0},
    {"predres", AArch64::AEK_PREDRES, "+predres", "-predres", FEAT_PREDRES, "+predres", 480},
    {"predres2", AArch64::AEK_SPECRES2, "+specres2", "-specres2", FEAT_MAX, "", 0},
    {"profile", AArch64::AEK_PROFILE, "+spe", "-spe", FEAT_MAX, "", 0},
    {"ras", AArch64::AEK_RAS, "+ras", "-ras", FEAT_MAX, "", 0},
    {"rasv2", AArch64::AEK_RASv2, "+rasv2", "-rasv2", FEAT_MAX, "", 0},
    {"rcpc", AArch64::AEK_RCPC, "+rcpc", "-rcpc", FEAT_RCPC, "+rcpc", 230},
    {"rcpc2", AArch64::AEK_NONE, {}, {}, FEAT_RCPC2, "+rcpc", 240},
    {"rcpc3", AArch64::AEK_RCPC3, "+rcpc3", "-rcpc3", FEAT_MAX, "", 0},
    {"rdm", AArch64::AEK_RDM, "+rdm", "-rdm", FEAT_RDM, "+rdm,+fp-armv8,+neon", 70},
    {"rng", AArch64::AEK_RAND, "+rand", "-rand", FEAT_RNG, "+rand", 10},
    {"rpres", AArch64::AEK_NONE, {}, {}, FEAT_RPRES, "", 300},
    {"sb", AArch64::AEK_SB, "+sb", "-sb", FEAT_SB, "+sb", 470},
    {"sha1", AArch64::AEK_NONE, {}, {}, FEAT_SHA1, "+fp-armv8,+neon", 120},
    {"sha2", AArch64::AEK_SHA2, "+sha2", "-sha2", FEAT_SHA2, "+sha2,+fp-armv8,+neon", 130},
    {"sha3", AArch64::AEK_SHA3, "+sha3", "-sha3", FEAT_SHA3, "+sha3,+sha2,+fp-armv8,+neon", 140},
    {"simd", AArch64::AEK_SIMD, "+neon", "-neon", FEAT_SIMD, "+fp-armv8,+neon", 100},
    {"sm4", AArch64::AEK_SM4, "+sm4", "-sm4", FEAT_SM4, "+sm4,+fp-armv8,+neon", 60},
    {"sme-f16f16", AArch64::AEK_SMEF16F16, "+sme-f16f16", "-sme-f16f16", FEAT_MAX, "", 0},
    {"sme-f64f64", AArch64::AEK_SMEF64F64, "+sme-f64f64", "-sme-f64f64", FEAT_SME_F64, "+sme,+sme-f64f64,+bf16", 560},
    {"sme-i16i64", AArch64::AEK_SMEI16I64, "+sme-i16i64", "-sme-i16i64", FEAT_SME_I64, "+sme,+sme-i16i64,+bf16", 570},
    {"sme", AArch64::AEK_SME, "+sme", "-sme", FEAT_SME, "+sme,+bf16", 430},
    {"sme2", AArch64::AEK_SME2, "+sme2", "-sme2", FEAT_SME2, "+sme2,+sme,+bf16", 580},
    {"sme2p1", AArch64::AEK_SME2p1, "+sme2p1", "-sme2p1", FEAT_MAX, "", 0},
    {"ssbs", AArch64::AEK_SSBS, "+ssbs", "-ssbs", FEAT_SSBS, "", 490},
    {"ssbs2", AArch64::AEK_NONE, {}, {}, FEAT_SSBS2, "+ssbs", 500},
    {"sve-bf16", AArch64::AEK_NONE, {}, {}, FEAT_SVE_BF16, "+sve,+bf16,+fullfp16,+fp-armv8,+neon", 320},
    {"sve-ebf16", AArch64::AEK_NONE, {}, {}, FEAT_SVE_EBF16, "+sve,+bf16,+fullfp16,+fp-armv8,+neon", 330},
    {"sve-i8mm", AArch64::AEK_NONE, {}, {}, FEAT_SVE_I8MM, "+sve,+i8mm,+fullfp16,+fp-armv8,+neon", 340},
    {"sve", AArch64::AEK_SVE, "+sve", "-sve", FEAT_SVE, "+sve,+fullfp16,+fp-armv8,+neon", 310},
    {"sve2-aes", AArch64::AEK_SVE2AES, "+sve2-aes", "-sve2-aes", FEAT_SVE_AES, "+sve2,+sve,+sve2-aes,+fullfp16,+fp-armv8,+neon", 380},
    {"sve2-bitperm", AArch64::AEK_SVE2BITPERM, "+sve2-bitperm", "-sve2-bitperm", FEAT_SVE_BITPERM, "+sve2,+sve,+sve2-bitperm,+fullfp16,+fp-armv8,+neon", 400},
    {"sve2-pmull128", AArch64::AEK_NONE, {}, {}, FEAT_SVE_PMULL128, "+sve2,+sve,+sve2-aes,+fullfp16,+fp-armv8,+neon", 390},
    {"sve2-sha3", AArch64::AEK_SVE2SHA3, "+sve2-sha3", "-sve2-sha3", FEAT_SVE_SHA3, "+sve2,+sve,+sve2-sha3,+fullfp16,+fp-armv8,+neon", 410},
    {"sve2-sm4", AArch64::AEK_SVE2SM4, "+sve2-sm4", "-sve2-sm4", FEAT_SVE_SM4, "+sve2,+sve,+sve2-sm4,+fullfp16,+fp-armv8,+neon", 420},
    {"sve2", AArch64::AEK_SVE2, "+sve2", "-sve2", FEAT_SVE2, "+sve2,+sve,+fullfp16,+fp-armv8,+neon", 370},
    {"sve2p1", AArch64::AEK_SVE2p1, "+sve2p1", "-sve2p1", FEAT_MAX, "", 0},
    {"the", AArch64::AEK_THE, "+the", "-the", FEAT_MAX, "", 0},
    {"tme", AArch64::AEK_TME, "+tme", "-tme", FEAT_MAX, "", 0},
    {"wfxt", AArch64::AEK_NONE, {}, {}, FEAT_WFXT, "+wfxt", 550},
    // Special cases
    {"none", AArch64::AEK_NONE, {}, {}, FEAT_MAX, "", ExtensionInfo::MaxFMVPriority},
    {"invalid", AArch64::AEK_INVALID, {}, {}, FEAT_MAX, "", 0},
};
// clang-format on

enum ArchProfile { AProfile = 'A', RProfile = 'R', InvalidProfile = '?' };

// Information about a specific architecture, e.g. V8.1-A
struct ArchInfo {
  VersionTuple Version;  // Architecture version, major + minor.
  ArchProfile Profile;   // Architecuture profile
  StringRef Name;        // Human readable name, e.g. "armv8.1-a"
  StringRef ArchFeature; // Command line feature flag, e.g. +v8a
  uint64_t DefaultExts;  // bitfield of default extensions ArchExtKind

  bool operator==(const ArchInfo &Other) const {
    return this->Name == Other.Name;
  }
  bool operator!=(const ArchInfo &Other) const {
    return this->Name != Other.Name;
  }

  // Defines the following partial order, indicating when an architecture is
  // a superset of another:
  //
  //     v9.4a > v9.3a > v9.3a > v9.3a > v9a;
  //       v       v       v       v       v
  //     v8.9a > v8.8a > v8.7a > v8.6a > v8.5a > v8.4a > ... > v8a;
  //
  // v8r and INVALID have no relation to anything. This is used to
  // determine which features to enable for a given architecture. See
  // AArch64TargetInfo::setFeatureEnabled.
  bool implies(const ArchInfo &Other) const {
    if (this->Profile != Other.Profile)
      return false; // ARMV8R and INVALID
    if (this->Version.getMajor() == Other.Version.getMajor()) {
      return this->Version > Other.Version;
    }
    if (this->Version.getMajor() == 9 && Other.Version.getMajor() == 8) {
      return this->Version.getMinor().value() + 5 >=
             Other.Version.getMinor().value();
    }
    return false;
  }

  // Return ArchFeature without the leading "+".
  StringRef getSubArch() const { return ArchFeature.substr(1); }

  // Search for ArchInfo by SubArch name
  static const ArchInfo &findBySubArch(StringRef SubArch);
};

// clang-format off
inline constexpr ArchInfo INVALID   = { VersionTuple{0, 0}, AProfile, "invalid", "+", (AArch64::AEK_NONE)};
inline constexpr ArchInfo ARMV8A    = { VersionTuple{8, 0}, AProfile, "armv8-a", "+v8a", (AArch64::AEK_FP | AArch64::AEK_SIMD), };
inline constexpr ArchInfo ARMV8_1A  = { VersionTuple{8, 1}, AProfile, "armv8.1-a", "+v8.1a", (ARMV8A.DefaultExts | AArch64::AEK_CRC | AArch64::AEK_LSE | AArch64::AEK_RDM)};
inline constexpr ArchInfo ARMV8_2A  = { VersionTuple{8, 2}, AProfile, "armv8.2-a", "+v8.2a", (ARMV8_1A.DefaultExts | AArch64::AEK_RAS)};
inline constexpr ArchInfo ARMV8_3A  = { VersionTuple{8, 3}, AProfile, "armv8.3-a", "+v8.3a", (ARMV8_2A.DefaultExts | AArch64::AEK_RCPC)};
inline constexpr ArchInfo ARMV8_4A  = { VersionTuple{8, 4}, AProfile, "armv8.4-a", "+v8.4a", (ARMV8_3A.DefaultExts | AArch64::AEK_DOTPROD)};
inline constexpr ArchInfo ARMV8_5A  = { VersionTuple{8, 5}, AProfile, "armv8.5-a", "+v8.5a", (ARMV8_4A.DefaultExts)};
inline constexpr ArchInfo ARMV8_6A  = { VersionTuple{8, 6}, AProfile, "armv8.6-a", "+v8.6a", (ARMV8_5A.DefaultExts | AArch64::AEK_BF16 | AArch64::AEK_I8MM)};
inline constexpr ArchInfo ARMV8_7A  = { VersionTuple{8, 7}, AProfile, "armv8.7-a", "+v8.7a", (ARMV8_6A.DefaultExts)};
inline constexpr ArchInfo ARMV8_8A  = { VersionTuple{8, 8}, AProfile, "armv8.8-a", "+v8.8a", (ARMV8_7A.DefaultExts | AArch64::AEK_MOPS | AArch64::AEK_HBC)};
inline constexpr ArchInfo ARMV8_9A  = { VersionTuple{8, 9}, AProfile, "armv8.9-a", "+v8.9a", (ARMV8_8A.DefaultExts | AArch64::AEK_SPECRES2 | AArch64::AEK_CSSC | AArch64::AEK_RASv2)};
inline constexpr ArchInfo ARMV9A    = { VersionTuple{9, 0}, AProfile, "armv9-a", "+v9a", (ARMV8_5A.DefaultExts | AArch64::AEK_FP16 | AArch64::AEK_SVE | AArch64::AEK_SVE2)};
inline constexpr ArchInfo ARMV9_1A  = { VersionTuple{9, 1}, AProfile, "armv9.1-a", "+v9.1a", (ARMV9A.DefaultExts | AArch64::AEK_BF16 | AArch64::AEK_I8MM)};
inline constexpr ArchInfo ARMV9_2A  = { VersionTuple{9, 2}, AProfile, "armv9.2-a", "+v9.2a", (ARMV9_1A.DefaultExts)};
inline constexpr ArchInfo ARMV9_3A  = { VersionTuple{9, 3}, AProfile, "armv9.3-a", "+v9.3a", (ARMV9_2A.DefaultExts | AArch64::AEK_MOPS | AArch64::AEK_HBC)};
inline constexpr ArchInfo ARMV9_4A  = { VersionTuple{9, 4}, AProfile, "armv9.4-a", "+v9.4a", (ARMV9_3A.DefaultExts | AArch64::AEK_SPECRES2 | AArch64::AEK_CSSC | AArch64::AEK_RASv2)};
// For v8-R, we do not enable crypto and align with GCC that enables a more minimal set of optional architecture extensions.
inline constexpr ArchInfo ARMV8R    = { VersionTuple{8, 0}, RProfile, "armv8-r", "+v8r", ((ARMV8_5A.DefaultExts ^ AArch64::AEK_LSE) | AArch64::AEK_SSBS | AArch64::AEK_FP16 | AArch64::AEK_FP16FML | AArch64::AEK_SB), };
// clang-format on

// The set of all architectures
static constexpr std::array<const ArchInfo *, 17> ArchInfos = {
    &INVALID,  &ARMV8A,   &ARMV8_1A, &ARMV8_2A, &ARMV8_3A, &ARMV8_4A,
    &ARMV8_5A, &ARMV8_6A, &ARMV8_7A, &ARMV8_8A, &ARMV8_9A, &ARMV9A,
    &ARMV9_1A, &ARMV9_2A, &ARMV9_3A, &ARMV9_4A, &ARMV8R,
};

// Details of a specific CPU.
struct CpuInfo {
  StringRef Name; // Name, as written for -mcpu.
  const ArchInfo &Arch;
  uint64_t DefaultExtensions; // Default extensions for this CPU. These will be
                              // ORd with the architecture defaults.
};

inline constexpr CpuInfo CpuInfos[] = {
    {"cortex-a34", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"cortex-a35", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"cortex-a53", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"cortex-a55", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_DOTPROD |
      AArch64::AEK_RCPC)},
    {"cortex-a510", ARMV9A,
     (AArch64::AEK_BF16 | AArch64::AEK_I8MM | AArch64::AEK_SB |
      AArch64::AEK_PAUTH | AArch64::AEK_MTE | AArch64::AEK_SSBS |
      AArch64::AEK_SVE | AArch64::AEK_SVE2 | AArch64::AEK_SVE2BITPERM |
      AArch64::AEK_FP16FML)},
    {"cortex-a57", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"cortex-a65", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_DOTPROD | AArch64::AEK_FP16 |
      AArch64::AEK_RCPC | AArch64::AEK_SSBS)},
    {"cortex-a65ae", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_DOTPROD | AArch64::AEK_FP16 |
      AArch64::AEK_RCPC | AArch64::AEK_SSBS)},
    {"cortex-a72", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"cortex-a73", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"cortex-a75", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_DOTPROD |
      AArch64::AEK_RCPC)},
    {"cortex-a76", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_DOTPROD |
      AArch64::AEK_RCPC | AArch64::AEK_SSBS)},
    {"cortex-a76ae", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_DOTPROD |
      AArch64::AEK_RCPC | AArch64::AEK_SSBS)},
    {"cortex-a77", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_RCPC |
      AArch64::AEK_DOTPROD | AArch64::AEK_SSBS)},
    {"cortex-a78", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_DOTPROD |
      AArch64::AEK_RCPC | AArch64::AEK_SSBS | AArch64::AEK_PROFILE)},
    {"cortex-a78c", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_DOTPROD |
      AArch64::AEK_RCPC | AArch64::AEK_SSBS | AArch64::AEK_PROFILE |
      AArch64::AEK_FLAGM | AArch64::AEK_PAUTH | AArch64::AEK_FP16FML)},
    {"cortex-a710", ARMV9A,
     (AArch64::AEK_MTE | AArch64::AEK_PAUTH | AArch64::AEK_FLAGM |
      AArch64::AEK_SB | AArch64::AEK_I8MM | AArch64::AEK_FP16FML |
      AArch64::AEK_SVE | AArch64::AEK_SVE2 | AArch64::AEK_SVE2BITPERM |
      AArch64::AEK_BF16)},
    {"cortex-a715", ARMV9A,
     (AArch64::AEK_SB | AArch64::AEK_SSBS | AArch64::AEK_MTE |
      AArch64::AEK_FP16 | AArch64::AEK_FP16FML | AArch64::AEK_PAUTH |
      AArch64::AEK_I8MM | AArch64::AEK_PREDRES | AArch64::AEK_PERFMON |
      AArch64::AEK_PROFILE | AArch64::AEK_SVE | AArch64::AEK_SVE2BITPERM |
      AArch64::AEK_BF16 | AArch64::AEK_FLAGM)},
    {"cortex-r82", ARMV8R, (AArch64::AEK_LSE)},
    {"cortex-x1", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_DOTPROD |
      AArch64::AEK_RCPC | AArch64::AEK_SSBS | AArch64::AEK_PROFILE)},
    {"cortex-x1c", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_DOTPROD |
      AArch64::AEK_RCPC | AArch64::AEK_SSBS | AArch64::AEK_PAUTH |
      AArch64::AEK_PROFILE)},
    {"cortex-x2", ARMV9A,
     (AArch64::AEK_MTE | AArch64::AEK_BF16 | AArch64::AEK_I8MM |
      AArch64::AEK_PAUTH | AArch64::AEK_SSBS | AArch64::AEK_SB |
      AArch64::AEK_SVE | AArch64::AEK_SVE2 | AArch64::AEK_SVE2BITPERM |
      AArch64::AEK_FP16FML)},
    {"cortex-x3", ARMV9A,
     (AArch64::AEK_SVE | AArch64::AEK_PERFMON | AArch64::AEK_PROFILE |
      AArch64::AEK_BF16 | AArch64::AEK_I8MM | AArch64::AEK_MTE |
      AArch64::AEK_SVE2BITPERM | AArch64::AEK_SB | AArch64::AEK_PAUTH |
      AArch64::AEK_FP16 | AArch64::AEK_FP16FML | AArch64::AEK_PREDRES |
      AArch64::AEK_FLAGM | AArch64::AEK_SSBS)},
    {"neoverse-e1", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_DOTPROD | AArch64::AEK_FP16 |
      AArch64::AEK_RCPC | AArch64::AEK_SSBS)},
    {"neoverse-n1", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_DOTPROD | AArch64::AEK_FP16 |
      AArch64::AEK_PROFILE | AArch64::AEK_RCPC | AArch64::AEK_SSBS)},
    {"neoverse-n2", ARMV8_5A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_BF16 | AArch64::AEK_DOTPROD |
      AArch64::AEK_FP16 | AArch64::AEK_I8MM | AArch64::AEK_MTE |
      AArch64::AEK_SB | AArch64::AEK_SSBS | AArch64::AEK_SVE |
      AArch64::AEK_SVE2 | AArch64::AEK_SVE2BITPERM)},
    {"neoverse-512tvb", ARMV8_4A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_SVE | AArch64::AEK_SSBS |
      AArch64::AEK_FP16 | AArch64::AEK_BF16 | AArch64::AEK_DOTPROD |
      AArch64::AEK_PROFILE | AArch64::AEK_RAND | AArch64::AEK_FP16FML |
      AArch64::AEK_I8MM)},
    {"neoverse-v1", ARMV8_4A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_SVE | AArch64::AEK_SSBS |
      AArch64::AEK_FP16 | AArch64::AEK_BF16 | AArch64::AEK_DOTPROD |
      AArch64::AEK_PROFILE | AArch64::AEK_RAND | AArch64::AEK_FP16FML |
      AArch64::AEK_I8MM)},
    {"neoverse-v2", ARMV9A,
     (AArch64::AEK_SVE | AArch64::AEK_SVE2 | AArch64::AEK_SSBS |
      AArch64::AEK_FP16 | AArch64::AEK_BF16 | AArch64::AEK_RAND |
      AArch64::AEK_DOTPROD | AArch64::AEK_PROFILE | AArch64::AEK_SVE2BITPERM |
      AArch64::AEK_FP16FML | AArch64::AEK_I8MM | AArch64::AEK_MTE)},
    {"cyclone", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_NONE)},
    {"apple-a7", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_NONE)},
    {"apple-a8", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_NONE)},
    {"apple-a9", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_NONE)},
    {"apple-a10", ARMV8A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_CRC | AArch64::AEK_RDM)},
    {"apple-a11", ARMV8_2A, (AArch64::AEK_CRYPTO | AArch64::AEK_FP16)},
    {"apple-a12", ARMV8_3A, (AArch64::AEK_CRYPTO | AArch64::AEK_FP16)},
    {"apple-a13", ARMV8_4A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_FP16FML |
      AArch64::AEK_SHA3)},
    {"apple-a14", ARMV8_5A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_FP16FML |
      AArch64::AEK_SHA3)},
    {"apple-a15", ARMV8_5A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_FP16FML |
      AArch64::AEK_SHA3 | AArch64::AEK_BF16 | AArch64::AEK_I8MM)},
    {"apple-a16", ARMV8_5A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_FP16FML |
      AArch64::AEK_SHA3 | AArch64::AEK_BF16 | AArch64::AEK_I8MM)},
    {"apple-m1", ARMV8_5A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_FP16FML |
      AArch64::AEK_SHA3)},
    {"apple-m2", ARMV8_5A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_FP16FML |
      AArch64::AEK_SHA3 | AArch64::AEK_BF16 | AArch64::AEK_I8MM)},
    {"apple-s4", ARMV8_3A, (AArch64::AEK_CRYPTO | AArch64::AEK_FP16)},
    {"apple-s5", ARMV8_3A, (AArch64::AEK_CRYPTO | AArch64::AEK_FP16)},
    {"exynos-m3", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"exynos-m4", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_DOTPROD | AArch64::AEK_FP16)},
    {"exynos-m5", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_DOTPROD | AArch64::AEK_FP16)},
    {"falkor", ARMV8A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_CRC | AArch64::AEK_RDM)},
    {"saphira", ARMV8_3A, (AArch64::AEK_CRYPTO | AArch64::AEK_PROFILE)},
    {"kryo", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"thunderx2t99", ARMV8_1A, (AArch64::AEK_CRYPTO)},
    {"thunderx3t110", ARMV8_3A, (AArch64::AEK_CRYPTO)},
    {"thunderx", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"thunderxt88", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"thunderxt81", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"thunderxt83", ARMV8A, (AArch64::AEK_CRYPTO | AArch64::AEK_CRC)},
    {"tsv110", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_DOTPROD | AArch64::AEK_FP16 |
      AArch64::AEK_FP16FML | AArch64::AEK_PROFILE)},
    {"a64fx", ARMV8_2A,
     (AArch64::AEK_CRYPTO | AArch64::AEK_FP16 | AArch64::AEK_SVE)},
    {"carmel", ARMV8_2A, (AArch64::AEK_CRYPTO | AArch64::AEK_FP16)},
    {"ampere1", ARMV8_6A,
     (AArch64::AEK_AES | AArch64::AEK_SHA2 | AArch64::AEK_SHA3 |
      AArch64::AEK_FP16 | AArch64::AEK_SB | AArch64::AEK_SSBS |
      AArch64::AEK_RAND)},
    {"ampere1a", ARMV8_6A,
     (AArch64::AEK_FP16 | AArch64::AEK_RAND | AArch64::AEK_SM4 |
      AArch64::AEK_SHA3 | AArch64::AEK_SHA2 | AArch64::AEK_AES |
      AArch64::AEK_MTE | AArch64::AEK_SB | AArch64::AEK_SSBS)},
    // Invalid CPU
    {"invalid", INVALID, (AArch64::AEK_INVALID)},
};

// An alias for a CPU.
struct CpuAlias {
  StringRef Alias;
  StringRef Name;
};

inline constexpr CpuAlias CpuAliases[] = {{"grace", "neoverse-v2"}};

bool getExtensionFeatures(uint64_t Extensions,
                          std::vector<StringRef> &Features);

StringRef getArchExtFeature(StringRef ArchExt);
StringRef resolveCPUAlias(StringRef CPU);

// Information by Name
uint64_t getDefaultExtensions(StringRef CPU, const ArchInfo &AI);
void getFeatureOption(StringRef Name, std::string &Feature);
const ArchInfo &getArchForCpu(StringRef CPU);

// Parser
const ArchInfo &parseArch(StringRef Arch);
ArchExtKind parseArchExt(StringRef ArchExt);
// Given the name of a CPU or alias, return the correponding CpuInfo.
const CpuInfo &parseCpu(StringRef Name);
// Used by target parser tests
void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values);

bool isX18ReservedByDefault(const Triple &TT);
uint64_t getCpuSupportsMask(ArrayRef<StringRef> FeatureStrs);

} // namespace AArch64
} // namespace llvm

#endif
