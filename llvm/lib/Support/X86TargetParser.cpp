//===-- X86TargetParser - Parser for X86 features ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise X86 hardware features.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/X86TargetParser.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"

using namespace llvm;
using namespace llvm::X86;

namespace {

/// Container class for CPU features.
/// This is a constexpr reimplementation of a subset of std::bitset. It would be
/// nice to use std::bitset directly, but it doesn't support constant
/// initialization.
class FeatureBitset {
  static constexpr unsigned NUM_FEATURE_WORDS =
      (X86::CPU_FEATURE_MAX + 31) / 32;

  // This cannot be a std::array, operator[] is not constexpr until C++17.
  uint32_t Bits[NUM_FEATURE_WORDS] = {};

public:
  constexpr FeatureBitset() = default;
  constexpr FeatureBitset(std::initializer_list<unsigned> Init) {
    for (auto I : Init)
      set(I);
  }

  constexpr FeatureBitset &set(unsigned I) {
    // GCC <6.2 crashes if this is written in a single statement.
    uint32_t NewBits = Bits[I / 32] | (uint32_t(1) << (I % 32));
    Bits[I / 32] = NewBits;
    return *this;
  }

  constexpr bool operator[](unsigned I) const {
    uint32_t Mask = uint32_t(1) << (I % 32);
    return (Bits[I / 32] & Mask) != 0;
  }

  constexpr FeatureBitset &operator&=(const FeatureBitset &RHS) {
    for (unsigned I = 0, E = array_lengthof(Bits); I != E; ++I) {
      // GCC <6.2 crashes if this is written in a single statement.
      uint32_t NewBits = Bits[I] & RHS.Bits[I];
      Bits[I] = NewBits;
    }
    return *this;
  }

  constexpr FeatureBitset &operator|=(const FeatureBitset &RHS) {
    for (unsigned I = 0, E = array_lengthof(Bits); I != E; ++I) {
      // GCC <6.2 crashes if this is written in a single statement.
      uint32_t NewBits = Bits[I] | RHS.Bits[I];
      Bits[I] = NewBits;
    }
    return *this;
  }

  // gcc 5.3 miscompiles this if we try to write this using operator&=.
  constexpr FeatureBitset operator&(const FeatureBitset &RHS) const {
    FeatureBitset Result;
    for (unsigned I = 0, E = array_lengthof(Bits); I != E; ++I)
      Result.Bits[I] = Bits[I] & RHS.Bits[I];
    return Result;
  }

  // gcc 5.3 miscompiles this if we try to write this using operator&=.
  constexpr FeatureBitset operator|(const FeatureBitset &RHS) const {
    FeatureBitset Result;
    for (unsigned I = 0, E = array_lengthof(Bits); I != E; ++I)
      Result.Bits[I] = Bits[I] | RHS.Bits[I];
    return Result;
  }

  constexpr FeatureBitset operator~() const {
    FeatureBitset Result;
    for (unsigned I = 0, E = array_lengthof(Bits); I != E; ++I)
      Result.Bits[I] = ~Bits[I];
    return Result;
  }
};

struct ProcInfo {
  StringLiteral Name;
  X86::CPUKind Kind;
  unsigned KeyFeature;
  FeatureBitset Features;
};

struct FeatureInfo {
  StringLiteral Name;
  FeatureBitset ImpliedFeatures;
};

} // end anonymous namespace

#define X86_FEATURE(ENUM, STRING)                                              \
  static constexpr FeatureBitset Feature##ENUM = {X86::FEATURE_##ENUM};
#include "llvm/Support/X86TargetParser.def"

// Pentium with MMX.
static constexpr FeatureBitset FeaturesPentiumMMX =
    FeatureX87 | FeatureCMPXCHG8B | FeatureMMX;

// Pentium 2 and 3.
static constexpr FeatureBitset FeaturesPentium2 =
    FeatureX87 | FeatureCMPXCHG8B | FeatureMMX | FeatureFXSR;
static constexpr FeatureBitset FeaturesPentium3 = FeaturesPentium2 | FeatureSSE;

// Pentium 4 CPUs
static constexpr FeatureBitset FeaturesPentium4 =
    FeaturesPentium3 | FeatureSSE2;
static constexpr FeatureBitset FeaturesPrescott =
    FeaturesPentium4 | FeatureSSE3;
static constexpr FeatureBitset FeaturesNocona =
    FeaturesPrescott | Feature64BIT | FeatureCMPXCHG16B;

// Basic 64-bit capable CPU.
static constexpr FeatureBitset FeaturesX86_64 = FeaturesPentium4 | Feature64BIT;

// Intel Core CPUs
static constexpr FeatureBitset FeaturesCore2 =
    FeaturesNocona | FeatureSAHF | FeatureSSSE3;
static constexpr FeatureBitset FeaturesPenryn = FeaturesCore2 | FeatureSSE4_1;
static constexpr FeatureBitset FeaturesNehalem =
    FeaturesPenryn | FeaturePOPCNT | FeatureSSE4_2;
static constexpr FeatureBitset FeaturesWestmere =
    FeaturesNehalem | FeaturePCLMUL;
static constexpr FeatureBitset FeaturesSandyBridge =
    FeaturesWestmere | FeatureAVX | FeatureXSAVE | FeatureXSAVEOPT;
static constexpr FeatureBitset FeaturesIvyBridge =
    FeaturesSandyBridge | FeatureF16C | FeatureFSGSBASE | FeatureRDRND;
static constexpr FeatureBitset FeaturesHaswell =
    FeaturesIvyBridge | FeatureAVX2 | FeatureBMI | FeatureBMI2 | FeatureFMA |
    FeatureINVPCID | FeatureLZCNT | FeatureMOVBE;
static constexpr FeatureBitset FeaturesBroadwell =
    FeaturesHaswell | FeatureADX | FeaturePRFCHW | FeatureRDSEED;

// Intel Knights Landing and Knights Mill
// Knights Landing has feature parity with Broadwell.
static constexpr FeatureBitset FeaturesKNL =
    FeaturesBroadwell | FeatureAES | FeatureAVX512F | FeatureAVX512CD |
    FeatureAVX512ER | FeatureAVX512PF | FeaturePREFETCHWT1;
static constexpr FeatureBitset FeaturesKNM =
    FeaturesKNL | FeatureAVX512VPOPCNTDQ;

// Intel Skylake processors.
static constexpr FeatureBitset FeaturesSkylakeClient =
    FeaturesBroadwell | FeatureAES | FeatureCLFLUSHOPT | FeatureXSAVEC |
    FeatureXSAVES | FeatureSGX;
// SkylakeServer inherits all SkylakeClient features except SGX.
// FIXME: That doesn't match gcc.
static constexpr FeatureBitset FeaturesSkylakeServer =
    (FeaturesSkylakeClient & ~FeatureSGX) | FeatureAVX512F | FeatureAVX512CD |
    FeatureAVX512DQ | FeatureAVX512BW | FeatureAVX512VL | FeatureCLWB |
    FeaturePKU;
static constexpr FeatureBitset FeaturesCascadeLake =
    FeaturesSkylakeServer | FeatureAVX512VNNI;
static constexpr FeatureBitset FeaturesCooperLake =
    FeaturesCascadeLake | FeatureAVX512BF16;

// Intel 10nm processors.
static constexpr FeatureBitset FeaturesCannonlake =
    FeaturesSkylakeClient | FeatureAVX512F | FeatureAVX512CD | FeatureAVX512DQ |
    FeatureAVX512BW | FeatureAVX512VL | FeatureAVX512IFMA | FeatureAVX512VBMI |
    FeaturePKU | FeatureSHA;
static constexpr FeatureBitset FeaturesICLClient =
    FeaturesCannonlake | FeatureAVX512BITALG | FeatureAVX512VBMI2 |
    FeatureAVX512VNNI | FeatureAVX512VPOPCNTDQ | FeatureCLWB | FeatureGFNI |
    FeatureRDPID | FeatureVAES | FeatureVPCLMULQDQ;
static constexpr FeatureBitset FeaturesICLServer =
    FeaturesICLClient | FeaturePCONFIG | FeatureWBNOINVD;
static constexpr FeatureBitset FeaturesTigerlake =
    FeaturesICLClient | FeatureAVX512VP2INTERSECT | FeatureMOVDIR64B |
    FeatureMOVDIRI | FeatureSHSTK;

// Intel Atom processors.
// Bonnell has feature parity with Core2 and adds MOVBE.
static constexpr FeatureBitset FeaturesBonnell = FeaturesCore2 | FeatureMOVBE;
// Silvermont has parity with Westmere and Bonnell plus PRFCHW and RDRND.
static constexpr FeatureBitset FeaturesSilvermont =
    FeaturesBonnell | FeaturesWestmere | FeaturePRFCHW | FeatureRDRND;
static constexpr FeatureBitset FeaturesGoldmont =
    FeaturesSilvermont | FeatureAES | FeatureCLFLUSHOPT | FeatureFSGSBASE |
    FeatureRDSEED | FeatureSHA | FeatureXSAVE | FeatureXSAVEC |
    FeatureXSAVEOPT | FeatureXSAVES;
static constexpr FeatureBitset FeaturesGoldmontPlus =
    FeaturesGoldmont | FeaturePTWRITE | FeatureRDPID | FeatureSGX;
static constexpr FeatureBitset FeaturesTremont =
    FeaturesGoldmontPlus | FeatureCLWB | FeatureGFNI;

// Geode Processor.
static constexpr FeatureBitset FeaturesGeode =
    FeatureX87 | FeatureCMPXCHG8B | FeatureMMX | Feature3DNOW | Feature3DNOWA;

// K6 processor.
static constexpr FeatureBitset FeaturesK6 =
    FeatureX87 | FeatureCMPXCHG8B | FeatureMMX;

// K7 and K8 architecture processors.
static constexpr FeatureBitset FeaturesAthlon =
    FeatureX87 | FeatureCMPXCHG8B | FeatureMMX | Feature3DNOW | Feature3DNOWA;
static constexpr FeatureBitset FeaturesAthlonXP =
    FeaturesAthlon | FeatureFXSR | FeatureSSE;
static constexpr FeatureBitset FeaturesK8 =
    FeaturesAthlonXP | FeatureSSE2 | Feature64BIT;
static constexpr FeatureBitset FeaturesK8SSE3 = FeaturesK8 | FeatureSSE3;
static constexpr FeatureBitset FeaturesAMDFAM10 =
    FeaturesK8SSE3 | FeatureCMPXCHG16B | FeatureLZCNT | FeaturePOPCNT |
    FeaturePRFCHW | FeatureSAHF | FeatureSSE4_A;

// Bobcat architecture processors.
static constexpr FeatureBitset FeaturesBTVER1 =
    FeatureX87 | FeatureCMPXCHG8B | FeatureCMPXCHG16B | Feature64BIT |
    FeatureFXSR | FeatureLZCNT | FeatureMMX | FeaturePOPCNT | FeaturePRFCHW |
    FeatureSSE | FeatureSSE2 | FeatureSSE3 | FeatureSSSE3 | FeatureSSE4_A |
    FeatureSAHF;
static constexpr FeatureBitset FeaturesBTVER2 =
    FeaturesBTVER1 | FeatureAES | FeatureAVX | FeatureBMI | FeatureF16C |
    FeatureMOVBE | FeaturePCLMUL | FeatureXSAVE | FeatureXSAVEOPT;

// AMD Bulldozer architecture processors.
static constexpr FeatureBitset FeaturesBDVER1 =
    FeatureX87 | FeatureAES | FeatureAVX | FeatureCMPXCHG8B |
    FeatureCMPXCHG16B | Feature64BIT | FeatureFMA4 | FeatureFXSR | FeatureLWP |
    FeatureLZCNT | FeatureMMX | FeaturePCLMUL | FeaturePOPCNT | FeaturePRFCHW |
    FeatureSAHF | FeatureSSE | FeatureSSE2 | FeatureSSE3 | FeatureSSSE3 |
    FeatureSSE4_1 | FeatureSSE4_2 | FeatureSSE4_A | FeatureXOP | FeatureXSAVE;
static constexpr FeatureBitset FeaturesBDVER2 =
    FeaturesBDVER1 | FeatureBMI | FeatureFMA | FeatureF16C | FeatureTBM;
static constexpr FeatureBitset FeaturesBDVER3 =
    FeaturesBDVER2 | FeatureFSGSBASE | FeatureXSAVEOPT;
static constexpr FeatureBitset FeaturesBDVER4 =
    FeaturesBDVER3 | FeatureAVX2 | FeatureBMI2 | FeatureMOVBE | FeatureMWAITX |
    FeatureRDRND;

// AMD Zen architecture processors.
static constexpr FeatureBitset FeaturesZNVER1 =
    FeatureX87 | FeatureADX | FeatureAES | FeatureAVX | FeatureAVX2 |
    FeatureBMI | FeatureBMI2 | FeatureCLFLUSHOPT | FeatureCLZERO |
    FeatureCMPXCHG8B | FeatureCMPXCHG16B | Feature64BIT | FeatureF16C |
    FeatureFMA | FeatureFSGSBASE | FeatureFXSR | FeatureLZCNT | FeatureMMX |
    FeatureMOVBE | FeatureMWAITX | FeaturePCLMUL | FeaturePOPCNT |
    FeaturePRFCHW | FeatureRDRND | FeatureRDSEED | FeatureSAHF | FeatureSHA |
    FeatureSSE | FeatureSSE2 | FeatureSSE3 | FeatureSSSE3 | FeatureSSE4_1 |
    FeatureSSE4_2 | FeatureSSE4_A | FeatureXSAVE | FeatureXSAVEC |
    FeatureXSAVEOPT | FeatureXSAVES;
static constexpr FeatureBitset FeaturesZNVER2 =
    FeaturesZNVER1 | FeatureCLWB | FeatureRDPID | FeatureWBNOINVD;

static constexpr ProcInfo Processors[] = {
  // Empty processor. Include X87 and CMPXCHG8 for backwards compatibility.
  { {""}, CK_None, ~0U, FeatureX87 | FeatureCMPXCHG8B },
  // i386-generation processors.
  { {"i386"}, CK_i386, ~0U, FeatureX87 },
  // i486-generation processors.
  { {"i486"}, CK_i486, ~0U, FeatureX87 },
  { {"winchip-c6"}, CK_WinChipC6, ~0U, FeaturesPentiumMMX },
  { {"winchip2"}, CK_WinChip2, ~0U, FeaturesPentiumMMX | Feature3DNOW },
  { {"c3"}, CK_C3, ~0U, FeaturesPentiumMMX | Feature3DNOW },
  // i586-generation processors, P5 microarchitecture based.
  { {"i586"}, CK_i586, ~0U, FeatureX87 | FeatureCMPXCHG8B },
  { {"pentium"}, CK_Pentium, ~0U, FeatureX87 | FeatureCMPXCHG8B },
  { {"pentium-mmx"}, CK_PentiumMMX, ~0U, FeaturesPentiumMMX },
  // i686-generation processors, P6 / Pentium M microarchitecture based.
  { {"pentiumpro"}, CK_PentiumPro, ~0U, FeatureX87 | FeatureCMPXCHG8B },
  { {"i686"}, CK_i686, ~0U, FeatureX87 | FeatureCMPXCHG8B },
  { {"pentium2"}, CK_Pentium2, ~0U, FeaturesPentium2 },
  { {"pentium3"}, CK_Pentium3, ~0U, FeaturesPentium3 },
  { {"pentium3m"}, CK_Pentium3, ~0U, FeaturesPentium3 },
  { {"pentium-m"}, CK_PentiumM, ~0U, FeaturesPentium4 },
  { {"c3-2"}, CK_C3_2, ~0U, FeaturesPentium3 },
  { {"yonah"}, CK_Yonah, ~0U, FeaturesPrescott },
  // Netburst microarchitecture based processors.
  { {"pentium4"}, CK_Pentium4, ~0U, FeaturesPentium4 },
  { {"pentium4m"}, CK_Pentium4, ~0U, FeaturesPentium4 },
  { {"prescott"}, CK_Prescott, ~0U, FeaturesPrescott },
  { {"nocona"}, CK_Nocona, ~0U, FeaturesNocona },
  // Core microarchitecture based processors.
  { {"core2"}, CK_Core2, ~0U, FeaturesCore2 },
  { {"penryn"}, CK_Penryn, ~0U, FeaturesPenryn },
  // Atom processors
  { {"bonnell"}, CK_Bonnell, FEATURE_SSSE3, FeaturesBonnell },
  { {"atom"}, CK_Bonnell, FEATURE_SSSE3, FeaturesBonnell },
  { {"silvermont"}, CK_Silvermont, FEATURE_SSE4_2, FeaturesSilvermont },
  { {"slm"}, CK_Silvermont, FEATURE_SSE4_2, FeaturesSilvermont },
  { {"goldmont"}, CK_Goldmont, FEATURE_SSE4_2, FeaturesGoldmont },
  { {"goldmont-plus"}, CK_GoldmontPlus, FEATURE_SSE4_2, FeaturesGoldmontPlus },
  { {"tremont"}, CK_Tremont, FEATURE_SSE4_2, FeaturesTremont },
  // Nehalem microarchitecture based processors.
  { {"nehalem"}, CK_Nehalem, FEATURE_SSE4_2, FeaturesNehalem },
  { {"corei7"}, CK_Nehalem, FEATURE_SSE4_2, FeaturesNehalem },
  // Westmere microarchitecture based processors.
  { {"westmere"}, CK_Westmere, FEATURE_PCLMUL, FeaturesWestmere },
  // Sandy Bridge microarchitecture based processors.
  { {"sandybridge"}, CK_SandyBridge, FEATURE_AVX, FeaturesSandyBridge },
  { {"corei7-avx"}, CK_SandyBridge, FEATURE_AVX, FeaturesSandyBridge },
  // Ivy Bridge microarchitecture based processors.
  { {"ivybridge"}, CK_IvyBridge, FEATURE_AVX, FeaturesIvyBridge },
  { {"core-avx-i"}, CK_IvyBridge, FEATURE_AVX, FeaturesIvyBridge },
  // Haswell microarchitecture based processors.
  { {"haswell"}, CK_Haswell, FEATURE_AVX2, FeaturesHaswell },
  { {"core-avx2"}, CK_Haswell, FEATURE_AVX2, FeaturesHaswell },
  // Broadwell microarchitecture based processors.
  { {"broadwell"}, CK_Broadwell, FEATURE_AVX2, FeaturesBroadwell },
  // Skylake client microarchitecture based processors.
  { {"skylake"}, CK_SkylakeClient, FEATURE_AVX2, FeaturesSkylakeClient },
  // Skylake server microarchitecture based processors.
  { {"skylake-avx512"}, CK_SkylakeServer, FEATURE_AVX512F, FeaturesSkylakeServer },
  { {"skx"}, CK_SkylakeServer, FEATURE_AVX512F, FeaturesSkylakeServer },
  // Cascadelake Server microarchitecture based processors.
  { {"cascadelake"}, CK_Cascadelake, FEATURE_AVX512VNNI, FeaturesCascadeLake },
  // Cooperlake Server microarchitecture based processors.
  { {"cooperlake"}, CK_Cooperlake, FEATURE_AVX512BF16, FeaturesCooperLake },
  // Cannonlake client microarchitecture based processors.
  { {"cannonlake"}, CK_Cannonlake, FEATURE_AVX512VBMI, FeaturesCannonlake },
  // Icelake client microarchitecture based processors.
  { {"icelake-client"}, CK_IcelakeClient, FEATURE_AVX512VBMI2, FeaturesICLClient },
  // Icelake server microarchitecture based processors.
  { {"icelake-server"}, CK_IcelakeServer, FEATURE_AVX512VBMI2, FeaturesICLServer },
  // Tigerlake microarchitecture based processors.
  { {"tigerlake"}, CK_Tigerlake, FEATURE_AVX512VP2INTERSECT, FeaturesTigerlake },
  // Knights Landing processor.
  { {"knl"}, CK_KNL, FEATURE_AVX512F, FeaturesKNL },
  // Knights Mill processor.
  { {"knm"}, CK_KNM, FEATURE_AVX5124FMAPS, FeaturesKNM },
  // Lakemont microarchitecture based processors.
  { {"lakemont"}, CK_Lakemont, ~0U, FeatureCMPXCHG8B },
  // K6 architecture processors.
  { {"k6"}, CK_K6, ~0U, FeaturesK6 },
  { {"k6-2"}, CK_K6_2, ~0U, FeaturesK6 | Feature3DNOW },
  { {"k6-3"}, CK_K6_3, ~0U, FeaturesK6 | Feature3DNOW },
  // K7 architecture processors.
  { {"athlon"}, CK_Athlon, ~0U, FeaturesAthlon },
  { {"athlon-tbird"}, CK_Athlon, ~0U, FeaturesAthlon },
  { {"athlon-xp"}, CK_AthlonXP, ~0U, FeaturesAthlonXP },
  { {"athlon-mp"}, CK_AthlonXP, ~0U, FeaturesAthlonXP },
  { {"athlon-4"}, CK_AthlonXP, ~0U, FeaturesAthlonXP },
  // K8 architecture processors.
  { {"k8"}, CK_K8, ~0U, FeaturesK8 },
  { {"athlon64"}, CK_K8, ~0U, FeaturesK8 },
  { {"athlon-fx"}, CK_K8, ~0U, FeaturesK8 },
  { {"opteron"}, CK_K8, ~0U, FeaturesK8 },
  { {"k8-sse3"}, CK_K8SSE3, ~0U, FeaturesK8SSE3 },
  { {"athlon64-sse3"}, CK_K8SSE3, ~0U, FeaturesK8SSE3 },
  { {"opteron-sse3"}, CK_K8SSE3, ~0U, FeaturesK8SSE3 },
  { {"amdfam10"}, CK_AMDFAM10, FEATURE_SSE4_A, FeaturesAMDFAM10 },
  { {"barcelona"}, CK_AMDFAM10, FEATURE_SSE4_A, FeaturesAMDFAM10 },
  // Bobcat architecture processors.
  { {"btver1"}, CK_BTVER1, FEATURE_SSE4_A, FeaturesBTVER1 },
  { {"btver2"}, CK_BTVER2, FEATURE_BMI, FeaturesBTVER2 },
  // Bulldozer architecture processors.
  { {"bdver1"}, CK_BDVER1, FEATURE_XOP, FeaturesBDVER1 },
  { {"bdver2"}, CK_BDVER2, FEATURE_FMA, FeaturesBDVER2 },
  { {"bdver3"}, CK_BDVER3, FEATURE_FMA, FeaturesBDVER3 },
  { {"bdver4"}, CK_BDVER4, FEATURE_AVX2, FeaturesBDVER4 },
  // Zen architecture processors.
  { {"znver1"}, CK_ZNVER1, FEATURE_AVX2, FeaturesZNVER1 },
  { {"znver2"}, CK_ZNVER2, FEATURE_AVX2, FeaturesZNVER2 },
  // Generic 64-bit processor.
  { {"x86-64"}, CK_x86_64, ~0U, FeaturesX86_64 },
  // Geode processors.
  { {"geode"}, CK_Geode, ~0U, FeaturesGeode },
};

X86::CPUKind llvm::X86::parseArchX86(StringRef CPU, bool Only64Bit) {
  for (const auto &P : Processors)
    if (P.Name == CPU && (P.Features[FEATURE_64BIT] || !Only64Bit))
      return P.Kind;

  return CK_None;
}

void llvm::X86::fillValidCPUArchList(SmallVectorImpl<StringRef> &Values,
                                     bool Only64Bit) {
  for (const auto &P : Processors)
    if (!P.Name.empty() && (P.Features[FEATURE_64BIT] || !Only64Bit))
      Values.emplace_back(P.Name);
}

ProcessorFeatures llvm::X86::getKeyFeature(X86::CPUKind Kind) {
  // FIXME: Can we avoid a linear search here? The table might be sorted by
  // CPUKind so we could binary search?
  for (const auto &P : Processors) {
    if (P.Kind == Kind) {
      assert(P.KeyFeature != ~0U && "Processor does not have a key feature.");
      return static_cast<ProcessorFeatures>(P.KeyFeature);
    }
  }

  llvm_unreachable("Unable to find CPU kind!");
}

// Features with no dependencies.
static constexpr FeatureBitset ImpliedFeatures64BIT = {};
static constexpr FeatureBitset ImpliedFeaturesADX = {};
static constexpr FeatureBitset ImpliedFeaturesBMI = {};
static constexpr FeatureBitset ImpliedFeaturesBMI2 = {};
static constexpr FeatureBitset ImpliedFeaturesCLDEMOTE = {};
static constexpr FeatureBitset ImpliedFeaturesCLFLUSHOPT = {};
static constexpr FeatureBitset ImpliedFeaturesCLWB = {};
static constexpr FeatureBitset ImpliedFeaturesCLZERO = {};
static constexpr FeatureBitset ImpliedFeaturesCMOV = {};
static constexpr FeatureBitset ImpliedFeaturesCMPXCHG16B = {};
static constexpr FeatureBitset ImpliedFeaturesCMPXCHG8B = {};
static constexpr FeatureBitset ImpliedFeaturesENQCMD = {};
static constexpr FeatureBitset ImpliedFeaturesFSGSBASE = {};
static constexpr FeatureBitset ImpliedFeaturesFXSR = {};
static constexpr FeatureBitset ImpliedFeaturesINVPCID = {};
static constexpr FeatureBitset ImpliedFeaturesLWP = {};
static constexpr FeatureBitset ImpliedFeaturesLZCNT = {};
static constexpr FeatureBitset ImpliedFeaturesMWAITX = {};
static constexpr FeatureBitset ImpliedFeaturesMOVBE = {};
static constexpr FeatureBitset ImpliedFeaturesMOVDIR64B = {};
static constexpr FeatureBitset ImpliedFeaturesMOVDIRI = {};
static constexpr FeatureBitset ImpliedFeaturesPCONFIG = {};
static constexpr FeatureBitset ImpliedFeaturesPOPCNT = {};
static constexpr FeatureBitset ImpliedFeaturesPKU = {};
static constexpr FeatureBitset ImpliedFeaturesPREFETCHWT1 = {};
static constexpr FeatureBitset ImpliedFeaturesPRFCHW = {};
static constexpr FeatureBitset ImpliedFeaturesPTWRITE = {};
static constexpr FeatureBitset ImpliedFeaturesRDPID = {};
static constexpr FeatureBitset ImpliedFeaturesRDRND = {};
static constexpr FeatureBitset ImpliedFeaturesRDSEED = {};
static constexpr FeatureBitset ImpliedFeaturesRTM = {};
static constexpr FeatureBitset ImpliedFeaturesSAHF = {};
static constexpr FeatureBitset ImpliedFeaturesSERIALIZE = {};
static constexpr FeatureBitset ImpliedFeaturesSGX = {};
static constexpr FeatureBitset ImpliedFeaturesSHSTK = {};
static constexpr FeatureBitset ImpliedFeaturesTBM = {};
static constexpr FeatureBitset ImpliedFeaturesTSXLDTRK = {};
static constexpr FeatureBitset ImpliedFeaturesWAITPKG = {};
static constexpr FeatureBitset ImpliedFeaturesWBNOINVD = {};
static constexpr FeatureBitset ImpliedFeaturesVZEROUPPER = {};
static constexpr FeatureBitset ImpliedFeaturesX87 = {};
static constexpr FeatureBitset ImpliedFeaturesXSAVE = {};

// Not really CPU features, but need to be in the table because clang uses
// target features to communicate them to the backend.
static constexpr FeatureBitset ImpliedFeaturesRETPOLINE_EXTERNAL_THUNK = {};
static constexpr FeatureBitset ImpliedFeaturesRETPOLINE_INDIRECT_BRANCHES = {};
static constexpr FeatureBitset ImpliedFeaturesRETPOLINE_INDIRECT_CALLS = {};
static constexpr FeatureBitset ImpliedFeaturesLVI_CFI = {};
static constexpr FeatureBitset ImpliedFeaturesLVI_LOAD_HARDENING = {};

// XSAVE features are dependent on basic XSAVE.
static constexpr FeatureBitset ImpliedFeaturesXSAVEC = FeatureXSAVE;
static constexpr FeatureBitset ImpliedFeaturesXSAVEOPT = FeatureXSAVE;
static constexpr FeatureBitset ImpliedFeaturesXSAVES = FeatureXSAVE;

// MMX->3DNOW->3DNOWA chain.
static constexpr FeatureBitset ImpliedFeaturesMMX = {};
static constexpr FeatureBitset ImpliedFeatures3DNOW = FeatureMMX;
static constexpr FeatureBitset ImpliedFeatures3DNOWA = Feature3DNOW;

// SSE/AVX/AVX512F chain.
static constexpr FeatureBitset ImpliedFeaturesSSE = {};
static constexpr FeatureBitset ImpliedFeaturesSSE2 = FeatureSSE;
static constexpr FeatureBitset ImpliedFeaturesSSE3 = FeatureSSE2;
static constexpr FeatureBitset ImpliedFeaturesSSSE3 = FeatureSSE3;
static constexpr FeatureBitset ImpliedFeaturesSSE4_1 = FeatureSSSE3;
static constexpr FeatureBitset ImpliedFeaturesSSE4_2 = FeatureSSE4_1;
static constexpr FeatureBitset ImpliedFeaturesAVX = FeatureSSE4_2;
static constexpr FeatureBitset ImpliedFeaturesAVX2 = FeatureAVX;
static constexpr FeatureBitset ImpliedFeaturesAVX512F =
    FeatureAVX2 | FeatureF16C | FeatureFMA;

// Vector extensions that build on SSE or AVX.
static constexpr FeatureBitset ImpliedFeaturesAES = FeatureSSE2;
static constexpr FeatureBitset ImpliedFeaturesF16C = FeatureAVX;
static constexpr FeatureBitset ImpliedFeaturesFMA = FeatureAVX;
static constexpr FeatureBitset ImpliedFeaturesGFNI = FeatureSSE2;
static constexpr FeatureBitset ImpliedFeaturesPCLMUL = FeatureSSE2;
static constexpr FeatureBitset ImpliedFeaturesSHA = FeatureSSE2;
static constexpr FeatureBitset ImpliedFeaturesVAES = FeatureAES | FeatureAVX;
static constexpr FeatureBitset ImpliedFeaturesVPCLMULQDQ =
    FeatureAVX | FeaturePCLMUL;

// AVX512 features.
static constexpr FeatureBitset ImpliedFeaturesAVX512CD = FeatureAVX512F;
static constexpr FeatureBitset ImpliedFeaturesAVX512BW = FeatureAVX512F;
static constexpr FeatureBitset ImpliedFeaturesAVX512DQ = FeatureAVX512F;
static constexpr FeatureBitset ImpliedFeaturesAVX512ER = FeatureAVX512F;
static constexpr FeatureBitset ImpliedFeaturesAVX512PF = FeatureAVX512F;
static constexpr FeatureBitset ImpliedFeaturesAVX512VL = FeatureAVX512F;

static constexpr FeatureBitset ImpliedFeaturesAVX512BF16 = FeatureAVX512BW;
static constexpr FeatureBitset ImpliedFeaturesAVX512BITALG = FeatureAVX512BW;
static constexpr FeatureBitset ImpliedFeaturesAVX512IFMA = FeatureAVX512F;
static constexpr FeatureBitset ImpliedFeaturesAVX512VNNI = FeatureAVX512F;
static constexpr FeatureBitset ImpliedFeaturesAVX512VPOPCNTDQ = FeatureAVX512F;
static constexpr FeatureBitset ImpliedFeaturesAVX512VBMI = FeatureAVX512BW;
static constexpr FeatureBitset ImpliedFeaturesAVX512VBMI2 = FeatureAVX512BW;
static constexpr FeatureBitset ImpliedFeaturesAVX512VP2INTERSECT =
    FeatureAVX512F;

// FIXME: These two aren't really implemented and just exist in the feature
// list for __builtin_cpu_supports. So omit their dependencies.
static constexpr FeatureBitset ImpliedFeaturesAVX5124FMAPS = {};
static constexpr FeatureBitset ImpliedFeaturesAVX5124VNNIW = {};

// SSE4_A->FMA4->XOP chain.
static constexpr FeatureBitset ImpliedFeaturesSSE4_A = FeatureSSSE3;
static constexpr FeatureBitset ImpliedFeaturesFMA4 = FeatureAVX | FeatureSSE4_A;
static constexpr FeatureBitset ImpliedFeaturesXOP = FeatureFMA4;

// AMX Features
static constexpr FeatureBitset ImpliedFeaturesAMX_TILE = {};
static constexpr FeatureBitset ImpliedFeaturesAMX_BF16 = FeatureAMX_TILE;
static constexpr FeatureBitset ImpliedFeaturesAMX_INT8 = FeatureAMX_TILE;

static constexpr FeatureInfo FeatureInfos[X86::CPU_FEATURE_MAX] = {
#define X86_FEATURE(ENUM, STR) {{STR}, ImpliedFeatures##ENUM},
#include "llvm/Support/X86TargetParser.def"
};

// Convert the set bits in FeatureBitset to a list of strings.
static void getFeatureBitsAsStrings(const FeatureBitset &Bits,
                                    SmallVectorImpl<StringRef> &Features) {
  for (unsigned i = 0; i != CPU_FEATURE_MAX; ++i)
    if (Bits[i] && !FeatureInfos[i].Name.empty())
      Features.push_back(FeatureInfos[i].Name);
}

void llvm::X86::getFeaturesForCPU(StringRef CPU,
                                  SmallVectorImpl<StringRef> &EnabledFeatures) {
  auto I = llvm::find_if(Processors,
                         [&](const ProcInfo &P) { return P.Name == CPU; });
  assert(I != std::end(Processors) && "Processor not found!");

  FeatureBitset Bits = I->Features;

  // Remove the 64-bit feature which we only use to validate if a CPU can
  // be used with 64-bit mode.
  Bits &= ~Feature64BIT;

  // Add the string version of all set bits.
  getFeatureBitsAsStrings(Bits, EnabledFeatures);
}

// For each feature that is (transitively) implied by this feature, set it.
static void getImpliedEnabledFeatures(FeatureBitset &Bits,
                                      const FeatureBitset &Implies) {
  Bits |= Implies;
  for (unsigned i = 0; i != CPU_FEATURE_MAX; ++i) {
    if (Implies[i])
      getImpliedEnabledFeatures(Bits, FeatureInfos[i].ImpliedFeatures);
  }
}

/// Create bit vector of features that are implied disabled if the feature
/// passed in Value is disabled.
static void getImpliedDisabledFeatures(FeatureBitset &Bits, unsigned Value) {
  // Check all features looking for any dependent on this feature. If we find
  // one, mark it and recursively find any feature that depend on it.
  for (unsigned i = 0; i != CPU_FEATURE_MAX; ++i) {
    if (FeatureInfos[i].ImpliedFeatures[Value]) {
      Bits.set(i);
      getImpliedDisabledFeatures(Bits, i);
    }
  }
}

void llvm::X86::getImpliedFeatures(
    StringRef Feature, bool Enabled,
    SmallVectorImpl<StringRef> &ImpliedFeatures) {
  auto I = llvm::find_if(
      FeatureInfos, [&](const FeatureInfo &FI) { return FI.Name == Feature; });
  if (I == std::end(FeatureInfos)) {
    // FIXME: This shouldn't happen, but may not have all features in the table
    // yet.
    return;
  }

  FeatureBitset ImpliedBits;
  if (Enabled)
    getImpliedEnabledFeatures(ImpliedBits, I->ImpliedFeatures);
  else
    getImpliedDisabledFeatures(ImpliedBits,
                               std::distance(std::begin(FeatureInfos), I));

  // Convert all the found bits into strings.
  getFeatureBitsAsStrings(ImpliedBits, ImpliedFeatures);
}
