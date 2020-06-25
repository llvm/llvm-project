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

struct ProcInfo {
  StringLiteral Name;
  X86::CPUKind Kind;
  unsigned KeyFeature;
  bool Is64Bit;
};

} // end anonymous namespace

#define PROC_64_BIT true
#define PROC_32_BIT false

static constexpr ProcInfo Processors[] = {
  // i386-generation processors.
  { {"i386"}, CK_i386, ~0U, PROC_32_BIT },
  // i486-generation processors.
  { {"i486"}, CK_i486, ~0U, PROC_32_BIT },
  { {"winchip-c6"}, CK_WinChipC6, ~0U, PROC_32_BIT },
  { {"winchip2"}, CK_WinChip2, ~0U, PROC_32_BIT },
  { {"c3"}, CK_C3, ~0U, PROC_32_BIT },
  // i586-generation processors, P5 microarchitecture based.
  { {"i586"}, CK_i586, ~0U, PROC_32_BIT },
  { {"pentium"}, CK_Pentium, ~0U, PROC_32_BIT },
  { {"pentium-mmx"}, CK_PentiumMMX, ~0U, PROC_32_BIT },
  { {"pentiumpro"}, CK_PentiumPro, ~0U, PROC_32_BIT },
  // i686-generation processors, P6 / Pentium M microarchitecture based.
  { {"i686"}, CK_i686, ~0U, PROC_32_BIT },
  { {"pentium2"}, CK_Pentium2, ~0U, PROC_32_BIT },
  { {"pentium3"}, CK_Pentium3, ~0U, PROC_32_BIT },
  { {"pentium3m"}, CK_Pentium3, ~0U, PROC_32_BIT },
  { {"pentium-m"}, CK_PentiumM, ~0U, PROC_32_BIT },
  { {"c3-2"}, CK_C3_2, ~0U, PROC_32_BIT },
  { {"yonah"}, CK_Yonah, ~0U, PROC_32_BIT },
  // Netburst microarchitecture based processors.
  { {"pentium4"}, CK_Pentium4, ~0U, PROC_32_BIT },
  { {"pentium4m"}, CK_Pentium4, ~0U, PROC_32_BIT },
  { {"prescott"}, CK_Prescott, ~0U, PROC_32_BIT },
  { {"nocona"}, CK_Nocona, ~0U, PROC_64_BIT },
  // Core microarchitecture based processors.
  { {"core2"}, CK_Core2, ~0U, PROC_64_BIT },
  { {"penryn"}, CK_Penryn, ~0U, PROC_64_BIT },
  // Atom processors
  { {"bonnell"}, CK_Bonnell, FEATURE_SSSE3, PROC_64_BIT },
  { {"atom"}, CK_Bonnell, FEATURE_SSSE3, PROC_64_BIT },
  { {"silvermont"}, CK_Silvermont, FEATURE_SSE4_2, PROC_64_BIT },
  { {"slm"}, CK_Silvermont, FEATURE_SSE4_2, PROC_64_BIT },
  { {"goldmont"}, CK_Goldmont, FEATURE_SSE4_2, PROC_64_BIT },
  { {"goldmont-plus"}, CK_GoldmontPlus, FEATURE_SSE4_2, PROC_64_BIT },
  { {"tremont"}, CK_Tremont, FEATURE_SSE4_2, PROC_64_BIT },
  // Nehalem microarchitecture based processors.
  { {"nehalem"}, CK_Nehalem, FEATURE_SSE4_2, PROC_64_BIT },
  { {"corei7"}, CK_Nehalem, FEATURE_SSE4_2, PROC_64_BIT },
  // Westmere microarchitecture based processors.
  { {"westmere"}, CK_Westmere, FEATURE_PCLMUL, PROC_64_BIT },
  // Sandy Bridge microarchitecture based processors.
  { {"sandybridge"}, CK_SandyBridge, FEATURE_AVX, PROC_64_BIT },
  { {"corei7-avx"}, CK_SandyBridge, FEATURE_AVX, PROC_64_BIT },
  // Ivy Bridge microarchitecture based processors.
  { {"ivybridge"}, CK_IvyBridge, FEATURE_AVX, PROC_64_BIT },
  { {"core-avx-i"}, CK_IvyBridge, FEATURE_AVX, PROC_64_BIT },
  // Haswell microarchitecture based processors.
  { {"haswell"}, CK_Haswell, FEATURE_AVX2, PROC_64_BIT },
  { {"core-avx2"}, CK_Haswell, FEATURE_AVX2, PROC_64_BIT },
  // Broadwell microarchitecture based processors.
  { {"broadwell"}, CK_Broadwell, FEATURE_AVX2, PROC_64_BIT },
  // Skylake client microarchitecture based processors.
  { {"skylake"}, CK_SkylakeClient, FEATURE_AVX2, PROC_64_BIT },
  // Skylake server microarchitecture based processors.
  { {"skylake-avx512"}, CK_SkylakeServer, FEATURE_AVX512F, PROC_64_BIT },
  { {"skx"}, CK_SkylakeServer, FEATURE_AVX512F, PROC_64_BIT },
  // Cascadelake Server microarchitecture based processors.
  { {"cascadelake"}, CK_Cascadelake, FEATURE_AVX512VNNI, PROC_64_BIT },
  // Cooperlake Server microarchitecture based processors.
  { {"cooperlake"}, CK_Cooperlake, FEATURE_AVX512BF16, PROC_64_BIT },
  // Cannonlake client microarchitecture based processors.
  { {"cannonlake"}, CK_Cannonlake, FEATURE_AVX512VBMI, PROC_64_BIT },
  // Icelake client microarchitecture based processors.
  { {"icelake-client"}, CK_IcelakeClient, FEATURE_AVX512VBMI2, PROC_64_BIT },
  // Icelake server microarchitecture based processors.
  { {"icelake-server"}, CK_IcelakeServer, FEATURE_AVX512VBMI2, PROC_64_BIT },
  // Tigerlake microarchitecture based processors.
  { {"tigerlake"}, CK_Tigerlake, FEATURE_AVX512VP2INTERSECT, PROC_64_BIT },
  // Knights Landing processor.
  { {"knl"}, CK_KNL, FEATURE_AVX512F, PROC_64_BIT },
  // Knights Mill processor.
  { {"knm"}, CK_KNM, FEATURE_AVX5124FMAPS, PROC_64_BIT },
  // Lakemont microarchitecture based processors.
  { {"lakemont"}, CK_Lakemont, ~0U, PROC_32_BIT },
  // K6 architecture processors.
  { {"k6"}, CK_K6, ~0U, PROC_32_BIT },
  { {"k6-2"}, CK_K6_2, ~0U, PROC_32_BIT },
  { {"k6-3"}, CK_K6_3, ~0U, PROC_32_BIT },
  // K7 architecture processors.
  { {"athlon"}, CK_Athlon, ~0U, PROC_32_BIT },
  { {"athlon-tbird"}, CK_Athlon, ~0U, PROC_32_BIT },
  { {"athlon-xp"}, CK_AthlonXP, ~0U, PROC_32_BIT },
  { {"athlon-mp"}, CK_AthlonXP, ~0U, PROC_32_BIT },
  { {"athlon-4"}, CK_AthlonXP, ~0U, PROC_32_BIT },
  // K8 architecture processors.
  { {"k8"}, CK_K8, ~0U, PROC_64_BIT },
  { {"athlon64"}, CK_K8, ~0U, PROC_64_BIT },
  { {"athlon-fx"}, CK_K8, ~0U, PROC_64_BIT },
  { {"opteron"}, CK_K8, ~0U, PROC_64_BIT },
  { {"k8-sse3"}, CK_K8SSE3, ~0U, PROC_64_BIT },
  { {"athlon64-sse3"}, CK_K8SSE3, ~0U, PROC_64_BIT },
  { {"opteron-sse3"}, CK_K8SSE3, ~0U, PROC_64_BIT },
  { {"amdfam10"}, CK_AMDFAM10, FEATURE_SSE4_A, PROC_64_BIT },
  { {"barcelona"}, CK_AMDFAM10, FEATURE_SSE4_A, PROC_64_BIT },
  // Bobcat architecture processors.
  { {"btver1"}, CK_BTVER1, FEATURE_SSE4_A, PROC_64_BIT },
  { {"btver2"}, CK_BTVER2, FEATURE_BMI, PROC_64_BIT },
  // Bulldozer architecture processors.
  { {"bdver1"}, CK_BDVER1, FEATURE_XOP, PROC_64_BIT },
  { {"bdver2"}, CK_BDVER2, FEATURE_FMA, PROC_64_BIT },
  { {"bdver3"}, CK_BDVER3, FEATURE_FMA, PROC_64_BIT },
  { {"bdver4"}, CK_BDVER4, FEATURE_AVX2, PROC_64_BIT },
  // Zen architecture processors.
  { {"znver1"}, CK_ZNVER1, FEATURE_AVX2, PROC_64_BIT },
  { {"znver2"}, CK_ZNVER2, FEATURE_AVX2, PROC_64_BIT },
  // Generic 64-bit processor.
  { {"x86-64"}, CK_x86_64, ~0U, PROC_64_BIT },
  // Geode processors.
  { {"geode"}, CK_Geode, ~0U, PROC_32_BIT },
};

X86::CPUKind llvm::X86::parseArchX86(StringRef CPU, bool Only64Bit) {
  for (const auto &P : Processors)
    if (P.Name == CPU && (P.Is64Bit || !Only64Bit))
      return P.Kind;

  return CK_None;
}

void llvm::X86::fillValidCPUArchList(SmallVectorImpl<StringRef> &Values,
                                     bool Only64Bit) {
  for (const auto &P : Processors)
    if (P.Is64Bit || !Only64Bit)
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
