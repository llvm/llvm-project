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

#include "llvm/Support/AArch64TargetParser.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/ARMTargetParserCommon.h"
#include <cctype>

using namespace llvm;

static unsigned checkArchVersion(llvm::StringRef Arch) {
  if (Arch.size() >= 2 && Arch[0] == 'v' && std::isdigit(Arch[1]))
    return (Arch[1] - 48);
  return 0;
}

uint64_t AArch64::getDefaultExtensions(StringRef CPU,
                                       const AArch64::ArchInfo &AI) {
  if (CPU == "generic")
    return AI.DefaultExts;

  return StringSwitch<uint64_t>(CPU)
#define AARCH64_CPU_NAME(NAME, ARCH_ID, DEFAULT_EXT)                           \
  .Case(NAME, ARCH_ID.DefaultExts | DEFAULT_EXT)
#include "../../include/llvm/Support/AArch64TargetParser.def"
      .Default(AArch64::AEK_INVALID);
}

const AArch64::ArchInfo &AArch64::getArchForCpu(StringRef CPU) {
  if (CPU == "generic")
    return ARMV8A;

  return *StringSwitch<const AArch64::ArchInfo *>(CPU)
#define AARCH64_CPU_NAME(NAME, ARCH_ID, DEFAULT_EXT) .Case(NAME, &ARCH_ID)
#include "../../include/llvm/Support/AArch64TargetParser.def"
              .Default(&INVALID);
}

const AArch64::ArchInfo &AArch64::ArchInfo::findBySubArch(StringRef SubArch) {
  for (const auto *A : AArch64::ArchInfos)
    if (A->getSubArch() == SubArch)
      return *A;
  return AArch64::INVALID;
}

bool AArch64::getExtensionFeatures(uint64_t Extensions,
                                   std::vector<StringRef> &Features) {
  if (Extensions == AArch64::AEK_INVALID)
    return false;

#define AARCH64_ARCH_EXT_NAME(NAME, ID, FEATURE, NEGFEATURE)                   \
  if (Extensions & ID) {                                                       \
    const char *feature = FEATURE;                                             \
    /* INVALID and NONE have no feature name. */                               \
    if (feature)                                                               \
      Features.push_back(feature);                                             \
  }
#include "../../include/llvm/Support/AArch64TargetParser.def"

  return true;
}

StringRef AArch64::resolveCPUAlias(StringRef CPU) {
  return StringSwitch<StringRef>(CPU)
#define AARCH64_CPU_ALIAS(ALIAS, NAME) .Case(ALIAS, NAME)
#include "../../include/llvm/Support/AArch64TargetParser.def"
      .Default(CPU);
}

StringRef AArch64::getArchExtFeature(StringRef ArchExt) {
  if (ArchExt.startswith("no")) {
    StringRef ArchExtBase(ArchExt.substr(2));
    for (const auto &AE : Extensions) {
      if (!AE.NegFeature.empty() && ArchExtBase == AE.Name)
        return AE.NegFeature;
    }
  }

  for (const auto &AE : Extensions)
    if (!AE.Feature.empty() && ArchExt == AE.Name)
      return AE.Feature;
  return StringRef();
}

void AArch64::fillValidCPUArchList(SmallVectorImpl<StringRef> &Values) {
  for (const auto &C : CpuInfos)
    if (C.Arch != INVALID)
      Values.push_back(C.Name);

  for (const auto &Alias : CpuAliases)
    Values.push_back(Alias.Alias);
}

bool AArch64::isX18ReservedByDefault(const Triple &TT) {
  return TT.isAndroid() || TT.isOSDarwin() || TT.isOSFuchsia() ||
         TT.isOSWindows();
}

// Allows partial match, ex. "v8a" matches "armv8a".
const AArch64::ArchInfo &AArch64::parseArch(StringRef Arch) {
  Arch = llvm::ARM::getCanonicalArchName(Arch);
  if (checkArchVersion(Arch) < 8)
    return AArch64::INVALID;

  StringRef Syn = llvm::ARM::getArchSynonym(Arch);
  for (const auto *A : ArchInfos) {
    if (A->Name.endswith(Syn))
      return *A;
  }
  return AArch64::INVALID;
}

AArch64::ArchExtKind AArch64::parseArchExt(StringRef ArchExt) {
  for (const auto &A : Extensions) {
    if (ArchExt == A.Name)
      return static_cast<ArchExtKind>(A.ID);
  }
  return AArch64::AEK_INVALID;
}

const AArch64::CpuInfo &AArch64::parseCpu(StringRef Name) {
  // Resolve aliases first.
  Name = resolveCPUAlias(Name);

  // Then find the CPU name.
  for (const auto &C : CpuInfos)
    if (Name == C.Name)
      return C;

  // "generic" returns invalid.
  assert(Name != "invalid" && "Unexpected recursion.");
  return parseCpu("invalid");
}
