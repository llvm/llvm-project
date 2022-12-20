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

#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TargetParser/ARMTargetParserCommon.h"
#include "llvm/TargetParser/Triple.h"
#include <cctype>

using namespace llvm;

static unsigned checkArchVersion(llvm::StringRef Arch) {
  if (Arch.size() >= 2 && Arch[0] == 'v' && std::isdigit(Arch[1]))
    return (Arch[1] - 48);
  return 0;
}

uint64_t AArch64::getDefaultExtensions(StringRef CPU, AArch64::ArchKind AK) {
  if (CPU == "generic")
    return AArch64ARCHNames[static_cast<unsigned>(AK)].ArchBaseExtensions;

  return StringSwitch<uint64_t>(CPU)
#define AARCH64_CPU_NAME(NAME, ID, DEFAULT_EXT)                                \
  .Case(NAME, AArch64ARCHNames[static_cast<unsigned>(ArchKind::ID)]            \
                      .ArchBaseExtensions |                                    \
                  DEFAULT_EXT)
#include "../../include/llvm/TargetParser/AArch64TargetParser.def"
      .Default(AArch64::AEK_INVALID);
}

void AArch64::getFeatureOption(StringRef Name, std::string &Feature) {
  Feature = llvm::StringSwitch<std::string>(Name.substr(1))
#define AARCH64_ARCH_EXT_NAME(NAME, ID, FEATURE, NEGFEATURE, FMV_ID,           \
                              DEP_FEATURES, FMV_PRIORITY)                      \
  .Case(NAME, FEATURE)
#include "../../include/llvm/TargetParser/AArch64TargetParser.def"
                .Default(Name.str());
}

AArch64::ArchKind AArch64::getCPUArchKind(StringRef CPU) {
  if (CPU == "generic")
    return ArchKind::ARMV8A;

  return StringSwitch<AArch64::ArchKind>(CPU)
#define AARCH64_CPU_NAME(NAME, ID, DEFAULT_EXT) .Case(NAME, ArchKind::ID)
#include "../../include/llvm/TargetParser/AArch64TargetParser.def"
      .Default(ArchKind::INVALID);
}

AArch64::ArchKind AArch64::getSubArchArchKind(StringRef SubArch) {
  for (const auto &A : AArch64ARCHNames)
    if (A.getSubArch() == SubArch)
      return A.ID;
  return ArchKind::INVALID;
}

uint64_t AArch64::getCpuSupportsMask(ArrayRef<StringRef> FeatureStrs) {
  uint64_t FeaturesMask = 0;
  for (const StringRef &FeatureStr : FeatureStrs) {
    unsigned Feature = StringSwitch<unsigned>(FeatureStr)
#define AARCH64_ARCH_EXT_NAME(NAME, ID, FEATURE, NEGFEATURE, FMV_ID,           \
                              DEP_FEATURES, FMV_PRIORITY)                      \
  .Case(NAME, llvm::AArch64::FEAT_##FMV_ID)
#include "../../include/llvm/TargetParser/AArch64TargetParser.def"
        ;
    FeaturesMask |= (1ULL << Feature);
  }
  return FeaturesMask;
}

bool AArch64::getExtensionFeatures(uint64_t Extensions,
                                   std::vector<StringRef> &Features) {
  if (Extensions == AArch64::AEK_INVALID)
    return false;

#define AARCH64_ARCH_EXT_NAME(NAME, ID, FEATURE, NEGFEATURE, FMV_ID,           \
                              DEP_FEATURES, FMV_PRIORITY)                      \
  if (Extensions & ID) {                                                       \
    const char *feature = FEATURE;                                             \
    /* INVALID and NONE have no feature name. */                               \
    if (feature)                                                               \
      Features.push_back(feature);                                             \
  }
#include "llvm/TargetParser/AArch64TargetParser.def"

  return true;
}

StringRef AArch64::resolveCPUAlias(StringRef CPU) {
  return StringSwitch<StringRef>(CPU)
#define AARCH64_CPU_ALIAS(ALIAS, NAME) .Case(ALIAS, NAME)
#include "../../include/llvm/TargetParser/AArch64TargetParser.def"
      .Default(CPU);
}

StringRef AArch64::getArchFeature(AArch64::ArchKind AK) {
  return AArch64ARCHNames[static_cast<unsigned>(AK)].ArchFeature;
}

StringRef AArch64::getArchName(AArch64::ArchKind AK) {
  return AArch64ARCHNames[static_cast<unsigned>(AK)].Name;
}

StringRef AArch64::getSubArch(AArch64::ArchKind AK) {
  return AArch64ARCHNames[static_cast<unsigned>(AK)].getSubArch();
}

StringRef AArch64::getArchExtFeature(StringRef ArchExt) {
  if (ArchExt.startswith("no")) {
    StringRef ArchExtBase(ArchExt.substr(2));
    for (const auto &AE : AArch64ARCHExtNames) {
      if (!AE.NegFeature.empty() && ArchExtBase == AE.Name)
        return AE.NegFeature;
    }
  }

  for (const auto &AE : AArch64ARCHExtNames)
    if (!AE.Feature.empty() && ArchExt == AE.Name)
      return AE.Feature;
  return StringRef();
}

AArch64::ArchKind AArch64::convertV9toV8(AArch64::ArchKind AK) {
  if (AK == AArch64::ArchKind::INVALID)
    return AK;
  if (AK < AArch64::ArchKind::ARMV9A)
    return AK;
  if (AK >= AArch64::ArchKind::ARMV8R)
    return AArch64::ArchKind::INVALID;
  unsigned AK_v8 = static_cast<unsigned>(AArch64::ArchKind::ARMV8_5A);
  AK_v8 += static_cast<unsigned>(AK) -
           static_cast<unsigned>(AArch64::ArchKind::ARMV9A);
  return static_cast<AArch64::ArchKind>(AK_v8);
}

void AArch64::fillValidCPUArchList(SmallVectorImpl<StringRef> &Values) {
  for (const auto &Arch : AArch64CPUNames) {
    if (Arch.ArchID != ArchKind::INVALID)
      Values.push_back(Arch.Name);
  }

  for (const auto &Alias: AArch64CPUAliases)
    Values.push_back(Alias.Alias);
}

bool AArch64::isX18ReservedByDefault(const Triple &TT) {
  return TT.isAndroid() || TT.isOSDarwin() || TT.isOSFuchsia() ||
         TT.isOSWindows();
}

// Allows partial match, ex. "v8a" matches "armv8a".
AArch64::ArchKind AArch64::parseArch(StringRef Arch) {
  Arch = llvm::ARM::getCanonicalArchName(Arch);
  if (checkArchVersion(Arch) < 8)
    return ArchKind::INVALID;

  StringRef Syn = llvm::ARM::getArchSynonym(Arch);
  for (const auto &A : AArch64ARCHNames) {
    if (A.Name.endswith(Syn))
      return A.ID;
  }
  return ArchKind::INVALID;
}

AArch64::ArchExtKind AArch64::parseArchExt(StringRef ArchExt) {
  for (const auto &A : AArch64ARCHExtNames) {
    if (ArchExt == A.Name)
      return static_cast<ArchExtKind>(A.ID);
  }
  return AArch64::AEK_INVALID;
}

AArch64::ArchKind AArch64::parseCPUArch(StringRef CPU) {
  // Resolve aliases first.
  for (const auto &Alias : AArch64CPUAliases) {
    if (CPU == Alias.Alias) {
      CPU = Alias.Name;
      break;
    }
  }
  // Then find the CPU name.
  for (const auto &C : AArch64CPUNames)
    if (CPU == C.Name)
      return C.ArchID;

  return ArchKind::INVALID;
}
