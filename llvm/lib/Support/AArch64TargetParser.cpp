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
#include <cctype>

using namespace llvm;

static unsigned checkArchVersion(llvm::StringRef Arch) {
  if (Arch.size() >= 2 && Arch[0] == 'v' && std::isdigit(Arch[1]))
    return (Arch[1] - 48);
  return 0;
}

unsigned AArch64::getDefaultFPU(StringRef CPU, AArch64::ArchKind AK) {
  if (CPU == "generic")
    return AArch64ARCHNames[static_cast<unsigned>(AK)].DefaultFPU;

  return StringSwitch<unsigned>(CPU)
#define AARCH64_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT)       \
  .Case(NAME, ARM::DEFAULT_FPU)
#include "../../include/llvm/Support/AArch64TargetParser.def"
  .Default(ARM::FK_INVALID);
}

uint64_t AArch64::getDefaultExtensions(StringRef CPU, AArch64::ArchKind AK) {
  if (CPU == "generic")
    return AArch64ARCHNames[static_cast<unsigned>(AK)].ArchBaseExtensions;

  return StringSwitch<uint64_t>(CPU)
#define AARCH64_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT)       \
  .Case(NAME, AArch64ARCHNames[static_cast<unsigned>(ArchKind::ID)]            \
                      .ArchBaseExtensions |                                    \
                  DEFAULT_EXT)
#include "../../include/llvm/Support/AArch64TargetParser.def"
  .Default(AArch64::AEK_INVALID);
}

AArch64::ArchKind AArch64::getCPUArchKind(StringRef CPU) {
  if (CPU == "generic")
    return ArchKind::ARMV8A;

  return StringSwitch<AArch64::ArchKind>(CPU)
#define AARCH64_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT)       \
  .Case(NAME, ArchKind::ID)
#include "../../include/llvm/Support/AArch64TargetParser.def"
  .Default(ArchKind::INVALID);
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

bool AArch64::getArchFeatures(AArch64::ArchKind AK,
                              std::vector<StringRef> &Features) {
  if (AK == ArchKind::ARMV8A)
    Features.push_back("+v8a");
  if (AK == ArchKind::ARMV8_1A)
    Features.push_back("+v8.1a");
  if (AK == ArchKind::ARMV8_2A)
    Features.push_back("+v8.2a");
  if (AK == ArchKind::ARMV8_3A)
    Features.push_back("+v8.3a");
  if (AK == ArchKind::ARMV8_4A)
    Features.push_back("+v8.4a");
  if (AK == ArchKind::ARMV8_5A)
    Features.push_back("+v8.5a");
  if (AK == AArch64::ArchKind::ARMV8_6A)
    Features.push_back("+v8.6a");
  if (AK == AArch64::ArchKind::ARMV8_7A)
    Features.push_back("+v8.7a");
  if (AK == AArch64::ArchKind::ARMV8_8A)
    Features.push_back("+v8.8a");
  if (AK == AArch64::ArchKind::ARMV9A)
    Features.push_back("+v9a");
  if (AK == AArch64::ArchKind::ARMV9_1A)
    Features.push_back("+v9.1a");
  if (AK == AArch64::ArchKind::ARMV9_2A)
    Features.push_back("+v9.2a");
  if (AK == AArch64::ArchKind::ARMV9_3A)
    Features.push_back("+v9.3a");
  if(AK == AArch64::ArchKind::ARMV8R)
    Features.push_back("+v8r");

  return AK != ArchKind::INVALID;
}

StringRef AArch64::getArchName(AArch64::ArchKind AK) {
  return AArch64ARCHNames[static_cast<unsigned>(AK)].getName();
}

StringRef AArch64::getCPUAttr(AArch64::ArchKind AK) {
  return AArch64ARCHNames[static_cast<unsigned>(AK)].getCPUAttr();
}

StringRef AArch64::getSubArch(AArch64::ArchKind AK) {
  return AArch64ARCHNames[static_cast<unsigned>(AK)].getSubArch();
}

unsigned AArch64::getArchAttr(AArch64::ArchKind AK) {
  return AArch64ARCHNames[static_cast<unsigned>(AK)].ArchAttr;
}

StringRef AArch64::getArchExtName(unsigned ArchExtKind) {
  for (const auto &AE : AArch64ARCHExtNames)
    if (ArchExtKind == AE.ID)
      return AE.getName();
  return StringRef();
}

StringRef AArch64::getArchExtFeature(StringRef ArchExt) {
  if (ArchExt.startswith("no")) {
    StringRef ArchExtBase(ArchExt.substr(2));
    for (const auto &AE : AArch64ARCHExtNames) {
      if (AE.NegFeature && ArchExtBase == AE.getName())
        return StringRef(AE.NegFeature);
    }
  }

  for (const auto &AE : AArch64ARCHExtNames)
    if (AE.Feature && ArchExt == AE.getName())
      return StringRef(AE.Feature);
  return StringRef();
}

StringRef AArch64::getDefaultCPU(StringRef Arch) {
  ArchKind AK = parseArch(Arch);
  if (AK == ArchKind::INVALID)
    return StringRef();

  // Look for multiple AKs to find the default for pair AK+Name.
  for (const auto &CPU : AArch64CPUNames)
    if (CPU.ArchID == AK && CPU.Default)
      return CPU.getName();

  // If we can't find a default then target the architecture instead
  return "generic";
}

void AArch64::fillValidCPUArchList(SmallVectorImpl<StringRef> &Values) {
  for (const auto &Arch : AArch64CPUNames) {
    if (Arch.ArchID != ArchKind::INVALID)
      Values.push_back(Arch.getName());
  }
}

bool AArch64::isX18ReservedByDefault(const Triple &TT) {
  return TT.isAndroid() || TT.isOSDarwin() || TT.isOSFuchsia() ||
         TT.isOSWindows();
}

// Allows partial match, ex. "v8a" matches "armv8a".
AArch64::ArchKind AArch64::parseArch(StringRef Arch) {
  Arch = ARM::getCanonicalArchName(Arch);
  if (checkArchVersion(Arch) < 8)
    return ArchKind::INVALID;

  StringRef Syn = ARM::getArchSynonym(Arch);
  for (const auto &A : AArch64ARCHNames) {
    if (A.getName().endswith(Syn))
      return A.ID;
  }
  return ArchKind::INVALID;
}

AArch64::ArchExtKind AArch64::parseArchExt(StringRef ArchExt) {
  for (const auto &A : AArch64ARCHExtNames) {
    if (ArchExt == A.getName())
      return static_cast<ArchExtKind>(A.ID);
  }
  return AArch64::AEK_INVALID;
}

AArch64::ArchKind AArch64::parseCPUArch(StringRef CPU) {
  for (const auto &C : AArch64CPUNames) {
    if (CPU == C.getName())
      return C.ArchID;
  }
  return ArchKind::INVALID;
}
