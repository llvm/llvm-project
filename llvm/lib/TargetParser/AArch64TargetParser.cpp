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
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/ARMTargetParserCommon.h"
#include "llvm/TargetParser/Triple.h"
#include <cctype>

#define DEBUG_TYPE "target-parser"

using namespace llvm;

static unsigned checkArchVersion(llvm::StringRef Arch) {
  if (Arch.size() >= 2 && Arch[0] == 'v' && std::isdigit(Arch[1]))
    return (Arch[1] - 48);
  return 0;
}

const AArch64::ArchInfo *AArch64::getArchForCpu(StringRef CPU) {
  if (CPU == "generic")
    return &ARMV8A;

  // Note: this now takes cpu aliases into account
  std::optional<CpuInfo> Cpu = parseCpu(CPU);
  if (!Cpu)
    return nullptr;
  return &Cpu->Arch;
}

std::optional<AArch64::ArchInfo> AArch64::ArchInfo::findBySubArch(StringRef SubArch) {
  for (const auto *A : AArch64::ArchInfos)
    if (A->getSubArch() == SubArch)
      return *A;
  return {};
}

uint64_t AArch64::getCpuSupportsMask(ArrayRef<StringRef> FeatureStrs) {
  uint64_t FeaturesMask = 0;
  for (const StringRef &FeatureStr : FeatureStrs) {
    if (auto Ext = parseArchExtension(FeatureStr))
      FeaturesMask |= (1ULL << Ext->CPUFeature);
  }
  return FeaturesMask;
}

bool AArch64::getExtensionFeatures(
    const AArch64::ExtensionBitset &InputExts,
    std::vector<StringRef> &Features) {
  for (const auto &E : Extensions)
    /* INVALID and NONE have no feature name. */
    if (InputExts.test(E.ID) && !E.Feature.empty())
      Features.push_back(E.Feature);

  return true;
}

StringRef AArch64::resolveCPUAlias(StringRef Name) {
  for (const auto &A : CpuAliases)
    if (A.Alias == Name)
      return A.Name;
  return Name;
}

StringRef AArch64::getArchExtFeature(StringRef ArchExt) {
  bool IsNegated = ArchExt.starts_with("no");
  StringRef ArchExtBase = IsNegated ? ArchExt.drop_front(2) : ArchExt;

  if (auto AE = parseArchExtension(ArchExtBase)) {
    // Note: the returned string can be empty.
    return IsNegated ? AE->NegFeature : AE->Feature;
  }

  return StringRef();
}

void AArch64::fillValidCPUArchList(SmallVectorImpl<StringRef> &Values) {
  for (const auto &C : CpuInfos)
      Values.push_back(C.Name);

  for (const auto &Alias : CpuAliases)
    Values.push_back(Alias.Alias);
}

bool AArch64::isX18ReservedByDefault(const Triple &TT) {
  return TT.isAndroid() || TT.isOSDarwin() || TT.isOSFuchsia() ||
         TT.isOSWindows() || TT.isOHOSFamily();
}

// Allows partial match, ex. "v8a" matches "armv8a".
const AArch64::ArchInfo *AArch64::parseArch(StringRef Arch) {
  Arch = llvm::ARM::getCanonicalArchName(Arch);
  if (checkArchVersion(Arch) < 8)
    return {};

  StringRef Syn = llvm::ARM::getArchSynonym(Arch);
  for (const auto *A : ArchInfos) {
    if (A->Name.ends_with(Syn))
      return A;
  }
  return {};
}

std::optional<AArch64::ExtensionInfo> AArch64::parseArchExtension(StringRef ArchExt) {
  for (const auto &A : Extensions) {
    if (ArchExt == A.Name)
      return A;
  }
  return {};
}

std::optional<AArch64::CpuInfo> AArch64::parseCpu(StringRef Name) {
  // Resolve aliases first.
  Name = resolveCPUAlias(Name);

  // Then find the CPU name.
  for (const auto &C : CpuInfos)
    if (Name == C.Name)
      return C;

  return {};
}

void AArch64::PrintSupportedExtensions(StringMap<StringRef> DescMap) {
  outs() << "All available -march extensions for AArch64\n\n"
         << "    " << left_justify("Name", 20)
         << (DescMap.empty() ? "\n" : "Description\n");
  for (const auto &Ext : Extensions) {
    // Extensions without a feature cannot be used with -march.
    if (!Ext.Feature.empty()) {
      std::string Description = DescMap[Ext.Name].str();
      outs() << "    "
             << format(Description.empty() ? "%s\n" : "%-20s%s\n",
                       Ext.Name.str().c_str(), Description.c_str());
    }
  }
}

const llvm::AArch64::ExtensionInfo &
lookupExtensionByID(llvm::AArch64::ArchExtKind ExtID) {
  for (const auto &E : llvm::AArch64::Extensions)
    if (E.ID == ExtID)
      return E;
  llvm_unreachable("Invalid extension ID");
}

void AArch64::ExtensionSet::enable(ArchExtKind E) {
  if (Enabled.test(E))
    return;

  LLVM_DEBUG(llvm::dbgs() << "Enable " << lookupExtensionByID(E).Name << "\n");

  Touched.set(E);
  Enabled.set(E);

  // Recursively enable all features that this one depends on. This handles all
  // of the simple cases, where the behaviour doesn't depend on the base
  // architecture version.
  for (auto Dep : ExtensionDependencies)
    if (E == Dep.Later)
      enable(Dep.Earlier);

  // Special cases for dependencies which vary depending on the base
  // architecture version.
  if (BaseArch) {
    // +sve implies +f32mm if the base architecture is v8.6A+ or v9.1A+
    // It isn't the case in general that sve implies both f64mm and f32mm
    if (E == AEK_SVE && BaseArch->is_superset(ARMV8_6A))
      enable(AEK_F32MM);

    // +fp16 implies +fp16fml for v8.4A+, but not v9.0-A+
    if (E == AEK_FP16 && BaseArch->is_superset(ARMV8_4A) &&
        !BaseArch->is_superset(ARMV9A))
      enable(AEK_FP16FML);

    // For all architectures, +crypto enables +aes and +sha2.
    if (E == AEK_CRYPTO) {
      enable(AEK_AES);
      enable(AEK_SHA2);
    }

    // For v8.4A+ and v9.0A+, +crypto also enables +sha3 and +sm4.
    if (E == AEK_CRYPTO && BaseArch->is_superset(ARMV8_4A)) {
      enable(AEK_SHA3);
      enable(AEK_SM4);
    }
  }
}

void AArch64::ExtensionSet::disable(ArchExtKind E) {
  // -crypto always disables aes, sha2, sha3 and sm4, even for architectures
  // where the latter two would not be enabled by +crypto.
  if (E == AEK_CRYPTO) {
    disable(AEK_AES);
    disable(AEK_SHA2);
    disable(AEK_SHA3);
    disable(AEK_SM4);
  }

  if (!Enabled.test(E))
    return;

  LLVM_DEBUG(llvm::dbgs() << "Disable " << lookupExtensionByID(E).Name << "\n");

  Touched.set(E);
  Enabled.reset(E);

  // Recursively disable all features that depends on this one.
  for (auto Dep : ExtensionDependencies)
    if (E == Dep.Earlier)
      disable(Dep.Later);
}

void AArch64::ExtensionSet::toLLVMFeatureList(
    std::vector<StringRef> &Features) const {
  if (BaseArch && !BaseArch->ArchFeature.empty())
    Features.push_back(BaseArch->ArchFeature);

  for (const auto &E : Extensions) {
    if (E.Feature.empty() || !Touched.test(E.ID))
      continue;
    if (Enabled.test(E.ID))
      Features.push_back(E.Feature);
    else
      Features.push_back(E.NegFeature);
  }
}

void AArch64::ExtensionSet::addCPUDefaults(const CpuInfo &CPU) {
  LLVM_DEBUG(llvm::dbgs() << "addCPUDefaults(" << CPU.Name << ")\n");
  BaseArch = &CPU.Arch;

  AArch64::ExtensionBitset CPUExtensions = CPU.getImpliedExtensions();
  for (const auto &E : Extensions)
    if (CPUExtensions.test(E.ID))
      enable(E.ID);
}

void AArch64::ExtensionSet::addArchDefaults(const ArchInfo &Arch) {
  LLVM_DEBUG(llvm::dbgs() << "addArchDefaults(" << Arch.Name << ")\n");
  BaseArch = &Arch;

  for (const auto &E : Extensions)
    if (Arch.DefaultExts.test(E.ID))
      enable(E.ID);
}

bool AArch64::ExtensionSet::parseModifier(StringRef Modifier) {
  LLVM_DEBUG(llvm::dbgs() << "parseModifier(" << Modifier << ")\n");

  // Negative modifiers, with the syntax "no<feat>"
  if (Modifier.starts_with("no")) {
    StringRef ModifierBase(Modifier.substr(2));
    for (const auto &AE : Extensions) {
      if (!AE.NegFeature.empty() && ModifierBase == AE.Name) {
        disable(AE.ID);
        return true;
      }
    }
  }

  // Positive modifiers
  for (const auto &AE : Extensions) {
    if (!AE.Feature.empty() && Modifier == AE.Name) {
      enable(AE.ID);
      return true;
    }
  }

  return false;
}
