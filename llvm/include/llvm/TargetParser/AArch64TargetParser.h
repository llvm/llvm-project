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
#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VersionTuple.h"
#include <array>
#include <vector>

namespace llvm {

class Triple;

namespace AArch64 {

struct ArchInfo;
struct CpuInfo;

// Function Multi Versioning CPU features. They must be kept in sync with
// compiler-rt enum CPUFeatures in lib/builtins/cpu_model/aarch64.c with
// FEAT_MAX as sentinel.
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
  FEAT_RCPC3,
  FEAT_MOPS,
  FEAT_MAX,
  FEAT_EXT = 62,
  FEAT_INIT
};

static_assert(FEAT_MAX < 62,
              "Number of features in CPUFeatures are limited to 62 entries");

// Each ArchExtKind correponds directly to a possible -target-feature.
#define EMIT_ARCHEXTKIND_ENUM
#include "llvm/TargetParser/AArch64TargetParserDef.inc"

using ExtensionBitset = Bitset<AEK_NUM_EXTENSIONS>;

// Represents an extension that can be enabled with -march=<arch>+<extension>.
// Typically these correspond to Arm Architecture extensions, unlike
// SubtargetFeature which may represent either an actual extension or some
// internal LLVM property.
struct ExtensionInfo {
  StringRef Name;                 // Human readable name, e.g. "profile".
  std::optional<StringRef> Alias; // An alias for this extension, if one exists.
  ArchExtKind ID;                 // Corresponding to the ArchExtKind, this
                                  // extensions representation in the bitfield.
  StringRef Feature;              // -mattr enable string, e.g. "+spe"
  StringRef NegFeature;           // -mattr disable string, e.g. "-spe"
  CPUFeatures CPUFeature;      // Function Multi Versioning (FMV) bitfield value
                               // set in __aarch64_cpu_features
  StringRef DependentFeatures; // FMV enabled features string,
                               // e.g. "+dotprod,+fp-armv8,+neon"
  unsigned FmvPriority;        // FMV feature priority
  static constexpr unsigned MaxFMVPriority =
      1000; // Maximum priority for FMV feature
};

#define EMIT_EXTENSIONS
#include "llvm/TargetParser/AArch64TargetParserDef.inc"

// Represents a dependency between two architecture extensions. Later is the
// feature which was added to the architecture after Earlier, and expands the
// functionality provided by it. If Later is enabled, then Earlier will also be
// enabled. If Earlier is disabled, then Later will also be disabled.
struct ExtensionDependency {
  ArchExtKind Earlier;
  ArchExtKind Later;
};

#define EMIT_EXTENSION_DEPENDENCIES
#include "llvm/TargetParser/AArch64TargetParserDef.inc"

enum ArchProfile { AProfile = 'A', RProfile = 'R', InvalidProfile = '?' };

// Information about a specific architecture, e.g. V8.1-A
struct ArchInfo {
  VersionTuple Version;  // Architecture version, major + minor.
  ArchProfile Profile;   // Architecuture profile
  StringRef Name;        // Name as supplied to -march e.g. "armv8.1-a"
  StringRef ArchFeature; // Name as supplied to -target-feature, e.g. "+v8a"
  AArch64::ExtensionBitset
      DefaultExts; // bitfield of default extensions ArchExtKind

  bool operator==(const ArchInfo &Other) const {
    return this->Name == Other.Name;
  }
  bool operator!=(const ArchInfo &Other) const {
    return this->Name != Other.Name;
  }

  // Defines the following partial order, indicating when an architecture is
  // a superset of another:
  //
  //   v9.5a > v9.4a > v9.3a > v9.2a > v9.1a > v9a;
  //             v       v       v       v       v
  //           v8.9a > v8.8a > v8.7a > v8.6a > v8.5a > v8.4a > ... > v8a;
  //
  // v8r has no relation to anything. This is used to determine which
  // features to enable for a given architecture. See
  // AArch64TargetInfo::setFeatureEnabled.
  bool implies(const ArchInfo &Other) const {
    if (this->Profile != Other.Profile)
      return false; // ARMV8R
    if (this->Version.getMajor() == Other.Version.getMajor()) {
      return this->Version > Other.Version;
    }
    if (this->Version.getMajor() == 9 && Other.Version.getMajor() == 8) {
      assert(this->Version.getMinor() && Other.Version.getMinor() &&
             "AArch64::ArchInfo should have a minor version.");
      return this->Version.getMinor().value_or(0) + 5 >=
             Other.Version.getMinor().value_or(0);
    }
    return false;
  }

  // True if this architecture is a superset of Other (including being equal to
  // it).
  bool is_superset(const ArchInfo &Other) const {
    return (*this == Other) || implies(Other);
  }

  // Return ArchFeature without the leading "+".
  StringRef getSubArch() const { return ArchFeature.substr(1); }

  // Search for ArchInfo by SubArch name
  static std::optional<ArchInfo> findBySubArch(StringRef SubArch);
};

#define EMIT_ARCHITECTURES
#include "llvm/TargetParser/AArch64TargetParserDef.inc"

// Details of a specific CPU.
struct CpuInfo {
  StringRef Name; // Name, as written for -mcpu.
  const ArchInfo &Arch;
  AArch64::ExtensionBitset
      DefaultExtensions; // Default extensions for this CPU. These will be
                         // ORd with the architecture defaults.

  AArch64::ExtensionBitset getImpliedExtensions() const {
    AArch64::ExtensionBitset ImpliedExts;
    ImpliedExts |= DefaultExtensions;
    ImpliedExts |= Arch.DefaultExts;
    return ImpliedExts;
  }
};

#define EMIT_CPU_INFO
#include "llvm/TargetParser/AArch64TargetParserDef.inc"

struct ExtensionSet {
  // Set of extensions which are currently enabled.
  ExtensionBitset Enabled;
  // Set of extensions which have been enabled or disabled at any point. Used
  // to avoid cluttering the cc1 command-line with lots of unneeded features.
  ExtensionBitset Touched;
  // Base architecture version, which we need to know because some feature
  // dependencies change depending on this.
  const ArchInfo *BaseArch;

  ExtensionSet() : Enabled(), Touched(), BaseArch(nullptr) {}

  // Enable the given architecture extension, and any other extensions it
  // depends on. Does not change the base architecture, or follow dependencies
  // between features which are only related by required arcitecture versions.
  void enable(ArchExtKind E);

  // Disable the given architecture extension, and any other extensions which
  // depend on it. Does not change the base architecture, or follow
  // dependencies between features which are only related by required
  // arcitecture versions.
  void disable(ArchExtKind E);

  // Add default extensions for the given CPU. Records the base architecture,
  // to later resolve dependencies which depend on it.
  void addCPUDefaults(const CpuInfo &CPU);

  // Add default extensions for the given architecture version. Records the
  // base architecture, to later resolve dependencies which depend on it.
  void addArchDefaults(const ArchInfo &Arch);

  // Add or remove a feature based on a modifier string. The string must be of
  // the form "<name>" to enable a feature or "no<name>" to disable it. This
  // will also enable or disable any features as required by the dependencies
  // between them.
  bool parseModifier(StringRef Modifier, const bool AllowNoDashForm = false);

  // Constructs a new ExtensionSet by toggling the corresponding bits for every
  // feature in the \p Features list without expanding their dependencies. Used
  // for reconstructing an ExtensionSet from the output of toLLVMFeatures().
  // Features that are not recognized are pushed back to \p NonExtensions.
  void reconstructFromParsedFeatures(const std::vector<std::string> &Features,
                                     std::vector<std::string> &NonExtensions);

  // Convert the set of enabled extension to an LLVM feature list, appending
  // them to Features.
  template <typename T> void toLLVMFeatureList(std::vector<T> &Features) const {
    if (BaseArch && !BaseArch->ArchFeature.empty())
      Features.emplace_back(T(BaseArch->ArchFeature));

    for (const auto &E : Extensions) {
      if (E.Feature.empty() || !Touched.test(E.ID))
        continue;
      if (Enabled.test(E.ID))
        Features.emplace_back(T(E.Feature));
      else
        Features.emplace_back(T(E.NegFeature));
    }
  }
};

// Name alias.
struct Alias {
  StringRef AltName;
  StringRef Name;
};

inline constexpr Alias CpuAliases[] = {{"cobalt-100", "neoverse-n2"},
                                       {"grace", "neoverse-v2"}};

const ExtensionInfo &getExtensionByID(ArchExtKind(ExtID));

bool getExtensionFeatures(
    const AArch64::ExtensionBitset &Extensions,
    std::vector<StringRef> &Features);

StringRef getArchExtFeature(StringRef ArchExt);
StringRef resolveCPUAlias(StringRef CPU);

// Information by Name
const ArchInfo *getArchForCpu(StringRef CPU);

// Parser
const ArchInfo *parseArch(StringRef Arch);

// Return the extension which has the given -target-feature name.
std::optional<ExtensionInfo> targetFeatureToExtension(StringRef TargetFeature);

// Parse a name as defined by the Extension class in tablegen.
std::optional<ExtensionInfo> parseArchExtension(StringRef Extension);

// Given the name of a CPU or alias, return the correponding CpuInfo.
std::optional<CpuInfo> parseCpu(StringRef Name);
// Used by target parser tests
void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values);

bool isX18ReservedByDefault(const Triple &TT);

// For given feature names, return a bitmask corresponding to the entries of
// AArch64::CPUFeatures. The values in CPUFeatures are not bitmasks
// themselves, they are sequential (0, 1, 2, 3, ...).
uint64_t getCpuSupportsMask(ArrayRef<StringRef> FeatureStrs);

void PrintSupportedExtensions(StringMap<StringRef> DescMap);

} // namespace AArch64
} // namespace llvm

#endif
