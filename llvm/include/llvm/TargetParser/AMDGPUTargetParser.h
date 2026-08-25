//===-- AMDGPUTargetParser - Parser for AMDGPU features ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise AMDGPU hardware features.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_AMDGPUTARGETPARSER_H
#define LLVM_TARGETPARSER_AMDGPUTARGETPARSER_H

#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace llvm {

class raw_ostream;
template <typename T> class SmallVectorImpl;
class Triple;

namespace AMDGPU {

/// GPU kinds supported by the AMDGPU target.
enum GPUKind : uint8_t {
  // Not specified processor.
  GK_NONE = 0,

#define GET_R600_GPU_ENUM
#include "llvm/TargetParser/R600TargetParserDef.inc"

#define GET_AMDGPU_GPU_ENUM
#include "llvm/TargetParser/AMDGPUTargetParserDef.inc"
};

/// One enumerator per frontend-visible feature bit; NUM_FEATURES is the count.
enum AMDGPUFeature : unsigned {
#define GET_AMDGPU_FEATURE_ENUM
#include "llvm/TargetParser/AMDGPUTargetParserDef.inc"
};

using AMDGPUFeatureBitset = Bitset<NUM_FEATURES>;

/// Instruction set architecture version.
struct IsaVersion {
  uint8_t Major;
  uint8_t Minor;
  uint8_t Stepping;

  bool operator==(const IsaVersion &Other) const {
    return Major == Other.Major && Minor == Other.Minor &&
           Stepping == Other.Stepping;
  }
  bool operator!=(const IsaVersion &Other) const { return !(*this == Other); }
};

// This isn't comprehensive for now, just things that are needed from the
// frontend driver.
enum R600FeatureKind : uint32_t {
  R600_FEATURE_NONE = 0,

  // Has fma instructions.
  R600_FEATURE_FMA = 1 << 0,
};

// GFX6+ features. This isn't comprehensive for now, just things that are needed
// from the frontend driver.
enum ArchFeatureKind : uint32_t {
  FEATURE_NONE = 0,

  // Common features.
  FEATURE_FAST_FMA_F32 = 1 << 0,
  FEATURE_FAST_DENORMAL_F32 = 1 << 1,

  // Wavefront 32 is available.
  FEATURE_WAVE32 = 1 << 2,

  // Xnack is available.
  FEATURE_XNACK = 1 << 3,

  // Sram-ecc is available.
  FEATURE_SRAMECC = 1 << 4,

  // WGP mode is supported.
  FEATURE_WGP = 1 << 5,

  // Xnack on/off modes are supported.
  FEATURE_XNACK_ON_OFF_MODES = 1 << 6,

  // VI SGPR initialization bug requiring a fixed SGPR allocation size.
  FEATURE_SGPR_INIT_BUG = 1 << 7
};

enum FeatureError : uint32_t {
  NO_ERROR = 0,
  INVALID_FEATURE_COMBINATION,
  UNSUPPORTED_TARGET_FEATURE
};

LLVM_ABI StringRef getArchFamilyNameAMDGCN(GPUKind AK);

/// The canonical GPU name for a variant name.
LLVM_ABI StringRef getBaseArchNameAMDGCN(GPUKind AK);

LLVM_ABI Triple::SubArchType getSubArch(GPUKind AK);
LLVM_ABI Triple::SubArchType getMajorSubArch(Triple::SubArchType SubArch);

/// Return true if subarch \p A is compatible with subarch \p B, i.e. they are
/// equal or one is the major-family subarch of the other (e.g. AMDGPUSubArch9
/// is compatible with AMDGPUSubArch900). NoSubArch is compatible with anything.
LLVM_ABI bool isSubArchCompatible(const Triple &A, const Triple &B);
LLVM_ABI bool isSubArchCompatible(Triple::SubArchType A, Triple::SubArchType B);

/// Return true if the GPU \p AK is usable with the triple subarch \p SubArch.
/// A NoSubArch triple (legacy "amdgcn") accepts any GPU. Otherwise the GPU's
/// subarch must equal \p SubArch, or \p SubArch must be the major-family
/// subarch of the GPU (e.g. the amdgpu9 triple accepts gfx900).
LLVM_ABI bool isCPUValidForSubArch(Triple::SubArchType SubArch, GPUKind AK);

/// Convenience overload of isCPUValidForSubArch taking a GPU name \p CPU, which
/// is parsed via parseArchAMDGCN. An unrecognized name is never valid.
LLVM_ABI bool isCPUValidForSubArch(Triple::SubArchType SubArch, StringRef CPU);

/// Return true if \p AK is a pseudo target (e.g. "generic"/"generic-hsa"): a
/// recognized AMDGCN GPU that represents no concrete hardware and has no
/// subarch of its own. Such targets are resolved by the backend as a default
/// device but are not valid as an explicit -mcpu.
LLVM_ABI bool isPseudoTarget(GPUKind AK);

/// Convenience overload of isPseudoTarget taking a GPU name \p CPU, which is
/// parsed via parseArchAMDGCN.
LLVM_ABI bool isPseudoTarget(StringRef CPU);

/// Returns the effective triple appropriate to use when linking \p B into \p A
/// by merging the subarches in case of inexact match.
///
/// In cases where isSubArchCompatible would return / false, returns \p B. This
/// assumes that the non-arch triple components are the same
LLVM_ABI std::string mergeSubArch(const Triple &A, const Triple &B);

LLVM_ABI StringRef getArchNameAMDGCN(GPUKind AK);
LLVM_ABI StringRef getArchNameR600(GPUKind AK);

/// Returns the canonical GPU name for an AMDGPU subarch, e.g.
/// AMDGPUSubArch1030 -> "gfx1030", AMDGPUSubArch9 -> "gfx9-generic",
/// AMDGPUSubArch6 -> "gfx600". Returns "" for NoSubArch or a non-AMDGPU
/// subarch. The major-only subarches map to their generic/lowest
/// representative, matching the default subtarget for an unspecified -mcpu.
LLVM_ABI StringRef getArchNameFromSubArch(Triple::SubArchType SubArch);

/// Returns the triple subarch name for an AMDGPU subarch, e.g.
/// AMDGPUSubArch900 -> "amdgpu9.00". Returns "amdgpu" for NoSubArch.
LLVM_ABI StringRef getSubArchName(Triple::SubArchType SubArch);
LLVM_ABI StringRef getCanonicalArchName(const Triple &T, StringRef Arch);
LLVM_ABI GPUKind parseArchAMDGCN(StringRef CPU);
LLVM_ABI GPUKind parseArchR600(StringRef CPU);
LLVM_ABI GPUKind getGPUKindFromSubArch(Triple::SubArchType SubArch);
/// \deprecated Use getFeatureBitset and test the relevant FEAT_* bits instead.
/// The legacy ArchFeatureKind bitfield is being removed.
LLVM_DEPRECATED("use getFeatureBitset instead", "getFeatureBitset")
LLVM_ABI unsigned getArchAttrAMDGCN(GPUKind AK);
LLVM_DEPRECATED("use getFeatureBitset instead", "getFeatureBitset")
LLVM_ABI unsigned getArchAttrAMDGCN(Triple::SubArchType SubArch);
LLVM_ABI R600FeatureKind getArchAttrR600(GPUKind AK);

/// Returns \p AK's feature bitset, or an empty bitset if unknown.
LLVM_ABI const AMDGPUFeatureBitset &getFeatureBitset(GPUKind AK);

/// Appends the feature name of each bit set in \p Features to \p Names.
LLVM_ABI void getFeatureNames(const AMDGPUFeatureBitset &Features,
                              SmallVectorImpl<StringRef> &Names);

/// Append the valid AMDGCN GPU names to \p Values. If \p SubArch is not
/// NoSubArch, only GPUs compatible with that subarch (see isCPUValidForSubArch)
/// are appended.
LLVM_ABI void
fillValidArchListAMDGCN(SmallVectorImpl<StringRef> &Values,
                        Triple::SubArchType SubArch = Triple::NoSubArch);
LLVM_ABI void fillValidArchListR600(SmallVectorImpl<StringRef> &Values);

LLVM_ABI IsaVersion getIsaVersion(StringRef GPU);
LLVM_ABI IsaVersion getIsaVersion(Triple::SubArchType SubArch);

enum { FIXED_NUM_SGPRS_FOR_INIT_BUG = 96 };

LLVM_ABI unsigned getTotalNumSGPRs(GPUKind AK);
LLVM_ABI unsigned getTotalNumSGPRs(Triple::SubArchType SubArch);

LLVM_ABI unsigned getAddressableNumSGPRs(GPUKind AK);
LLVM_ABI unsigned getAddressableNumSGPRs(Triple::SubArchType SubArch);

LLVM_ABI unsigned getSGPRAllocGranule(GPUKind AK);
LLVM_ABI unsigned getSGPRAllocGranule(Triple::SubArchType SubArch);

/// Fills Features map with default values for given target GPU.
/// \p Features contains overriding target features and this function returns
/// default target features with entries overridden by \p Features.
LLVM_ABI std::pair<FeatureError, StringRef>
fillAMDGPUFeatureMap(StringRef GPU, const Triple &T, StringMap<bool> &Features);

enum class TargetIDSetting { Unsupported, Any, Off, On };

class LLVM_ABI TargetID {
private:
  GPUKind Arch;
  std::string TargetTripleString;
  TargetIDSetting XnackSetting;
  TargetIDSetting SramEccSetting;
  bool IsAMDHSA;

public:
  TargetID(GPUKind Arch, const Triple &TT, TargetIDSetting XnackSetting,
           TargetIDSetting SramEccSetting);

  /// Construct a TargetID from a triple \p TT and the processor+features string
  /// e.g. "gfx90a", "gfx90a:xnack+:sramecc-", "".
  TargetID(const Triple &TT, StringRef TargetIDStr);

  ~TargetID() = default;

  /// \return True if the current xnack setting is not "Unsupported".
  bool isXnackSupported() const {
    return XnackSetting != TargetIDSetting::Unsupported;
  }

  /// \returns True if the current xnack setting is "On" or "Any".
  bool isXnackOnOrAny() const {
    return XnackSetting == TargetIDSetting::On ||
           XnackSetting == TargetIDSetting::Any;
  }

  /// \returns True if current xnack setting is "On" or "Off",
  /// false otherwise.
  bool isXnackOnOrOff() const {
    return getXnackSetting() == TargetIDSetting::On ||
           getXnackSetting() == TargetIDSetting::Off;
  }

  /// \returns The current xnack TargetIDSetting, possible options are
  /// "Unsupported", "Any", "Off", and "On".
  TargetIDSetting getXnackSetting() const { return XnackSetting; }

  /// Sets xnack setting to \p NewXnackSetting.
  void setXnackSetting(TargetIDSetting NewXnackSetting) {
    XnackSetting = NewXnackSetting;
  }

  /// \return True if the current sramecc setting is not "Unsupported".
  bool isSramEccSupported() const {
    return SramEccSetting != TargetIDSetting::Unsupported;
  }

  /// \returns True if the current sramecc setting is "On" or "Any".
  bool isSramEccOnOrAny() const {
    return SramEccSetting == TargetIDSetting::On ||
           SramEccSetting == TargetIDSetting::Any;
  }

  /// \returns True if current sramecc setting is "On" or "Off",
  /// false otherwise.
  bool isSramEccOnOrOff() const {
    return getSramEccSetting() == TargetIDSetting::On ||
           getSramEccSetting() == TargetIDSetting::Off;
  }

  /// \returns The current sramecc TargetIDSetting, possible options are
  /// "Unsupported", "Any", "Off", and "On".
  TargetIDSetting getSramEccSetting() const { return SramEccSetting; }

  /// Sets sramecc setting to \p NewSramEccSetting.
  void setSramEccSetting(TargetIDSetting NewSramEccSetting) {
    SramEccSetting = NewSramEccSetting;
  }

  GPUKind getGPUKind() const { return Arch; }

  StringRef getTargetTripleString() const { return TargetTripleString; }

  /// \returns True if this is an AMDHSA target.
  bool isAMDHSA() const { return IsAMDHSA; }

  /// Parse and validate a TargetID for triple \p TT from the processor+features
  /// string \p ProcAndFeatures (e.g. "gfx90a", "gfx90a:xnack+:sramecc-", "").
  /// Returns std::nullopt if the triple is not AMDGCN, the processor is
  /// unrecognized, or a feature modifier is invalid for the processor.
  static std::optional<TargetID> parse(const Triple &TT,
                                       StringRef ProcAndFeatures);

  /// Parse and validate a TargetID from a full
  /// "<triple>-<processor>:<features>" directive string.
  static std::optional<TargetID>
  parseTargetIDString(StringRef TargetIDDirective);

  /// Returns true if \p Other denotes the same target as *this, i.e. the same
  /// processor and xnack/sramecc settings on a compatible triple. This is a
  /// semantic equality that looks through spelling differences.
  bool isEquivalent(const TargetID &Other) const;

  /// Returns true if a device image for *this can provide the device code for a
  /// request for \p Other. This is directional and models logical-linking
  /// compatibility.
  bool providesFor(const TargetID &Other) const;

  void print(raw_ostream &OS) const;

  std::string toString() const;

  /// Print the canonical processor name followed by any explicit xnack and
  /// sramecc feature modifiers (e.g. "gfx908:sramecc-:xnack+"), without the
  /// triple prefix.
  void printCanonicalTargetIDString(raw_ostream &OS) const;

  /// \returns the canonical processor name followed by any explicit xnack and
  /// sramecc feature modifiers order (e.g.  "gfx908:sramecc-:xnack+"), without
  /// the triple prefix.
  std::string getCanonicalTargetIDString() const;

  bool operator==(const TargetID &Other) const;
  bool operator!=(const TargetID &Other) const { return !(*this == Other); }
};

inline raw_ostream &operator<<(raw_ostream &OS, const TargetID &TargetID) {
  TargetID.print(OS);
  return OS;
}

} // namespace AMDGPU

} // namespace llvm

#endif
