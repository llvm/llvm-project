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
enum GPUKind : uint32_t {
  // Not specified processor.
  GK_NONE = 0,

#define R600_GPU(NAME, ENUM, FEATURES) ENUM,
#define AMDGCN_GPU(NAME, ENUM, SUBARCH, ISAVERSION, FEATURES) ENUM,
#include "AMDGPUTargetParser.def"

  GK_AMDGCN_GENERIC_FIRST = GK_GFX9_GENERIC,
  GK_AMDGCN_GENERIC_LAST = GK_GFX13_GENERIC,
};

/// Instruction set architecture version.
struct IsaVersion {
  unsigned Major;
  unsigned Minor;
  unsigned Stepping;

  bool operator==(const IsaVersion &Other) const {
    return Major == Other.Major && Minor == Other.Minor &&
           Stepping == Other.Stepping;
  }
  bool operator!=(const IsaVersion &Other) const { return !(*this == Other); }
};

// This isn't comprehensive for now, just things that are needed from the
// frontend driver.
enum ArchFeatureKind : uint32_t {
  FEATURE_NONE = 0,

  // These features only exist for r600, and are implied true for amdgcn.
  FEATURE_FMA = 1 << 1,
  FEATURE_LDEXP = 1 << 2,
  FEATURE_FP64 = 1 << 3,

  // Common features.
  FEATURE_FAST_FMA_F32 = 1 << 4,
  FEATURE_FAST_DENORMAL_F32 = 1 << 5,

  // Wavefront 32 is available.
  FEATURE_WAVE32 = 1 << 6,

  // Xnack is available.
  FEATURE_XNACK = 1 << 7,

  // Sram-ecc is available.
  FEATURE_SRAMECC = 1 << 8,

  // WGP mode is supported.
  FEATURE_WGP = 1 << 9,

  // Xnack on/off modes are supported.
  FEATURE_XNACK_ON_OFF_MODES = 1 << 10,

  // VI SGPR initialization bug requiring a fixed SGPR allocation size.
  FEATURE_SGPR_INIT_BUG = 1 << 11
};

enum FeatureError : uint32_t {
  NO_ERROR = 0,
  INVALID_FEATURE_COMBINATION,
  UNSUPPORTED_TARGET_FEATURE
};

LLVM_ABI StringRef getArchFamilyNameAMDGCN(GPUKind AK);
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
LLVM_ABI StringRef getCanonicalArchName(const Triple &T, StringRef Arch);
LLVM_ABI GPUKind parseArchAMDGCN(StringRef CPU);
LLVM_ABI GPUKind parseArchR600(StringRef CPU);
LLVM_ABI GPUKind getGPUKindFromSubArch(Triple::SubArchType SubArch);
LLVM_ABI unsigned getArchAttrAMDGCN(GPUKind AK);
LLVM_ABI unsigned getArchAttrAMDGCN(Triple::SubArchType SubArch);
LLVM_ABI unsigned getArchAttrR600(GPUKind AK);

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
  std::string getCanonicalFeatureString() const;

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
