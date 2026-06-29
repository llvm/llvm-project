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
#define AMDGCN_GPU(NAME, ENUM, ISAVERSION, FEATURES) ENUM,
#include "AMDGPUTargetParser.def"

  GK_AMDGCN_GENERIC_FIRST = GK_GFX9_GENERIC,
  GK_AMDGCN_GENERIC_LAST = GK_GFX13_GENERIC,
};

/// Instruction set architecture version.
struct IsaVersion {
  unsigned Major;
  unsigned Minor;
  unsigned Stepping;
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
  FEATURE_XNACK_ON_OFF_MODES = 1 << 10
};

enum FeatureError : uint32_t {
  NO_ERROR = 0,
  INVALID_FEATURE_COMBINATION,
  UNSUPPORTED_TARGET_FEATURE
};

LLVM_ABI StringRef getArchFamilyNameAMDGCN(GPUKind AK);

LLVM_ABI StringRef getArchNameAMDGCN(GPUKind AK);
LLVM_ABI StringRef getArchNameR600(GPUKind AK);
LLVM_ABI StringRef getCanonicalArchName(const Triple &T, StringRef Arch);
LLVM_ABI GPUKind parseArchAMDGCN(StringRef CPU);
LLVM_ABI GPUKind parseArchR600(StringRef CPU);
LLVM_ABI unsigned getArchAttrAMDGCN(GPUKind AK);
LLVM_ABI unsigned getArchAttrR600(GPUKind AK);

LLVM_ABI void fillValidArchListAMDGCN(SmallVectorImpl<StringRef> &Values);
LLVM_ABI void fillValidArchListR600(SmallVectorImpl<StringRef> &Values);

LLVM_ABI IsaVersion getIsaVersion(StringRef GPU);

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

  void setTargetIDFromTargetIDStream(StringRef TargetID);

  GPUKind getGPUKind() const { return Arch; }

  StringRef getTargetTripleString() const { return TargetTripleString; }

  /// \returns True if this is an AMDHSA target.
  bool isAMDHSA() const { return IsAMDHSA; }

  static std::optional<TargetID>
  parseTargetIDString(StringRef TargetIDDirective);

  void print(raw_ostream &OS) const;

  std::string toString() const;

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
