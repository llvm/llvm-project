//===-- IntelGPUTargetParser.h - Parser for Intel GPU targets ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file names the Intel GPU a driver reports, and finds the Intel GPU a
// user names.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H
#define LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H

#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <string>

namespace llvm {

namespace IntelGPU {

/// Intel GPU architecture names, covering both physical devices and the
/// compatibility names that stand for a whole product line.
///
/// The underlying type is fixed because clang/Basic/OffloadArch.h forward
/// declares this enumeration; the two declarations have to agree.
enum GPUKind : uint8_t {
  GK_NONE = 0,
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  GK_##KIND,
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_TARGET, IGCA_FEATURE_SETS) GK_##KIND,
#include "llvm/TargetParser/IntelGPUTargetParser.def"
};

/// Represents an IGCA feature set, including whether or not it is exact. The
/// values are bitfields where the lowest bit represents exact. See IGCATarget
/// for the packed representation.
enum class IGCAFeatureSet : uint8_t {
  Core = 0b000,
  Compute = 0b010,
  ComputeExact = 0b011,
  Render = 0b100,
  RenderExact = 0b101,
};

/// Wrapper around an IGCA target's uint32_t representation, packed using the
/// following format:
///
///    31              16 15                     3 2             0
///   +------------------+------------------------+---------------+
///   |      Target      |        Reserved        |  Feature set  |
///   +------------------+------------------------+---------------+
///          16 bits              13 bits              3 bits
///
/// Note that bit 0 is a flag that represents if the feature set is "exact".
/// Only IGCA targets that some device in IntelGPUTargetParser.def can run are
/// valid; any other value is stored as invalid(), whose packed representation
/// is 0.
class IGCATarget {
  uint32_t V = 0;

  // Positions of the fields in V; see the bit layout above the class.
  static constexpr uint32_t TargetShift = 16;
  static constexpr uint32_t FeatureSetMask = 0x7;
  static constexpr uint32_t IsExactMask = 0x1;
  /// Bits 3-15, which no IGCA target uses.
  static constexpr uint32_t ReservedMask =
      ((uint32_t(1) << TargetShift) - 1) & ~FeatureSetMask;

  /// \return true if some device in IntelGPUTargetParser.def with the same
  /// numeric IGCA target can run code built for the IGCA target in \p Packed.
  LLVM_ABI static bool isKnownIGCATarget(uint32_t Packed);

  /// Creates an invalid IGCA target, whose packed representation is 0, for
  /// invalid().
  constexpr IGCATarget() = default;

  explicit IGCATarget(uint32_t Packed) {
    // IGCA targets that no device can run (including Target == 0) are
    // invalid, so V remains 0. This keeps V == 0 as the only canonical invalid
    // representation.
    if (isKnownIGCATarget(Packed))
      V = Packed;
  }

public:
  IGCATarget(uint16_t Target, IGCAFeatureSet FeatureSet)
      : IGCATarget(uint32_t(Target) << TargetShift |
                   (llvm::to_underlying(FeatureSet) & FeatureSetMask)) {}

  /// \return an invalid IGCATarget (whose packed representation is 0).
  static constexpr IGCATarget invalid() { return IGCATarget(); }

  /// \return the packed uint32_t representation of the IGCA target.
  constexpr uint32_t pack() const { return V; }
  /// Wrap a packed uint32_t IGCA target with an IGCATarget class.
  static IGCATarget unpack(uint32_t V) { return IGCATarget(V); }

  constexpr uint16_t getTarget() const { return uint16_t(V >> TargetShift); }
  /// \return the feature set, including whether it is exact.
  ///
  /// To check if a target has a feature set regardless of exactness, use
  /// isAnyCompute() or isAnyRender() instead of raw comparison between
  /// IGCAFeatureSet enums.
  constexpr IGCAFeatureSet getFeatureSet() const {
    return IGCAFeatureSet(V & FeatureSetMask);
  }
  constexpr bool isExact() const { return V & IsExactMask; }
  constexpr bool isValid() const { return getTarget() != 0; }
  constexpr explicit operator bool() const { return isValid(); }

  /// \return true if the target has the Core feature set.
  constexpr bool isCore() const {
    // invalid() has the same feature set bits as Core, so we check explicitly.
    return isValid() && getFeatureSet() == IGCAFeatureSet::Core;
  }
  constexpr bool isComputeExact() const {
    return getFeatureSet() == IGCAFeatureSet::ComputeExact;
  }
  constexpr bool isRenderExact() const {
    return getFeatureSet() == IGCAFeatureSet::RenderExact;
  }
  /// \return true if target has either Compute or ComputeExact feature set.
  constexpr bool isAnyCompute() const {
    return getFeatureSet() == IGCAFeatureSet::Compute ||
           getFeatureSet() == IGCAFeatureSet::ComputeExact;
  }
  /// \return true if target has either Render or RenderExact feature set.
  constexpr bool isAnyRender() const {
    return getFeatureSet() == IGCAFeatureSet::Render ||
           getFeatureSet() == IGCAFeatureSet::RenderExact;
  }

  friend constexpr bool operator==(IGCATarget A, IGCATarget B) {
    return A.V == B.V;
  }
  friend constexpr bool operator!=(IGCATarget A, IGCATarget B) {
    return !(A == B);
  }
};

/// \return the device name that \p GPUIPVersion identifies, as reported by the
/// driver, e.g. "xe-pvc". If the table lists no such device, return an empty
/// string. Only the major and minor versions are looked at, so every revision
/// of a device resolves to the same name.
LLVM_ABI StringRef getArchName(uint32_t GPUIPVersion);

/// \return the human-friendly name of \p Kind, e.g. "xe-pvc", or an empty
/// string for GK_NONE.
LLVM_ABI StringRef getArchName(GPUKind Kind);

/// \return the numeric name of \p GPUIPVersion, e.g. "xe_35.11.0".
LLVM_ABI std::string getNumericArchName(uint32_t GPUIPVersion);

/// \return the device \p Name denotes, or GK_NONE for a name this build does
/// not know.
///
/// \p Name may be a human-friendly name ("xe-pvc"), a compatibility name
/// ("xe-dg2"), an alias ("xe-arl-u") or a numeric name ("xe_12.60.7"). A
/// numeric name must end in a revision, which may be any number and is ignored,
/// so "xe_12.60.0" and "xe_12.60.7" denote the same device.
LLVM_ABI GPUKind parseArch(StringRef Name);

/// Parse an IGCA target string such as "igca_20ca".
///
/// \return an invalid IGCATarget if no device in IntelGPUTargetParser.def can
/// run \p MaybeTarget, e.g. "igca_10c".
LLVM_ABI IGCATarget parseIGCATarget(StringRef MaybeTarget);

/// \return the \p Target as a string, e.g. "igca_20ca", or an empty string if
/// \p Target is invalid.
LLVM_ABI StringRef getIGCATargetName(IGCATarget Target);

} // namespace IntelGPU
} // namespace llvm

#endif // LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H
