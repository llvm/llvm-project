//===-- IntelGPUTargetParser.h - Parser for Intel GPU targets ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides access to the Intel GPU list in IntelGPUTargetParser.def.
// Only what is needed to name the device a driver reports is declared here; the
// table itself carries more, and a consumer that needs the rest either declares
// it here as well or expands the table directly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H
#define LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <string>

namespace llvm {
template <typename T> class SmallVectorImpl;

namespace IntelGPU {

/// Intel GPU architecture names, covering both physical devices and the
/// compatibility names that stand for a whole product line.
enum GPUKind : uint8_t {
  GK_NONE = 0,
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  GK_##KIND,
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_TARGET, IGCA_FEATURE_SETS) GK_##KIND,
#include "llvm/TargetParser/IntelGPUTargetParser.def"
};

enum IGCAFeatureSet : uint8_t {
  IGCA_CORE = 0,
  IGCA_COMPUTE,
  IGCA_RENDER
};

/// Wrapper around an IGCA target's uint32_t representation, packed using the
/// following format:
///
///    31              16 15           3 2           1 0         0
///   +------------------+--------------+-------------+-----------+
///   |      Target      |   Reserved   | Feature set | Is Exact? |
///   +------------------+--------------+-------------+-----------+
///          16 bits         13 bits        2 bits        1 bit
///
/// Target == 0 denotes an invalid IGCA Target.
class IGCATarget {
  uint32_t V = 0;
  constexpr explicit IGCATarget(uint32_t V) : V(V) {}

public:
  static constexpr uint32_t TargetShift = 16;
  static constexpr uint32_t FeatureSetShift = 1;
  static constexpr uint32_t IsExactShift = 0;
  static constexpr uint32_t FeatureSetMask = 0x3;
  static constexpr uint32_t IsExactMask = 0x1;

  constexpr IGCATarget(uint16_t Target, IGCAFeatureSet FeatureSet, bool IsExact)
      : V(uint32_t(Target) << TargetShift |
          (uint32_t(FeatureSet) & FeatureSetMask) << FeatureSetShift |
          (uint32_t(IsExact) & IsExactMask) << IsExactShift) {}

  /// \return an invalid IGCATarget.
  static constexpr IGCATarget invalid() { return IGCATarget(0); }

  /// \return the packed uint32_t representation of the IGCA target.
  constexpr uint32_t pack() const { return V; }
  /// Wrap a packed uint32_t IGCA target with an IGCATarget class.
  static constexpr IGCATarget unpack(uint32_t V) { return IGCATarget(V); }

  uint16_t getTarget() const { return uint16_t(V >> TargetShift); }
  IGCAFeatureSet getFeatureSet() const {
    return IGCAFeatureSet((V >> FeatureSetShift) & FeatureSetMask);
  }
  bool isExact() const { return (V >> IsExactShift) & IsExactMask; }
  bool isValid() const { return getTarget() != 0; }
  explicit operator bool() const { return isValid(); }

  bool isCore() const { return getFeatureSet() == IGCA_CORE; }
  bool isCompute() const { return getFeatureSet() == IGCA_COMPUTE; }
  bool isRender() const { return getFeatureSet() == IGCA_RENDER; }
  bool isComputeExact() const { return isCompute() && isExact(); }
  bool isRenderExact() const { return isRender() && isExact(); }

  friend bool operator==(IGCATarget A, IGCATarget B) { return A.V == B.V; }
  friend bool operator!=(IGCATarget A, IGCATarget B) { return A.V != B.V; }
};

/// \return the device name that \p GPUIPVersion identifies, as reported by the
/// driver, e.g. "xe-pvc". If the table lists no such device, return an empty
/// string. Only the major and minor versions are looked at, so every revision
/// of a device resolves to the same name.
/// If several rows match, the first one in IntelGPUTargetParser.def wins.
LLVM_ABI StringRef getArchName(uint32_t GPUIPVersion);

/// \return the numeric name of \p GPUIPVersion, e.g. "xe_35.11.0".
LLVM_ABI std::string getNumericArchName(uint32_t GPUIPVersion);

/// Parse an IGCA target string, such as "igca_60ca". \return an invalid
/// IGCATarget if \p TargetStr is not a known target in 
/// IntelGPUTargetParser.def
LLVM_ABI IGCATarget parseIGCATarget(StringRef TargetStr);

/// \return the \p Target as a string, i.e. "igca_60ca". \return an empty string
/// if \p Target is invalid or not a known target in IntelGPUTargetParser.def.
LLVM_ABI StringRef getIGCATargetName(IGCATarget Target);

/// Append every legal IGCA target spelling to \p Values.
LLVM_ABI void fillValidIGCATargetList(SmallVectorImpl<StringRef> &Values);

} // namespace IntelGPU
} // namespace llvm

#endif // LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H
