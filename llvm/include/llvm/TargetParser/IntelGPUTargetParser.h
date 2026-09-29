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

// TODO Do I want the IGCA_ prefix here?
enum class IGCAFeatureSet : uint8_t {
  IGCA_CORE = 0,
  IGCA_COMPUTE,
  IGCA_RENDER
};

// TODO these need to be tested:
struct IGCATarget {
  uint16_t Target = 0;
  IGCAFeatureSet FeatureSet = IGCAFeatureSet::IGCA_CORE;
  bool IsExactFeatureSet = false;

  /// Return true if the IGCATarget is a valid, properly initialized target.
  bool isValid() const { return Target != 0; }
  explicit operator bool() const { return isValid(); }

  bool isCore() const { return FeatureSet == IGCAFeatureSet::IGCA_CORE; }
  bool isExact() const { return IsExactFeatureSet; }
  bool isCompute() const { return FeatureSet == IGCAFeatureSet::IGCA_COMPUTE; }
  bool isRender() const { return FeatureSet == IGCAFeatureSet::IGCA_RENDER; }
  bool isComputeExact() const { return isCompute() && isExact(); };
  bool isRenderExact() const { return isRender() && isExact(); };

  /// Pack an IGCATarget into an uint32_t.
  ///
  /// We encode as [31:16] Target, [2:1] FeatureSet, [0] IsExactFeatureSet.
  /// [15:3] is left as reserved.
  uint32_t pack() const;

  /// Obtain an IGCATarget from a packed uint32_t.
  static IGCATarget unpack(uint32_t V);

  friend bool operator==(IGCATarget A, IGCATarget B) {
    // TODO is this bad?
    return A.pack() == B.pack();
  }

  /// Return an invalid IGCA Target that returns false on IGCATarget::isValid().
  static IGCATarget invalid() { return IGCATarget{}; }
};

/// \return the device name that \p GPUIPVersion identifies, as reported by the
/// driver, e.g. "xe-pvc". If the table lists no such device, return an empty
/// string. Only the major and minor versions are looked at, so every revision
/// of a device resolves to the same name.
/// If several rows match, the first one in IntelGPUTargetParser.def wins.
LLVM_ABI StringRef getArchName(uint32_t GPUIPVersion);

/// \return the numeric name of \p GPUIPVersion, e.g. "xe_35.11.0".
LLVM_ABI std::string getNumericArchName(uint32_t GPUIPVersion);

/// Parse an IGCA Target.
IGCATarget parseIGCATarget(StringRef TargetStr);

/// Return the IGCA string for a given IGCA Target, i.e. "igca_60ca". Returns
/// empty string if Target is not a valid target.
std::string getIGCATargetName(IGCATarget Target);
// TODO to carry stringref's, I need to create that IGCA Target table
// If that IGCA Target table gets really verbose, we should just use the
// strings from that table and use StringRef here instead of std::string.
// TODO DO STRINGREF'S

// TODO
// - get igcatarget
// - translate other targets?

} // namespace IntelGPU
} // namespace llvm

#endif // LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H
