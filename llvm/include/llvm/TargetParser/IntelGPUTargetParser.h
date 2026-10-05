//===-- IntelGPUTargetParser.h - Parser for Intel GPU targets ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides access to the Intel GPU list in IntelGPUTargetParser.def.
// It answers the two questions the compiler asks of the list: what to call the
// device a driver reports, and what to compile for when the user names one.
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

/// \return the device name that \p GPUIPVersion identifies, as reported by the
/// driver, e.g. "xe-pvc". If the table lists no such device, return an empty
/// string. Only the major and minor versions are looked at, so every revision
/// of a device resolves to the same name.
/// If several rows match, the first one in IntelGPUTargetParser.def wins.
LLVM_ABI StringRef getArchName(uint32_t GPUIPVersion);

/// \return the numeric name of \p GPUIPVersion, e.g. "xe_35.11.0".
LLVM_ABI std::string getNumericArchName(uint32_t GPUIPVersion);

/// \return the human-friendly name of \p Kind, e.g. "xe-pvc", or an empty
/// string for GK_NONE.
LLVM_ABI StringRef getArchName(GPUKind Kind);

/// \return the device \p Name denotes, or GK_NONE for a name this build does
/// not know.
///
/// Every spelling the user may write is accepted: a human-friendly name such as
/// "xe-pvc", a compatibility name such as "xe-dg2", an alias such as "bmg_g21",
/// and a numeric name such as "xe_12.60.7". Only the human-friendly name is
/// reported back for a device, so a spelling that is not one canonicalizes to
/// the one that is. The revision of a numeric name takes no part in the lookup,
/// just as it takes none in getArchName, so "xe_12.60.0" and "xe_12.60.7" name
/// the same device; the revision may also be omitted. A numeric name for a
/// device that is not in the table yields GK_NONE like any other unknown name.
LLVM_ABI GPUKind parseArch(StringRef Name);

/// \return true if \p Name is a well-formed numeric name, e.g. "xe_40.11.0",
/// whether or not the table knows the device. The offload-arch utility prints
/// one for a device that has no row yet, so that a newer device is usable with
/// a compiler that predates it.
LLVM_ABI bool isNumericArchName(StringRef Name);

/// \return the IGCA target name to compile \p Kind for, e.g. "xe-pvc" ->
/// "igca_20ca", or an empty string for GK_NONE. This is the spelling
/// -target-cpu is invoked with.
///
/// The name is the numeric IGCA target followed by the suffix of the feature
/// sets: igca_60 is the core features, igca_60c adds the compute features, and
/// igca_60ca is exact, meaning that only a device at that target will do.
LLVM_ABI StringRef getIGCAName(GPUKind Kind);

/// Append every architecture name this build accepts, for diagnostics that
/// offer the user an alternative to a name that did not parse.
LLVM_ABI void fillValidArchList(SmallVectorImpl<StringRef> &Values);

} // namespace IntelGPU
} // namespace llvm

#endif // LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H
