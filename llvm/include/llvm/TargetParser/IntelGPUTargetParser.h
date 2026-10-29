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

} // namespace IntelGPU
} // namespace llvm

#endif // LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H
