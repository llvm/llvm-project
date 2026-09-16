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
enum GPUKind : uint16_t {
  GK_NONE = 0,
#define INTEL_GPU(NAME, KIND, ARCHITECTURE, RELEASE, IGCA_TARGET, IGCA_SUFFIX) \
  GK_##KIND,
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_TARGET, IGCA_SUFFIX) GK_##KIND,
#include "llvm/TargetParser/IntelGPUTargetParser.def"
};

/// The components that a GPU IP version, the "GMDID", packs into one 32-bit
/// value. The revision identifies the hardware stepping.
struct GMDID {
  unsigned Architecture = 0;
  unsigned Release = 0;
  unsigned Revision = 0;
};

/// Split the \p GPUIPVersion, as reported by the driver, into its components.
LLVM_ABI GMDID decodeGMDID(uint32_t GPUIPVersion);

/// Return the kind matching the architecture and release of \p ID, or GK_NONE
/// if the table lists no such device. The revision is ignored: every stepping
/// of a release is one device. If several rows match, the first one in
/// IntelGPUTargetParser.def wins.
LLVM_ABI GPUKind getKindForGMDID(GMDID ID);

/// Return the human-friendly name of \p Kind, e.g. "xe-pvc", or "" for GK_NONE.
LLVM_ABI StringRef getArchName(GPUKind Kind);

/// Return the numeric name of \p ID, e.g. "xe_35.11.0", which every device has,
/// even one that the table does not list.
LLVM_ABI std::string getNumericArchName(GMDID ID);

} // namespace IntelGPU
} // namespace llvm

#endif // LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H
