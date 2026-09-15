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

/// The Intel GPU architecture names this build knows, covering both physical
/// devices and the compatibility names that stand for a whole product line.
enum GPUKind : uint16_t {
  GK_NONE = 0,
#define INTEL_GPU(NAME, KIND, ARCHITECTURE, RELEASE, IGCA_LEVEL, IGCA_SUFFIX)  \
  GK_##KIND,
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_LEVEL, IGCA_SUFFIX) GK_##KIND,
#include "llvm/TargetParser/IntelGPUTargetParser.def"
};

/// The components that a GPU IP version, the "GMDID", packs into one 32-bit
/// value. The revision identifies the hardware stepping.
struct GMDID {
  unsigned Architecture = 0;
  unsigned Release = 0;
  unsigned Revision = 0;
};

/// Split the GPU IP version \p IPVersion, as reported by the driver, into its
/// components.
LLVM_ABI GMDID decodeGMDID(uint32_t IPVersion);

/// The device whose GMDID has the same architecture and release as \p ID, or
/// GK_NONE if this build knows no such device. The revision is ignored: as far
/// as the compiler is concerned, every stepping of a release is one device.
/// When several devices share an architecture and a release, the first one
/// listed in IntelGPUTargetParser.def names the group and is returned.
LLVM_ABI GPUKind getKindForGMDID(GMDID ID);

/// The human-friendly name of \p Kind, e.g. "xe-pvc", or "" for GK_NONE.
LLVM_ABI StringRef getArchName(GPUKind Kind);

/// Spell \p ID the way an architecture name spells a GMDID, e.g. "xe_35.11.0".
/// Every device has such a name, including one that is not in the table, which
/// makes this the only way to name a device this build does not know.
LLVM_ABI std::string getNumericArchName(GMDID ID);

} // namespace IntelGPU
} // namespace llvm

#endif // LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H
