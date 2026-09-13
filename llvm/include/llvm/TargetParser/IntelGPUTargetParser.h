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

/// The Intel GPU architecture names this build knows, covering both physical
/// devices and the compatibility names that stand for a whole product line.
///
/// The underlying type is fixed because clang/Basic/OffloadArch.h forward
/// declares this enumeration; the two declarations have to agree.
enum GPUKind : uint8_t {
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

/// The device \p Name denotes, or GK_NONE for a name this build does not know.
///
/// Every spelling the user may write is accepted: a human-friendly name such as
/// "xe-pvc", a compatibility name such as "xe-dg2", an alias such as "bmg_g21",
/// and a numeric name such as "xe_12.60.7". Only the human-friendly name is
/// reported back for a device, so a spelling that is not one canonicalizes to
/// the one that is. The revision of a numeric name takes no part in the lookup,
/// since the table is keyed on the architecture and the release alone, so
/// "xe_12.60.0" and "xe_12.60.7" name the same device; the revision may also be
/// omitted. A numeric name for a device that is not in the table is not a name
/// this build knows, and yields GK_NONE like any other unknown name.
LLVM_ABI GPUKind parseArch(StringRef Name);

/// The IGCA level name to compile \p Kind for, e.g. "xe-pvc" -> "igca_20ca",
/// or "" for GK_NONE. This is the spelling -target-cpu is invoked with.
///
/// A level names a set of features rather than a device, and its suffix says
/// which sets it comprises: igca_60 is the core features, igca_60c adds the
/// compute features, and igca_60ca is exact, meaning that only a device at that
/// level will do.
LLVM_ABI StringRef getIGCAName(GPUKind Kind);

/// Append every architecture name this build accepts, for diagnostics that
/// offer the user an alternative to a name that did not parse.
LLVM_ABI void fillValidArchList(SmallVectorImpl<StringRef> &Values);

} // namespace IntelGPU
} // namespace llvm

#endif // LLVM_TARGETPARSER_INTELGPUTARGETPARSER_H
