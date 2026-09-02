//===-- NVPTXTargetParser.h - Parser for NVPTX target ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_NVPTXTARGETPARSER_H
#define LLVM_TARGETPARSER_NVPTXTARGETPARSER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>

namespace llvm {
namespace NVPTX {

/// GPU kinds supported by the NVPTX target.
enum GPUKind : uint8_t {
  GK_NONE = 0,
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  GK_##KIND,
#include "llvm/TargetParser/NVPTXTargetParser.def"

  // Alias for the last GPUKind. Keep in sync with the final .def row.
  // FIXME: Should be generated once the GPU list moves to TableGen.
  GK_LAST = GK_SM_121f,
};

/// Suffix class of an NVPTX architecture name. Enumerator spellings match the
/// SUFFIX column tokens in NVPTXTargetParser.def.
enum class ArchSuffix { NONE, ACCELERATED, FAMILY };

/// Parse \p CPU (e.g. "sm_90") into a GPUKind, or GK_NONE if unrecognized.
LLVM_ABI GPUKind parseArch(StringRef CPU);

/// Return the canonical processor name (e.g. "sm_90") for \p Kind, or "" if
/// \p Kind is GK_NONE.
LLVM_ABI StringRef getArchName(GPUKind Kind);

/// Return the virtual (compute_) arch name (e.g. "compute_90") for \p Kind, or
/// "" if \p Kind is GK_NONE.
LLVM_ABI StringRef getVirtualArch(GPUKind Kind);

/// Return the numeric compute-capability id (e.g. sm_90 -> 900) for \p Kind, or
/// 0 if \p Kind is GK_NONE.
LLVM_ABI unsigned getSmVersion(GPUKind Kind);

/// Return the suffix class of \p Kind.
LLVM_ABI ArchSuffix getArchSuffix(GPUKind Kind);

/// Whether \p Kind is an accelerated variant (e.g. sm_90a).
inline bool isAcceleratedArch(GPUKind Kind) {
  return getArchSuffix(Kind) == ArchSuffix::ACCELERATED;
}

/// Whether \p Kind is a family-specific variant (e.g. sm_90f) or accelerated.
inline bool isFamilySpecificArch(GPUKind Kind) {
  ArchSuffix S = getArchSuffix(Kind);
  return S == ArchSuffix::FAMILY || S == ArchSuffix::ACCELERATED;
}

/// Whether \p Kind supports unified addressing. Unified addressing was
/// introduced with the Pascal generation (sm_60).
inline bool supportsUnifiedAddressing(GPUKind Kind) {
  return getSmVersion(Kind) >= 600;
}

} // namespace NVPTX
} // namespace llvm

#endif // LLVM_TARGETPARSER_NVPTXTARGETPARSER_H
