//===- AMDGPUObjectLinkingHelper.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Helpers for AMDGPU object-linking resource resolution and patching.
///
/// These APIs consume link-time ABI data from .amdgpu.info plus target
/// properties from TargetParser.  They intentionally do not depend on
/// MCSubtargetInfo or backend feature strings.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AMDGPUOBJECTLINKINGHELPER_H
#define LLVM_SUPPORT_AMDGPUOBJECTLINKINGHELPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include <cstdint>

namespace llvm {

namespace msgpack {
class Document;
class MapDocNode;
} // namespace msgpack

namespace AMDGPU {

/// Raw resource usage values propagated across the call graph by the linker.
/// No ISA-specific encoding has been applied.
struct RawLinkingResources {
  uint32_t NumArchVGPR = 0;
  uint32_t NumAccVGPR = 0;
  uint32_t NumSGPR = 0;
  uint32_t ScratchSize = 0;
  uint32_t LDSSize = 0;
  uint32_t NumNamedBarrier = 0;
  bool UsesVCC = false;
  bool UsesFlatScratch = false;
  bool HasDynSizedStack = false;
};

/// Fully resolved and encoded resource values ready to write into a kernel
/// descriptor or HSA metadata.
struct ResolvedLinkingResources {
  uint32_t TotalVGPR = 0;
  uint32_t TotalSGPR = 0;
  uint32_t NumAccVGPR = 0;
  uint32_t ScratchSize = 0;
  uint32_t LDSSize = 0;
  uint32_t AccumOffset = 0;
  uint32_t EncodedNamedBarrierCount = 0;
  uint32_t SGPRBlocks = 0;
  bool ScratchEnable = false;
  bool UsesDynamicStack = false;
};

LLVM_ABI ResolvedLinkingResources resolveObjectLinkingResources(
    const ObjectLinkingTargetInfo &Target, const RawLinkingResources &Raw);

/// Patch a 64-byte kernel descriptor buffer in-place with resolved resource
/// values.
LLVM_ABI void patchKernelDescriptor(MutableArrayRef<uint8_t> KD,
                                    const ObjectLinkingTargetInfo &Target,
                                    const ResolvedLinkingResources &Resources,
                                    bool PatchLDS, bool PatchResources);

/// Update HSA metadata fields for one kernel in a msgpack document.
LLVM_ABI void patchHSAMetadataKernel(msgpack::MapDocNode &KernMap,
                                     msgpack::Document &Doc,
                                     const ResolvedLinkingResources &Resources,
                                     bool PatchLDS, bool PatchResources);

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_SUPPORT_AMDGPUOBJECTLINKINGHELPER_H
