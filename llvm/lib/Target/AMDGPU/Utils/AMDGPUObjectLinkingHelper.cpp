//===- AMDGPUObjectLinkingHelper.cpp - AMDGPU link patch helpers ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/AMDGPUObjectLinkingHelper.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

using namespace llvm;
using namespace llvm::amdhsa;
using namespace llvm::support::endian;

AMDGPU::ResolvedLinkingResources
AMDGPU::resolveObjectLinkingResources(const ObjectLinkingTargetInfo &Target,
                                      const RawLinkingResources &Raw) {
  ResolvedLinkingResources Res;
  Res.TotalVGPR = Target.hasAccVGPRs() && Raw.NumAccVGPR
                      ? alignTo(Raw.NumArchVGPR, 4) + Raw.NumAccVGPR
                      : std::max(Raw.NumArchVGPR, Raw.NumAccVGPR);
  Res.TotalSGPR =
      Raw.NumSGPR + Target.getNumExtraSGPRs(Raw.UsesVCC, Raw.UsesFlatScratch);
  Res.NumAccVGPR = Raw.NumAccVGPR;
  Res.ScratchSize = Raw.ScratchSize;
  Res.LDSSize = Raw.LDSSize;
  if (Target.hasAccVGPRs())
    Res.AccumOffset = divideCeil(std::max(Raw.NumArchVGPR, 1u), 4u) - 1;
  if (Target.hasNamedBarrier())
    Res.EncodedNamedBarrierCount = divideCeil(Raw.NumNamedBarrier, 4);
  Res.SGPRBlocks =
      Target.isGFX10Plus()
          ? 0
          : ObjectLinkingTargetInfo::getNumSGPRBlocks(Res.TotalSGPR);
  Res.ScratchEnable = Raw.ScratchSize > 0 || Raw.HasDynSizedStack;
  Res.UsesDynamicStack = Raw.HasDynSizedStack;
  return Res;
}

void AMDGPU::patchKernelDescriptor(MutableArrayRef<uint8_t> KD,
                                   const ObjectLinkingTargetInfo &Target,
                                   const ResolvedLinkingResources &Resources,
                                   bool PatchLDS, bool PatchResources) {
  assert(KD.size() >= sizeof(kernel_descriptor_t) &&
         "kernel descriptor buffer too small");

  uint8_t *Buf = KD.data();

  if (PatchLDS)
    write32le(Buf + GROUP_SEGMENT_FIXED_SIZE_OFFSET, Resources.LDSSize);

  if (!PatchResources)
    return;

  write32le(Buf + PRIVATE_SEGMENT_FIXED_SIZE_OFFSET, Resources.ScratchSize);

  uint16_t KCP = read16le(Buf + KERNEL_CODE_PROPERTIES_OFFSET);
  AMDHSA_BITS_SET(KCP, KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK,
                  Resources.UsesDynamicStack ? 1 : 0);
  write16le(Buf + KERNEL_CODE_PROPERTIES_OFFSET, KCP);

  bool EnableWavefrontSize32 =
      AMDHSA_BITS_GET(KCP, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
  unsigned WaveSize = EnableWavefrontSize32 ? 32 : 64;
  unsigned VGPRBlocks =
      Target.getEncodedNumVGPRBlocks(Resources.TotalVGPR, WaveSize);

  uint32_t Rsrc1 = read32le(Buf + COMPUTE_PGM_RSRC1_OFFSET);
  AMDHSA_BITS_SET(Rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  VGPRBlocks);
  AMDHSA_BITS_SET(Rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  Resources.SGPRBlocks);
  write32le(Buf + COMPUTE_PGM_RSRC1_OFFSET, Rsrc1);

  uint32_t Rsrc2 = read32le(Buf + COMPUTE_PGM_RSRC2_OFFSET);
  AMDHSA_BITS_SET(Rsrc2, COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT,
                  Resources.ScratchEnable ? 1 : 0);
  write32le(Buf + COMPUTE_PGM_RSRC2_OFFSET, Rsrc2);

  if (Resources.AccumOffset != 0 || Resources.EncodedNamedBarrierCount != 0) {
    uint32_t Rsrc3 = read32le(Buf + COMPUTE_PGM_RSRC3_OFFSET);
    if (Resources.AccumOffset != 0) {
      assert(Target.hasAccVGPRs() &&
             "nonzero AccumOffset requires AccVGPR support");
      AMDHSA_BITS_SET(Rsrc3, COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                      Resources.AccumOffset);
    }
    if (Resources.EncodedNamedBarrierCount != 0) {
      assert(Target.hasNamedBarrier() &&
             "nonzero named-barrier count requires GFX12.5 support");
      AMDHSA_BITS_SET(Rsrc3, COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT,
                      Resources.EncodedNamedBarrierCount);
    }
    write32le(Buf + COMPUTE_PGM_RSRC3_OFFSET, Rsrc3);
  }
}

void AMDGPU::patchHSAMetadataKernel(msgpack::MapDocNode &KernMap,
                                    msgpack::Document &Doc,
                                    const ResolvedLinkingResources &Resources,
                                    bool PatchLDS, bool PatchResources) {
  if (PatchLDS)
    KernMap[".group_segment_fixed_size"] = Doc.getNode(Resources.LDSSize);

  if (!PatchResources)
    return;

  KernMap[".sgpr_count"] = Doc.getNode(Resources.TotalSGPR);
  KernMap[".vgpr_count"] = Doc.getNode(Resources.TotalVGPR);
  KernMap[".agpr_count"] = Doc.getNode(Resources.NumAccVGPR);
  KernMap[".private_segment_fixed_size"] = Doc.getNode(Resources.ScratchSize);
  KernMap[".uses_dynamic_stack"] = Doc.getNode(Resources.UsesDynamicStack);
}
