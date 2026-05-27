//===- AMDGPUWaterfallHelper.h - Waterfall intrinsic opcode mapping --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helper for mapping waterfall intrinsic IDs and value sizes to
// SI_WATERFALL_* pseudo opcodes. Used by both SelectionDAG and GlobalISel
// instruction selectors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUWATERFALLHELPER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUWATERFALLHELPER_H

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

namespace llvm::AMDGPU {

inline unsigned getWaterfallPseudoOpcode(unsigned IntrID,
                                         unsigned SizeDwords) {
  switch (IntrID) {
  case Intrinsic::amdgcn_waterfall_begin:
    switch (SizeDwords) {
    case 1: return AMDGPU::SI_WATERFALL_BEGIN_V1;
    case 2: return AMDGPU::SI_WATERFALL_BEGIN_V2;
    case 4: return AMDGPU::SI_WATERFALL_BEGIN_V4;
    case 8: return AMDGPU::SI_WATERFALL_BEGIN_V8;
    }
    break;
  case Intrinsic::amdgcn_waterfall_readfirstlane:
    switch (SizeDwords) {
    case 1: return AMDGPU::SI_WATERFALL_READFIRSTLANE_V1;
    case 2: return AMDGPU::SI_WATERFALL_READFIRSTLANE_V2;
    case 4: return AMDGPU::SI_WATERFALL_READFIRSTLANE_V4;
    case 8: return AMDGPU::SI_WATERFALL_READFIRSTLANE_V8;
    }
    break;
  case Intrinsic::amdgcn_waterfall_end:
    switch (SizeDwords) {
    case 1: return AMDGPU::SI_WATERFALL_END_V1;
    case 2: return AMDGPU::SI_WATERFALL_END_V2;
    case 4: return AMDGPU::SI_WATERFALL_END_V4;
    case 8: return AMDGPU::SI_WATERFALL_END_V8;
    }
    break;
  case Intrinsic::amdgcn_waterfall_last_use:
    switch (SizeDwords) {
    case 1: return AMDGPU::SI_WATERFALL_LAST_USE_V1;
    case 2: return AMDGPU::SI_WATERFALL_LAST_USE_V2;
    case 4: return AMDGPU::SI_WATERFALL_LAST_USE_V4;
    case 8: return AMDGPU::SI_WATERFALL_LAST_USE_V8;
    }
    break;
  case Intrinsic::amdgcn_waterfall_last_use_vgpr:
    switch (SizeDwords) {
    case 1: return AMDGPU::SI_WATERFALL_LAST_USE_V1_V;
    case 2: return AMDGPU::SI_WATERFALL_LAST_USE_V2_V;
    case 4: return AMDGPU::SI_WATERFALL_LAST_USE_V4_V;
    case 8: return AMDGPU::SI_WATERFALL_LAST_USE_V8_V;
    }
    break;
  case Intrinsic::amdgcn_waterfall_loop_end:
    return AMDGPU::SI_WATERFALL_LOOP_END;
  }
  return 0;
}

} // namespace llvm::AMDGPU

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUWATERFALLHELPER_H
