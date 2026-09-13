//===- AMDGPUMFMAIGroupLP.h - AMDGPU MFMA IGroupLP --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMAIGROUPLP_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMAIGROUPLP_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include <memory>

namespace llvm {

namespace AMDGPU {
// The current phase of instruction scheduling
enum class SchedulingPhase { Initial, PreRAReentry, PostRA };

/// Operand 0 immediate for IGLP_OPT pseudo instructions.
enum IGLPStrategyID : int {
  MFMASmallGemmOptID = 0,
  MFMASmallGemmSingleWaveOptID = 1,
  MFMAExpInterleaveID = 2,
  MFMAExpSimpleInterleaveID = 3,
};

// Components of the mask that determines which instruction types may be may be
// classified into a SchedGroup.
enum class SchedGroupMask {
  NONE = 0u,
  ALU = 1u << 0,
  VALU = 1u << 1,
  SALU = 1u << 2,
  MFMA = 1u << 3,
  VMEM = 1u << 4,
  VMEM_READ = 1u << 5,
  VMEM_WRITE = 1u << 6,
  DS = 1u << 7,
  DS_READ = 1u << 8,
  DS_WRITE = 1u << 9,
  TRANS = 1u << 10,
  LDSDMA = 1u << 11,
  ALL = ALU | VALU | SALU | MFMA | VMEM | VMEM_READ | VMEM_WRITE | DS |
      DS_READ | DS_WRITE | TRANS | LDSDMA,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ ALL)
};
} // namespace AMDGPU

std::unique_ptr<ScheduleDAGMutation>
createIGroupLPDAGMutation(AMDGPU::SchedulingPhase Phase);

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMAIGROUPLP_H
