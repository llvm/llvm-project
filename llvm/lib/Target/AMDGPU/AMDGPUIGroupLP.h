//===- AMDGPUMFMAIGroupLP.h - AMDGPU MFMA IGroupLP --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMAIGROUPLP_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMAIGROUPLP_H

#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include <memory>

namespace llvm {

// Components of the mask that determines which instruction types may be may be
// classified into a SchedGroup.
enum class IGLPPhase {
  Initial = 0u,
  PreRAReentry = 1u << 0,
  PostRA = 1u << 1
};

std::unique_ptr<ScheduleDAGMutation> createIGroupLPDAGMutation(IGLPPhase Phase);

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMAIGROUPLP_H
