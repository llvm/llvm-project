//===- AMDGPUBarrierLatency.h - AMDGPU Export Clustering --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUBARRIERLATENCY_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUBARRIERLATENCY_H

#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include <memory>

namespace llvm {

class MachineFunction;

std::unique_ptr<ScheduleDAGMutation>
createAMDGPUBarrierLatencyDAGMutation(MachineFunction *MF);

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUBARRIERLATENCY_H
