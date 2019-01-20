//===- AMDGPUMacroFusion.h - AMDGPU Macro Fusion ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

/// Note that you have to add:
///   DAG.addMutation(createAMDGPUMacroFusionDAGMutation());
/// to AMDGPUPassConfig::createMachineScheduler() to have an effect.
std::unique_ptr<ScheduleDAGMutation> createAMDGPUMacroFusionDAGMutation();

} // llvm
