//===--- AMDGPUMachineModuleInfo.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDGPU Machine Module Info.
///
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMachineModuleInfo.h"
#include "llvm/IR/Module.h"

namespace llvm {

AMDGPUMachineModuleInfo::AMDGPUMachineModuleInfo(const MachineModuleInfo &MMI)
    : MachineModuleInfoELF(MMI) {
  LLVMContext &CTX = MMI.getModule()->getContext();
  AgentSSID = CTX.getOrInsertSyncScopeID("agent");
  WorkgroupSSID = CTX.getOrInsertSyncScopeID("workgroup");
  WavefrontSSID = CTX.getOrInsertSyncScopeID("wavefront");
}

} // end namespace llvm
