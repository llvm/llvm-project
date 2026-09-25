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
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/AtomicScope.h"

using namespace llvm;

AMDGPUMachineModuleInfo::AMDGPUMachineModuleInfo(const MachineModuleInfo &MMI)
    : MachineModuleInfoELF(MMI) {
  LLVMContext &CTX = MMI.getModule()->getContext();
  const Triple &TT = MMI.getTarget().getTargetTriple();

  auto InsertScope = [&](AtomicScope Scope, bool OneAS) {
    return CTX.getOrInsertSyncScopeID(
        *getAtomicScopeIRString(TT, Scope, OneAS));
  };
  AgentSSID = InsertScope(AtomicScope::Device, /*OneAS=*/false);
  WorkgroupSSID = InsertScope(AtomicScope::Workgroup, /*OneAS=*/false);
  WavefrontSSID = InsertScope(AtomicScope::Wavefront, /*OneAS=*/false);
  ClusterSSID = InsertScope(AtomicScope::Cluster, /*OneAS=*/false);
  SystemOneAddressSpaceSSID = InsertScope(AtomicScope::System, /*OneAS=*/true);
  AgentOneAddressSpaceSSID = InsertScope(AtomicScope::Device, /*OneAS=*/true);
  WorkgroupOneAddressSpaceSSID =
      InsertScope(AtomicScope::Workgroup, /*OneAS=*/true);
  WavefrontOneAddressSpaceSSID =
      InsertScope(AtomicScope::Wavefront, /*OneAS=*/true);
  SingleThreadOneAddressSpaceSSID =
      InsertScope(AtomicScope::Single, /*OneAS=*/true);
  ClusterOneAddressSpaceSSID =
      InsertScope(AtomicScope::Cluster, /*OneAS=*/true);
}
