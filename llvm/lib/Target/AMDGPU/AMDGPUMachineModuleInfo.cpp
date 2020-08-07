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
#include "llvm/Support/TargetParser.h"

namespace llvm {

AMDGPUMachineModuleInfo::AMDGPUMachineModuleInfo(const MachineModuleInfo &MMI)
    : MachineModuleInfoELF(MMI) {
  LLVMContext &CTX = MMI.getModule()->getContext();
  AgentSSID = CTX.getOrInsertSyncScopeID("agent");
  WorkgroupSSID = CTX.getOrInsertSyncScopeID("workgroup");
  WavefrontSSID = CTX.getOrInsertSyncScopeID("wavefront");
  SystemOneAddressSpaceSSID =
      CTX.getOrInsertSyncScopeID("one-as");
  AgentOneAddressSpaceSSID =
      CTX.getOrInsertSyncScopeID("agent-one-as");
  WorkgroupOneAddressSpaceSSID =
      CTX.getOrInsertSyncScopeID("workgroup-one-as");
  WavefrontOneAddressSpaceSSID =
      CTX.getOrInsertSyncScopeID("wavefront-one-as");
  SingleThreadOneAddressSpaceSSID =
      CTX.getOrInsertSyncScopeID("singlethread-one-as");

  if (MMI.getTarget().getTargetTriple().getArch() != Triple::amdgcn) {
    return;
  }

  SmallVector<Module::ModuleFlagEntry, 8> ModuleFlags;
  MMI.getModule()->getModuleFlagsMetadata(ModuleFlags);
  for (const auto &MFE : ModuleFlags) {
    if (MFE.Behavior != Module::MergeTargetID) {
      continue;
    }

    assert(MFE.Key->getString().equals("target-id"));
    TargetID = cast<MDString>(MFE.Val)->getString().str();
  }
  if (TargetID.empty()) {
    auto TargetTriple = MMI.getTarget().getTargetTriple();
    auto CPU = MMI.getTarget().getTargetCPU();
    auto Version = AMDGPU::getIsaVersion(CPU);

    raw_string_ostream ConstructedTargetIDOStr(TargetID);
    ConstructedTargetIDOStr << TargetTriple.getArchName() << '-'
                            << TargetTriple.getVendorName() << '-'
                            << TargetTriple.getOSName() << '-'
                            << TargetTriple.getEnvironmentName() << '-';
    if (Version.Major >= 9) {
      ConstructedTargetIDOStr << CPU;
    } else {
      ConstructedTargetIDOStr << "gfx" << Version.Major << Version.Minor
                              << Version.Stepping;
    }
    ConstructedTargetIDOStr.flush();
  }
}

} // end namespace llvm
