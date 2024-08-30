//=======--- GCNDPPCombine.h - optimization for DPP instructions ---==========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNDPPCOMBINE_H
#define LLVM_LIB_TARGET_AMDGPU_GCNDPPCOMBINE_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {
class GCNDPPCombinePass : public PassInfoMixin<GCNDPPCombinePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MAM);

  MachineFunctionProperties getRequiredProperties() {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNDPPCOMBINE_H
