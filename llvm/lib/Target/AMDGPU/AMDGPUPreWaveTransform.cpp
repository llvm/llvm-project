//===------- AMDGPUPreWaveTransform.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass just schedules MachineUniformityAnalysisPass at the moment.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-wave-transform"

namespace {

class AMDGPUPreWaveTransform : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPUPreWaveTransform() : MachineFunctionPass(ID) {
    initializeAMDGPUPreWaveTransformPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AMDGPU Pre Wave Transform"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineUniformityAnalysisPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  MachineUniformityInfo *UniformInfo = nullptr;
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUPreWaveTransform, DEBUG_TYPE,
                      "AMDGPU Pre Wave Transform", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineUniformityAnalysisPass)
INITIALIZE_PASS_END(AMDGPUPreWaveTransform, DEBUG_TYPE,
                    "AMDGPU Pre Wave Transform", false, false)

char AMDGPUPreWaveTransform::ID = 0;
char &llvm::AMDGPUPreWaveTransformID = AMDGPUPreWaveTransform::ID;

FunctionPass *llvm::createAMDGPUPreWaveTransformPass() {
  return new AMDGPUPreWaveTransform();
}

/// \brief Run the AMDGPU Pre Wave Transform.
bool AMDGPUPreWaveTransform::runOnMachineFunction(MachineFunction &MF) {
  UniformInfo =
      &getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();
  LLVM_DEBUG(UniformInfo->print(dbgs()));
  return false;
}
