//===-- AMDGPURegBankLegalize.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Lower G_ instructions that can't be inst-selected with register bank
/// assignment from AMDGPURegBankSelect based on machine uniformity info.
/// Given types on all operands, some register bank assignments require lowering
/// while others do not.
/// Note: cases where all register bank assignments would require lowering are
/// lowered in legalizer.
/// For example vgpr S64 G_AND requires lowering to S32 while sgpr S64 does not.
/// Eliminate sgpr S1 by lowering to sgpr S32.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-regbanklegalize"

using namespace llvm;

namespace {

class AMDGPURegBankLegalize : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPURegBankLegalize() : MachineFunctionPass(ID) {
    initializeAMDGPURegBankLegalizePass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Register Bank Legalize";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // If there were no phis and we do waterfall expansion machine verifier would
  // fail.
  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoPHIs);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPURegBankLegalize, DEBUG_TYPE,
                      "AMDGPU Register Bank Legalize", false, false)
INITIALIZE_PASS_END(AMDGPURegBankLegalize, DEBUG_TYPE,
                    "AMDGPU Register Bank Legalize", false, false)

char AMDGPURegBankLegalize::ID = 0;

char &llvm::AMDGPURegBankLegalizeID = AMDGPURegBankLegalize::ID;

FunctionPass *llvm::createAMDGPURegBankLegalizePass() {
  return new AMDGPURegBankLegalize();
}

using namespace AMDGPU;

bool AMDGPURegBankLegalize::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  return true;
}
