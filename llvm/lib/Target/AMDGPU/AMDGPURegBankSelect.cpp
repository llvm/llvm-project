//===-- AMDGPURegBankSelect.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Assign register banks to all register operands of G_ instructions using
/// machine uniformity analysis.
/// Sgpr - uniform values and some lane masks
/// Vgpr - divergent, non S1, values
/// Vcc  - divergent S1 values(lane masks)
/// However in some cases G_ instructions with this register bank assignment
/// can't be inst-selected. This is solved in AMDGPURegBankLegalize.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-regbankselect"

using namespace llvm;

namespace {

class AMDGPURegBankSelect : public MachineFunctionPass {
public:
  static char ID;

  AMDGPURegBankSelect() : MachineFunctionPass(ID) {
    initializeAMDGPURegBankSelectPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Register Bank Select";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // This pass assigns register banks to all virtual registers, and we maintain
  // this property in subsequent passes
  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::RegBankSelected);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPURegBankSelect, DEBUG_TYPE,
                      "AMDGPU Register Bank Select", false, false)
INITIALIZE_PASS_END(AMDGPURegBankSelect, DEBUG_TYPE,
                    "AMDGPU Register Bank Select", false, false)

char AMDGPURegBankSelect::ID = 0;

char &llvm::AMDGPURegBankSelectID = AMDGPURegBankSelect::ID;

FunctionPass *llvm::createAMDGPURegBankSelectPass() {
  return new AMDGPURegBankSelect();
}

bool AMDGPURegBankSelect::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  return true;
}
