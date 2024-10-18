//===-- AMDGPURBSelect.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Assign register banks to all register operands of G_ instructions using
/// machine uniformity analysis.
/// SGPR - uniform values and some lane masks
/// VGPR - divergent, non S1, values
/// VCC  - divergent S1 values(lane masks)
/// However in some cases G_ instructions with this register bank assignment
/// can't be inst-selected. This is solved in RBLegalize.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "rb-select"

using namespace llvm;

namespace {

class AMDGPURBSelect : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPURBSelect() : MachineFunctionPass(ID) {
    initializeAMDGPURBSelectPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AMDGPU RB select"; }

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

INITIALIZE_PASS_BEGIN(AMDGPURBSelect, DEBUG_TYPE, "AMDGPU RB select", false,
                      false)
INITIALIZE_PASS_END(AMDGPURBSelect, DEBUG_TYPE, "AMDGPU RB select", false,
                    false)

char AMDGPURBSelect::ID = 0;

char &llvm::AMDGPURBSelectID = AMDGPURBSelect::ID;

FunctionPass *llvm::createAMDGPURBSelectPass() { return new AMDGPURBSelect(); }

bool AMDGPURBSelect::runOnMachineFunction(MachineFunction &MF) { return true; }
