//===------ BPFAssignSpillsStackID.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFTargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-assign-spills-stack-id"

namespace {

struct BPFAssignSpillsStackID : public MachineFunctionPass {

  static char ID;
  MachineFunction *MF;
  const TargetRegisterInfo *TRI;

  BPFAssignSpillsStackID() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    MachineFrameInfo &MFI = MF.getFrameInfo();
    for (unsigned I = 0, E = MFI.getObjectIndexEnd(); I != E; ++I)
      // R11 is used as a base register for objects with non-default StackID.
      if (MFI.isSpillSlotObjectIndex(I))
        MFI.setStackID(I, 1);
    return false;
  }
};

} // namespace

INITIALIZE_PASS(BPFAssignSpillsStackID, "bpf-assign-spills-stack-id",
                "BPF Assign Spills StackID", false, false)

char BPFAssignSpillsStackID::ID = 0;
FunctionPass *llvm::createBPFAssignSpillsStackIDPass() {
  return new BPFAssignSpillsStackID();
}
