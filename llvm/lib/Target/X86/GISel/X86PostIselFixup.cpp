//===- X86InstructionSelector.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the fixup after instruction selection for
/// X86.
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86RegisterBankInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "X86-post-isel-fixup"

using namespace llvm;
/*  Motivation:
    After DAGISEL, InstrEmitter emits Sub_Reg Node by applying constraint on
    register classes. e.g. GR32 defination with subreg_8bit adds constrainting
    register class GR32_ABCD.
    With GISEL, this fixup needs to be after ISEL, before verifier.
 */
namespace {
class X86PostIselFixup : public MachineFunctionPass {
public:
  static char ID;
  X86PostIselFixup() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "X86 Post ISel Fixup"; }

  bool runOnMachineFunction(MachineFunction &MF) override {
    LLVM_DEBUG(dbgs() << "Running X86 Post ISel Fixup on function: "
                      << MF.getName() << "\n");
    const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
    MachineRegisterInfo &MRI = MF.getRegInfo();
    bool Changed = false;
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        if (MI.getOpcode() == TargetOpcode::COPY) {
          LLVM_DEBUG(dbgs() << "Reg Constraint Fixup: "; MI.dump());
          auto CopyOpd = MI.getOperand(1);
          if (CopyOpd.getSubReg() != 0) {
            auto VReg = CopyOpd.getReg();
            const TargetRegisterClass *RC = TRI.getSubClassWithSubReg(
                MRI.getRegClass(VReg), CopyOpd.getSubReg());
            MRI.constrainRegClass(VReg, RC);
            Changed |= true;
          }
        }
      }
    }
    return Changed;
  }
};

} // end anonymous namespace

char X86PostIselFixup::ID = 0;

INITIALIZE_PASS(X86PostIselFixup, DEBUG_TYPE, "X86 Post ISel Fixup", false,
                false)

FunctionPass *llvm::createX86PostIselFixupPass() {
  return new X86PostIselFixup();
}