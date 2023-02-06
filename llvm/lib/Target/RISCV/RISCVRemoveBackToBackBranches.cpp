//===----------------------- RISCVRemoveBackToBackBranches.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-remove-back-to-back-branches"

STATISTIC(NumInsertedAligments, "Number of aligments set");

namespace {

// According to the MIPS specification, there shouldn't be two conditional
// branches in the same 8-byte aligned region of code.
constexpr unsigned NumberOfBytesOfCodeRegion = 8;

class RISCVRemoveBackToBackBranches : public MachineFunctionPass {
public:
  static char ID;

  RISCVRemoveBackToBackBranches() : MachineFunctionPass(ID) {
    initializeRISCVRemoveBackToBackBranchesPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "RISCV Remove Back To Back Branches Pass";
  }

  bool runOnMachineFunction(MachineFunction &F) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  const RISCVSubtarget *STI;
  const RISCVInstrInfo *TII;
};

} // end of anonymous namespace

char RISCVRemoveBackToBackBranches::ID = 0;

INITIALIZE_PASS(RISCVRemoveBackToBackBranches, DEBUG_TYPE,
                "Fix hazards by removing back to back branches", false, false)

/// Returns a pass that clears pipeline hazards.
FunctionPass *llvm::createRISCVRemoveBackToBackBranches() {
  return new RISCVRemoveBackToBackBranches();
}

static bool CheckCompressedISA(MachineBasicBlock *MBB,
                               const RISCVInstrInfo *TII) {
  unsigned SizeInBytes = 0;
  for (auto &I : *MBB) {
    // Skip some 0-sized meta instrucitons, such as debug ones.
    if (!TII->getInstSizeInBytes(I))
      continue;

    SizeInBytes += TII->getInstSizeInBytes(I);

    // This means that there is something other than the conditional branch
    // here.
    if (!I.isConditionalBranch())
      continue;

    // If it is a conditional branch, make sure it is the last one
    // in this MBB and the cumulative size in bytes of other instructions in the
    // block is <= 6 (since there potentially could be space for the two
    // branches in the same 8-byte aligned code region, when compressed version
    // of the instructions (16-bit size) is being used).
    if (&I == &*MBB->getLastNonDebugInstr()) {
      if (SizeInBytes <= 6)
        return true;
      return false;
    }
  }

  return false;
}

static bool CheckNonCompressedISA(MachineBasicBlock *MBB,
                                  const RISCVInstrInfo *TII) {
  for (auto &I : *MBB) {
    // Skip some 0-sized meta instrucitons, such as debug ones.
    if (!TII->getInstSizeInBytes(I))
      continue;

    // This means that there is something other than the conditional branch
    // here.
    if (!I.isConditionalBranch())
      return false;

    // If it is a conditional branch, make sure it is the last one
    // in this MBB.
    if (&I == &*MBB->getLastNonDebugInstr())
      return true;
    return false;
  }
  return false;
}

bool RISCVRemoveBackToBackBranches::runOnMachineFunction(MachineFunction &MF) {
  STI = &static_cast<const RISCVSubtarget &>(MF.getSubtarget());
  TII = static_cast<const RISCVInstrInfo *>(STI->getInstrInfo());

  if (!STI->shouldRemoveBackToBackBranches()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Ignoring RISCV Remove Back To Back Branches Pass\n");
    return false;
  }

  bool Changed = false;
  for (auto &MBB : MF) {
    auto BBTerminator = MBB.getFirstTerminator();
    // If it is not a conditional branch, we are not interested.
    if (BBTerminator == MBB.end() ||
        &*BBTerminator != &*MBB.getLastNonDebugInstr() ||
        !BBTerminator->isConditionalBranch())
      continue;

    for (auto &Successor : MBB.successors()) {
      // Set up aligment in order to avoid hazards. No 2 conditional branches
      // should be in the same 8-byte aligned region of code. Similar to MIPS
      // forbidden slots problem. We may want to insert a NOP only, but we
      // need to think of Compressed ISA, so it is more safe to just set up
      // aligment to the successor block if it meets requirements.
      bool ShouldSetAligment = STI->getFeatureBits()[RISCV::FeatureStdExtC]
                                   ? CheckCompressedISA(Successor, TII)
                                   : CheckNonCompressedISA(Successor, TII);
      if (ShouldSetAligment) {
        Successor->setAlignment(Align(NumberOfBytesOfCodeRegion));
        Changed = true;
        ++NumInsertedAligments;
      }
    }
  }

  return Changed;
}
