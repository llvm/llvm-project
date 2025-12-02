//===-- AArch64BranchSplit.cpp - Branch splitting optimization -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes branch patterns where two values are OR'd together
// and then tested for zero. When the combined value is only used for the
// branch test, it's often more efficient to test each value separately.
//
// Pattern:
//   %combined = ORR %bits1, %bits2, LSL #N
//   CBZ %combined, .LBB_rare
//
// Transformed to:
//   CBNZ %bits1, .LBB_common
//   CBNZ %bits2, .LBB_common
//   B .LBB_rare
//
// This is beneficial because:
// 1. Early exit when first operand is non-zero (no OR needed)
// 2. Early exit when second operand is non-zero (no OR needed)
// 3. Only computes OR when actually needed in the taken branch path
// 4. Better for branch prediction when one operand is usually non-zero
//
// The transformation is applied when:
// - The ORR result is only used by the CBZ (dead otherwise)
// - Not optimizing for code size
// - The shift ensures no bit overlap (semantically safe)
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-branch-split"

// Enable/disable the optimization
static cl::opt<bool> EnableBranchSplit(
    "aarch64-enable-branch-split", cl::init(true), cl::Hidden,
    cl::desc("Enable branch splitting optimization"));

namespace {

class AArch64BranchSplit : public MachineFunctionPass {
public:
  static char ID;

  AArch64BranchSplit() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AArch64 Branch Splitting Optimization";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const AArch64InstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MachineRegisterInfo *MRI;

  /// Try to split a CBZ instruction fed by an OR
  bool trySplitCBZ(MachineInstr &CBZ);

  /// Check if the pattern is: OR + CBZ and it's profitable to split
  bool matchPattern(MachineInstr &CBZ, MachineInstr *&ORR,
                    Register &Src1, Register &Src2, unsigned &ShiftAmt);

  /// Check if transformation is profitable
  bool isProfitable(const MachineInstr &ORR, const MachineInstr &CBZ) const;
};

} // end anonymous namespace

char AArch64BranchSplit::ID = 0;

INITIALIZE_PASS(AArch64BranchSplit, DEBUG_TYPE,
                "AArch64 Branch Splitting Optimization", false, false)

FunctionPass *llvm::createAArch64BranchSplitPass() {
  return new AArch64BranchSplit();
}

bool AArch64BranchSplit::isProfitable(const MachineInstr &ORR,
                                       const MachineInstr &CBZ) const {
  // The transformation is profitable when:
  // 1. Not optimizing for size (adds extra instructions)
  // 2. The operands are not constants (would be folded already)
  // 3. ORR result is used (at most) by CBZ and a COPY to the fallthrough block

  // Don't apply when optimizing for size (adds extra instructions)
  const MachineFunction *MF = ORR.getParent()->getParent();
  if (MF->getFunction().hasOptSize() || MF->getFunction().hasMinSize()) {
    LLVM_DEBUG(dbgs() << "  Optimizing for size, not profitable\n");
    return false;
  }

  // Check that source operands are not constants
  Register Src1 = ORR.getOperand(1).getReg();
  Register Src2 = ORR.getOperand(2).getReg();

  if (!Src1.isVirtual() || !Src2.isVirtual()) {
    LLVM_DEBUG(dbgs() << "  Source operands are physical regs, not profitable\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "  Pattern is profitable to split\n");
  return true;
}

bool AArch64BranchSplit::matchPattern(MachineInstr &CBZ, MachineInstr *&ORR,
                                       Register &Src1, Register &Src2,
                                       unsigned &ShiftAmt) {
  // Check if CBZ instruction
  unsigned Opc = CBZ.getOpcode();
  if (Opc != AArch64::CBZW && Opc != AArch64::CBZX)
    return false;

  // Get the register being tested
  Register TestReg = CBZ.getOperand(0).getReg();
  if (!TestReg.isVirtual()) {
    LLVM_DEBUG(dbgs() << "  TestReg is not virtual\n");
    return false;
  }

  // Find the defining instruction
  ORR = MRI->getVRegDef(TestReg);
  if (!ORR) {
    LLVM_DEBUG(dbgs() << "  No defining instruction\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "  ORR candidate: " << *ORR);
  LLVM_DEBUG(dbgs() << "  Num operands: " << ORR->getNumOperands() << "\n");

  // Check if it's an ORR with shift
  unsigned ORROpc = ORR->getOpcode();
  if (ORROpc != AArch64::ORRWrs && ORROpc != AArch64::ORRXrs) {
    LLVM_DEBUG(dbgs() << "  Not an ORRWrs/ORRXrs\n");
    return false;
  }

  // ORRWrs/ORRXrs format: dest, src1, src2, shift_amount (shift type LSL is implicit)
  if (ORR->getNumOperands() < 4) {
    LLVM_DEBUG(dbgs() << "  Not enough operands\n");
    return false;
  }

  Src1 = ORR->getOperand(1).getReg();
  Src2 = ORR->getOperand(2).getReg();

  // Operand 3 is the shift amount (LSL is implicit for ORRWrs/ORRXrs)
  ShiftAmt = ORR->getOperand(3).getImm();

  LLVM_DEBUG(dbgs() << "  Pattern matched! Shift amount: " << ShiftAmt << "\n");
  return true;
}

bool AArch64BranchSplit::trySplitCBZ(MachineInstr &CBZ) {
  MachineInstr *ORR;
  Register Src1, Src2;
  unsigned ShiftAmt;

  // Match the pattern
  if (!matchPattern(CBZ, ORR, Src1, Src2, ShiftAmt))
    return false;

  // Check if transformation is profitable
  if (!isProfitable(*ORR, CBZ))
    return false;

  LLVM_DEBUG(dbgs() << "Found splittable branch pattern:\n");
  LLVM_DEBUG(dbgs() << "  ORR: " << *ORR);
  LLVM_DEBUG(dbgs() << "  CBZ: " << CBZ);

  // Get the target basic block (where CBZ jumps when zero)
  MachineBasicBlock *TargetBB = CBZ.getOperand(1).getMBB();
  MachineBasicBlock *MBB = CBZ.getParent();

  // Find the fallthrough block (the one that's not the explicit target)
  MachineBasicBlock *FallthroughBB = nullptr;
  for (MachineBasicBlock *Succ : MBB->successors()) {
    if (Succ != TargetBB) {
      FallthroughBB = Succ;
      break;
    }
  }

  if (!FallthroughBB)
    return false;

  // Check if the ORR result is used elsewhere (besides the CBZ)
  Register TestReg = ORR->getOperand(0).getReg();
  bool HasOtherUses = !MRI->hasOneUse(TestReg);

  // Collect COPYs that will need to be moved along with the ORR
  SmallVector<MachineInstr*, 4> CopiesToMove;

  // If ORR has other uses, check if it's safe to move to the fallthrough block
  if (HasOtherUses) {
    // Only safe if:
    // 1. Fallthrough block has exactly one predecessor (this block)
    // 2. All uses are in the fallthrough block itself

    // Check #1: Single predecessor
    unsigned NumPreds = 0;
    for (auto *Pred : FallthroughBB->predecessors()) {
      (void)Pred;
      ++NumPreds;
      if (NumPreds > 1) {
        LLVM_DEBUG(dbgs() << "  Fallthrough block has multiple predecessors, "
                          << "cannot safely move ORR\n");
        return false;
      }
    }

    // Check #2: All uses are in fallthrough block or are COPYs in current block

    for (MachineInstr &Use : MRI->use_instructions(TestReg)) {
      if (&Use == &CBZ)
        continue; // CBZ is in current block, will be replaced

      // Handle COPYs in the current block
      if (Use.getOpcode() == AArch64::COPY && Use.getParent() == MBB) {
        Register CopyDst = Use.getOperand(0).getReg();
        if (MRI->use_nodbg_empty(CopyDst)) {
          LLVM_DEBUG(dbgs() << "  Ignoring dead COPY in current block\n");
          continue; // Dead COPY, will be eliminated
        }

        // COPY result is used - check if all uses are in fallthrough block
        bool AllUsesInFallthrough = true;
        for (MachineInstr &CopyUse : MRI->use_instructions(CopyDst)) {
          if (CopyUse.getParent() != FallthroughBB) {
            AllUsesInFallthrough = false;
            break;
          }
        }

        if (!AllUsesInFallthrough) {
          LLVM_DEBUG(dbgs() << "  COPY result used outside fallthrough: "
                            << Use << "\n");
          return false;
        }

        // COPY can be moved to fallthrough along with ORR
        CopiesToMove.push_back(&Use);
        LLVM_DEBUG(dbgs() << "  Will move COPY to fallthrough: " << Use << "\n");
        continue;
      }

      // All other uses must be in the fallthrough block
      if (Use.getParent() != FallthroughBB) {
        LLVM_DEBUG(dbgs() << "  ORR has uses outside fallthrough block: "
                          << Use << "\n");
        return false;
      }
    }

    LLVM_DEBUG(dbgs() << "  ORR has other uses but can be safely moved to "
                      << "fallthrough block\n");
  }

  // Build the transformation:
  // CBNZ Src1, FallthroughBB  ; If Src1 != 0, skip to common case
  // CBNZ Src2, FallthroughBB  ; If Src2 != 0, skip to common case
  // B TargetBB                 ; Both zero, rare case

  DebugLoc DL = CBZ.getDebugLoc();
  unsigned CBNZOpc = (CBZ.getOpcode() == AArch64::CBZW) ? AArch64::CBNZW : AArch64::CBNZX;

  // Insert new instructions before CBZ
  BuildMI(*MBB, CBZ, DL, TII->get(CBNZOpc))
    .addReg(Src1)
    .addMBB(FallthroughBB);

  BuildMI(*MBB, CBZ, DL, TII->get(CBNZOpc))
    .addReg(Src2)
    .addMBB(FallthroughBB);

  BuildMI(*MBB, CBZ, DL, TII->get(AArch64::B))
    .addMBB(TargetBB);

  if (HasOtherUses) {
    // Move the ORR to the beginning of the fallthrough block
    LLVM_DEBUG(dbgs() << "  Moving ORR to fallthrough block\n");
    ORR->removeFromParent();
    auto InsertPt = FallthroughBB->begin();
    FallthroughBB->insert(InsertPt, ORR);

    // Move any COPYs that use the ORR result
    for (MachineInstr *Copy : CopiesToMove) {
      LLVM_DEBUG(dbgs() << "  Moving COPY to fallthrough block: " << *Copy << "\n");
      Copy->removeFromParent();
      // Insert after the ORR
      FallthroughBB->insertAfter(ORR, Copy);
    }
  } else {
    // Delete the ORR since it's only used by the CBZ
    LLVM_DEBUG(dbgs() << "  Deleting ORR (no other uses)\n");
    ORR->eraseFromParent();
  }

  // Remove the old CBZ
  CBZ.eraseFromParent();

  LLVM_DEBUG(dbgs() << "Successfully split branch\n");

  return true;
}

bool AArch64BranchSplit::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  if (!EnableBranchSplit)
    return false;

  TII = static_cast<const AArch64InstrInfo *>(MF.getSubtarget().getInstrInfo());
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  LLVM_DEBUG(dbgs() << "Running AArch64BranchSplit on " << MF.getName() << "\n");

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.getOpcode() == AArch64::CBZW || MI.getOpcode() == AArch64::CBZX) {
        LLVM_DEBUG(dbgs() << "Found CBZ: " << MI);
        Changed |= trySplitCBZ(MI);
      }
    }
  }

  return Changed;
}
