//===-- AVRBranchSelector.cpp - Emit long conditional branches ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that scans a machine function to determine which
// conditional branches need more than 8 bits of displacement to reach their
// target basic block.  It does this in two passes; a calculation of basic block
// positions pass, and a branch pseudo op to machine branch opcode pass.  This
// pass should be run last, just before the assembly printer.
//
//===----------------------------------------------------------------------===//

#include "AVR.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

#include "AVRInstrInfo.h"
#include "AVRTargetMachine.h"
#include "MCTargetDesc/AVRMCTargetDesc.h"

#define DEBUG_TYPE "avr-branch-select"

STATISTIC(NumExpanded, "Number of branches expanded to long format");

namespace llvm {

/// Ensures branch targets can fit inside the instruction
/// they reside in.
///
/// If a branch target is too large for the instruction it
/// is being used with, this pass replaces it with a larger,
/// equivalent instruction which can fit the target.
class AVRBSel : public MachineFunctionPass {
public:
  static char ID;

  explicit AVRBSel() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AVR Branch Selector"; }

protected:

  /// Measure and sum the size of all basic blocks in a function.
  unsigned calculateFunctionSize(const MachineFunction &MF);

private:
  /// The sizes of the basic blocks in the function.
  std::vector<unsigned> BlockSizes;
};

char AVRBSel::ID = 0;

/// Checks whether the passed opcode is a conditional branch.
static bool isConditionalBranch(int Opcode) {
  switch (Opcode) {
  default:
    return false;
  case AVR::BREQk:
  case AVR::BRNEk:
  case AVR::BRSHk:
  case AVR::BRLOk:
  case AVR::BRMIk:
  case AVR::BRPLk:
  case AVR::BRGEk:
  case AVR::BRLTk:
    return true;
  }
}

unsigned AVRBSel::calculateFunctionSize(const MachineFunction &MF) {
  const AVRSubtarget &STI = MF.getSubtarget<AVRSubtarget>();
  const AVRInstrInfo &TII = *STI.getInstrInfo();

  unsigned FuncSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    unsigned BlockSize = 0;

    for (const MachineInstr &MI : MBB) {
      BlockSize += TII.getInstSizeInBytes(MI);
    }

    BlockSizes[MBB.getNumber()] = BlockSize;
    FuncSize += BlockSize;
  }

  assert(FuncSize % 2 == 0 && "function should have an even number of bytes");

  return FuncSize;
}

bool AVRBSel::runOnMachineFunction(MachineFunction &MF) {
  const AVRSubtarget &STI = MF.getSubtarget<AVRSubtarget>();
  const AVRInstrInfo &TII = *STI.getInstrInfo();

  // Give the blocks of the function a dense, in-order, numbering.
  MF.RenumberBlocks();
  BlockSizes.resize(MF.getNumBlockIDs());

  unsigned FuncSize = calculateFunctionSize(MF);

  // If the entire function is smaller than the displacement of a branch field,
  // we know we don't need to shrink any branches in this function.  This is a
  // common case.
  if (isUInt<7>(FuncSize)) {
    BlockSizes.clear();
    return false;
  }

  // For each conditional branch, if the offset to its destination is larger
  // than the offset field allows, transform it into a long or a huge branch
  // sequence like this:
  //  -short branch:
  //     brCC MBB
  //  -long branch:
  //     br!CC $PC+2
  //     rjmp MBB
  //  -huge branch:
  //     br!CC $PC+4
  //     jmp MBB
  bool MadeChange = true;
  while (MadeChange) {
    // Iteratively expand branches until we reach a fixed point.
    MadeChange = false;

    for (MachineBasicBlock &MBB : MF) {
      unsigned MBBStartOffset = 0;

      for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;
           ++I) {
        int Opc = I->getOpcode();
        if ((!isConditionalBranch(Opc) || I->getOperand(0).isImm()) &&
            (Opc != AVR::RJMPk)) {
          MBBStartOffset += TII.getInstSizeInBytes(*I);
          continue;
        }

        // Determine the offset from the current branch to the destination
        // block.
        MachineBasicBlock &Dest = *I->getOperand(0).getMBB();

        assert(Dest.getNumber() >= 0 &&
            "Destination basic block isn't in a function");

        int BranchSize;
        if (Dest.getNumber() <= MBB.getNumber()) {
          // If this is a backwards branch, the delta is the offset from the
          // start of this block to this branch, plus the sizes of all blocks
          // from this block to the dest.
          BranchSize = MBBStartOffset;

          for (unsigned i = Dest.getNumber(), e = MBB.getNumber(); i != e;
               ++i) {
            BranchSize += BlockSizes[i];
          }

          // Set the size of backwards branches to a negative value.
          BranchSize = -BranchSize;
        } else {
          // Otherwise, add the size of the blocks between this block and the
          // dest to the number of bytes left in this block.
          BranchSize = -MBBStartOffset;

          for (unsigned i = MBB.getNumber(), e = Dest.getNumber(); i != e;
               ++i) {
            BranchSize += BlockSizes[i];
          }
        }

        if (isConditionalBranch(Opc))
          BranchSize -= 2; // take the size of the current instruction.

        assert(BranchSize % 2 == 0 &&
               "BranchSize should have an even number of bytes");

        // If this branch is in range, ignore it.
        if ((isConditionalBranch(Opc) && isInt<8>(BranchSize)) ||
            (Opc == AVR::RJMPk && isInt<13>(BranchSize))) {
          MBBStartOffset += 2;
          continue;
        }

        // Otherwise, we have to expand it to a long branch.
        unsigned NewSize;
        int UncondOpc;
        MachineInstr &OldBranch = *I;
        DebugLoc DL = OldBranch.getDebugLoc();

        if (Opc == AVR::RJMPk) {
          // Replace this instruction with a jmp which has a size of 4 bytes.
          NewSize = 4;
          UncondOpc = AVR::JMPk;

          // We may be converting a conditional long jump to a huge one, if this
          // is the case, update the $PC+2 operand in brCC to $PC+4.
          // Skip the check when this instruction is the first inside the BB.
          if (I != MBB.begin()) {
            MachineInstr &PI = *std::prev(I);

            if (isConditionalBranch(PI.getOpcode()) &&
                PI.getOperand(0).isImm() &&
                PI.getOperand(0).getImm() == 2) {
              PI.getOperand(0).setImm(4);
            }
          }
        } else {
          assert(isConditionalBranch(Opc) &&
              "opcode should be a conditional branch");

          unsigned BrCCOffs;
          // Determine if we can reach the destination block with a rjmp,
          // otherwise a jmp instruction is needed.
          if (isInt<13>(BranchSize)) {
            NewSize = 4;
            BrCCOffs = 2;
            UncondOpc = AVR::RJMPk;
          } else {
            NewSize = 6;
            BrCCOffs = 4;
            UncondOpc = AVR::JMPk;
          }

          AVRCC::CondCodes OCC =
              TII.getOppositeCondition(TII.getCondFromBranchOpc(Opc));
          // Jump over the uncond branch inst (i.e. $+2) on opposite condition.
          BuildMI(MBB, I, DL, TII.getBrCond(OCC)).addImm(BrCCOffs);
        }

        // Uncond branch to the real destination.
        I = BuildMI(MBB, I, DL, TII.get(UncondOpc)).addMBB(&Dest);

        // Remove the old branch from the function.
        OldBranch.eraseFromParent();

        // Remember that this instruction is NewSize bytes, increase the size of
        // the block by NewSize-2, remember to iterate.
        BlockSizes[MBB.getNumber()] += NewSize - 2;
        MBBStartOffset += NewSize;

        ++NumExpanded;
        MadeChange = true;
      }
    }
  }

  BlockSizes.clear();
  return true;
}

FunctionPass *createAVRBranchSelectionPass() { return new AVRBSel(); }

} // end of namespace llvm

