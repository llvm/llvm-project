//===-- PPCBranchSelector.cpp - Emit long conditional branches ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that scans a machine function to determine which
// conditional branches need more than 16 bits of displacement to reach their
// target basic block.  It does this in two passes; a calculation of basic block
// positions pass, and a branch pseudo op to machine branch opcode pass.  This
// pass should be run last, just before the assembly printer.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ppc-branch-select"
#include "PPC.h"
#include "PPCInstrBuilder.h"
#include "PPCInstrInfo.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

STATISTIC(NumExpanded, "Number of branches expanded to long format");

namespace {
  struct PPCBSel : public MachineFunctionPass {
    static char ID;
    PPCBSel() : MachineFunctionPass(ID) {}

    /// BlockSizes - The sizes of the basic blocks in the function.
    std::vector<unsigned> BlockSizes;

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "PowerPC Branch Selector";
    }
  };
  char PPCBSel::ID = 0;
}

/// createPPCBranchSelectionPass - returns an instance of the Branch Selection
/// Pass
///
FunctionPass *llvm::createPPCBranchSelectionPass() {
  return new PPCBSel();
}

bool PPCBSel::runOnMachineFunction(MachineFunction &Fn) {
  const PPCInstrInfo *TII =
                static_cast<const PPCInstrInfo*>(Fn.getTarget().getInstrInfo());
  // Give the blocks of the function a dense, in-order, numbering.
  Fn.RenumberBlocks();
  BlockSizes.resize(Fn.getNumBlockIDs());

  // Measure each MBB and compute a size for the entire function.
  unsigned FuncSize = 0;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock *MBB = MFI;

    unsigned BlockSize = 0;
    for (MachineBasicBlock::iterator MBBI = MBB->begin(), EE = MBB->end();
         MBBI != EE; ++MBBI)
      BlockSize += TII->GetInstSizeInBytes(MBBI);
    
    BlockSizes[MBB->getNumber()] = BlockSize;
    FuncSize += BlockSize;
  }
  
  // If the entire function is smaller than the displacement of a branch field,
  // we know we don't need to shrink any branches in this function.  This is a
  // common case.
  if (FuncSize < (1 << 15)) {
    BlockSizes.clear();
    return false;
  }
  
  // For each conditional branch, if the offset to its destination is larger
  // than the offset field allows, transform it into a long branch sequence
  // like this:
  //   short branch:
  //     bCC MBB
  //   long branch:
  //     b!CC $PC+8
  //     b MBB
  //
  bool MadeChange = true;
  bool EverMadeChange = false;
  while (MadeChange) {
    // Iteratively expand branches until we reach a fixed point.
    MadeChange = false;
  
    for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
         ++MFI) {
      MachineBasicBlock &MBB = *MFI;
      unsigned MBBStartOffset = 0;
      for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
           I != E; ++I) {
        if (I->getOpcode() != PPC::BCC || I->getOperand(2).isImm()) {
          MBBStartOffset += TII->GetInstSizeInBytes(I);
          continue;
        }
        
        // Determine the offset from the current branch to the destination
        // block.
        MachineBasicBlock *Dest = I->getOperand(2).getMBB();
        
        int BranchSize;
        if (Dest->getNumber() <= MBB.getNumber()) {
          // If this is a backwards branch, the delta is the offset from the
          // start of this block to this branch, plus the sizes of all blocks
          // from this block to the dest.
          BranchSize = MBBStartOffset;
          
          for (unsigned i = Dest->getNumber(), e = MBB.getNumber(); i != e; ++i)
            BranchSize += BlockSizes[i];
        } else {
          // Otherwise, add the size of the blocks between this block and the
          // dest to the number of bytes left in this block.
          BranchSize = -MBBStartOffset;

          for (unsigned i = MBB.getNumber(), e = Dest->getNumber(); i != e; ++i)
            BranchSize += BlockSizes[i];
        }

        // If this branch is in range, ignore it.
        if (isInt<16>(BranchSize)) {
          MBBStartOffset += 4;
          continue;
        }

        // Otherwise, we have to expand it to a long branch.
        MachineInstr *OldBranch = I;
        DebugLoc dl = OldBranch->getDebugLoc();
 
        if (I->getOpcode() == PPC::BCC) {
          // The BCC operands are:
          // 0. PPC branch predicate
          // 1. CR register
          // 2. Target MBB
          PPC::Predicate Pred = (PPC::Predicate)I->getOperand(0).getImm();
          unsigned CRReg = I->getOperand(1).getReg();
       
          // Jump over the uncond branch inst (i.e. $PC+8) on opposite condition.
          BuildMI(MBB, I, dl, TII->get(PPC::BCC))
            .addImm(PPC::InvertPredicate(Pred)).addReg(CRReg).addImm(2);
        } else if (I->getOpcode() == PPC::BDNZ) {
          BuildMI(MBB, I, dl, TII->get(PPC::BDZ)).addImm(2);
        } else if (I->getOpcode() == PPC::BDNZ8) {
          BuildMI(MBB, I, dl, TII->get(PPC::BDZ8)).addImm(2);
        } else if (I->getOpcode() == PPC::BDZ) {
          BuildMI(MBB, I, dl, TII->get(PPC::BDNZ)).addImm(2);
        } else if (I->getOpcode() == PPC::BDZ8) {
          BuildMI(MBB, I, dl, TII->get(PPC::BDNZ8)).addImm(2);
        } else {
           llvm_unreachable("Unhandled branch type!");
        }
        
        // Uncond branch to the real destination.
        I = BuildMI(MBB, I, dl, TII->get(PPC::B)).addMBB(Dest);

        // Remove the old branch from the function.
        OldBranch->eraseFromParent();
        
        // Remember that this instruction is 8-bytes, increase the size of the
        // block by 4, remember to iterate.
        BlockSizes[MBB.getNumber()] += 4;
        MBBStartOffset += 8;
        ++NumExpanded;
        MadeChange = true;
      }
    }
    EverMadeChange |= MadeChange;
  }
  
  BlockSizes.clear();
  return true;
}

