//===--------- BreakCriticalMachineEdge.h - Break critical edges ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// Helper function to break a critical machine edge.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BREAKCRITICALMACHINEEDGE_H
#define LLVM_CODEGEN_BREAKCRITICALMACHINEEDGE_H

#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

MachineBasicBlock* SplitCriticalMachineEdge(MachineBasicBlock* src,
                                            MachineBasicBlock* dst) {
  MachineFunction &MF = *src->getParent();
  const BasicBlock* srcBB = src->getBasicBlock();

  MachineBasicBlock* crit_mbb = MF.CreateMachineBasicBlock(srcBB);

  // modify the llvm control flow graph
  src->removeSuccessor(dst);
  src->addSuccessor(crit_mbb);
  crit_mbb->addSuccessor(dst);

  // insert the new block into the machine function.
  MF.push_back(crit_mbb);

  // insert a unconditional branch linking the new block to dst
  const TargetMachine& TM = MF.getTarget();
  const TargetInstrInfo* TII = TM.getInstrInfo();
  std::vector<MachineOperand> emptyConditions;
  TII->InsertBranch(*crit_mbb, dst, (MachineBasicBlock*)0, 
                    emptyConditions);

  // modify every branch in src that points to dst to point to the new
  // machine basic block instead:
  MachineBasicBlock::iterator mii = src->end();
  bool found_branch = false;
  while (mii != src->begin()) {
    mii--;
    // if there are no more branches, finish the loop
    if (!mii->getDesc().isTerminator()) {
      break;
    }

    // Scan the operands of this branch, replacing any uses of dst with
    // crit_mbb.
    for (unsigned i = 0, e = mii->getNumOperands(); i != e; ++i) {
      MachineOperand & mo = mii->getOperand(i);
      if (mo.isMBB() && mo.getMBB() == dst) {
        found_branch = true;
        mo.setMBB(crit_mbb);
      }
    }
  }

  // TODO: This is tentative. It may be necessary to fix this code. Maybe
  // I am inserting too many gotos, but I am trusting that the asm printer
  // will optimize the unnecessary gotos.
  if(!found_branch) {
    TII->InsertBranch(*src, crit_mbb, (MachineBasicBlock*)0, 
                      emptyConditions);
  }

  /// Change all the phi functions in dst, so that the incoming block be
  /// crit_mbb, instead of src
  for(mii = dst->begin(); mii != dst->end(); mii++) {
    /// the first instructions are always phi functions.
    if(mii->getOpcode() != TargetInstrInfo::PHI)
      break;

    // Find the operands corresponding to the source block
    std::vector<unsigned> toRemove;
    unsigned reg = 0;
    for (unsigned u = 0; u != mii->getNumOperands(); ++u)
      if (mii->getOperand(u).isMBB() &&
          mii->getOperand(u).getMBB() == src) {
        reg = mii->getOperand(u-1).getReg();
        toRemove.push_back(u-1);
      }
    // Remove all uses of this MBB
    for (std::vector<unsigned>::reverse_iterator I = toRemove.rbegin(),
         E = toRemove.rend(); I != E; ++I) {
      mii->RemoveOperand(*I+1);
      mii->RemoveOperand(*I);
    }

    // Add a single use corresponding to the new MBB
    mii->addOperand(MachineOperand::CreateReg(reg, false));
    mii->addOperand(MachineOperand::CreateMBB(crit_mbb));
  }

  return crit_mbb;
}

}

#endif
