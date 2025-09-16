#pragma once

#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/ErrorHandling.h"

#include "SIInstrInfo.h"

#include <cassert>
#include <unordered_set>

using namespace llvm;

using std::unordered_set;
using std::vector;

static inline MachineInstr &get_branch_with_dest(MachineBasicBlock &branching_MBB,
                                                 MachineBasicBlock &dest_MBB) {
  auto& TII = *branching_MBB.getParent()->getSubtarget<GCNSubtarget>().getInstrInfo();
  for (MachineInstr &branch_MI : reverse(branching_MBB.instrs()))
    if (branch_MI.isBranch() && TII.getBranchDestBlock(branch_MI) == &dest_MBB)
      return branch_MI;

  llvm_unreachable("Don't call this if there's no branch to the destination.");
}

static inline void move_ins_before_phis(MachineInstr &MI) {
  MachineBasicBlock& MBB = *MI.getParent();
  MachineFunction& MF = *MBB.getParent();
  auto& TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  bool phi_seen = false;
  for (MachineInstr &maybe_phi : MBB)
    if (maybe_phi.getOpcode() == AMDGPU::PHI) {
      phi_seen = true;
      break;
    }
  
  if (!phi_seen) {
    MI.removeFromParent();
    MBB.insert(MBB.begin(), &MI);
  } else {
    for (auto* pred_MBB : MBB.predecessors())
    {
      MachineInstr& branch_MI = get_branch_with_dest(*pred_MBB,MBB);
      if (branch_MI.isBranch() && TII.getBranchDestBlock(branch_MI) == &MBB) {
        MachineInstr* cloned_MI = MF.CloneMachineInstr(&MI);
        pred_MBB->insertAfterBundle(branch_MI.getIterator(), cloned_MI);
        cloned_MI->bundleWithPred();
      }
    }
    MI.eraseFromParent();
  }
}

static inline MachineBasicBlock::instr_iterator
get_epilog_for_successor(MachineBasicBlock& pred_MBB, MachineBasicBlock& succ_MBB) {
  MachineFunction& MF = *pred_MBB.getParent();
  auto& TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  for (MachineInstr &branch_MI : reverse(pred_MBB.instrs()))
    if (branch_MI.isBranch() && TII.getBranchDestBlock(branch_MI) == &succ_MBB)
      return std::next(branch_MI.getIterator());
    else
      assert(branch_MI.isBranch() && "Shouldn't have fall-throughs here.");

  llvm_unreachable("There should always be a branch to succ_MBB.");
}

static inline void normalize_ir_post_phi_elimination(MachineFunction &MF) {
  auto& TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  struct CFG_Rewrite_Entry {
    unordered_set<MachineBasicBlock *> pred_MBBs;
    MachineBasicBlock *succ_MBB;
    vector<MachineInstr*> body;
  };

  auto epilogs_are_identical = [](const vector<MachineInstr *> left,
                                  const vector<MachineInstr *> right) {
    if (left.size() != right.size())
      return false;

    for (unsigned i = 0; i < left.size(); i++)
      if (!left[i]->isIdenticalTo(*right[i]))
        return false;
    return true;
  };

  auto move_body = [](vector<MachineInstr *> &body,
                      MachineBasicBlock &dest_MBB) {
    for (auto rev_it = body.rbegin(); rev_it != body.rend(); rev_it++) {
      MachineInstr &body_ins = **rev_it;
      body_ins.removeFromBundle();
      dest_MBB.insert(dest_MBB.begin(), &body_ins);
    }
  };
  
  vector<CFG_Rewrite_Entry> cfg_rewrite_entries;
  for (MachineBasicBlock &MBB : MF) {
    CFG_Rewrite_Entry to_insert = {{}, &MBB, {}};
    for (MachineBasicBlock *pred_MBB : MBB.predecessors()) {
      MachineBasicBlock::instr_iterator ep_it =
          get_epilog_for_successor(*pred_MBB, MBB)->getIterator();

      vector<MachineInstr *> epilog;
      while (!ep_it.isEnd())
        epilog.push_back(&*ep_it++);

      if (!epilogs_are_identical(to_insert.body, epilog)) {
        if (to_insert.pred_MBBs.size() && to_insert.body.size()) {
          // Potentially, we need to insert a new entry.  But first see if we
          // can find an existing entry with the same epilog.
          bool existing_entry_found = false;
          for (auto rev_it = cfg_rewrite_entries.rbegin();
               rev_it != cfg_rewrite_entries.rend() && rev_it->succ_MBB == &MBB;
               rev_it++)
            if (epilogs_are_identical(rev_it->body, epilog)) {
              rev_it->pred_MBBs.insert(pred_MBB);
              existing_entry_found = true;
              break;
            }

          if(!existing_entry_found)
            cfg_rewrite_entries.push_back(to_insert);
        }
        to_insert.pred_MBBs.clear();
        to_insert.body = epilog;
      }
      
      to_insert.pred_MBBs.insert(pred_MBB);
    }

    // Handle the last potential rewrite entry.  Lower instead of journaling a
    // rewrite entry if all predecessor MBBs are in this single entry.
    if (to_insert.pred_MBBs.size() == MBB.pred_size())
      // Lower
      move_body(to_insert.body,MBB);
    else if (to_insert.body.size())
      cfg_rewrite_entries.push_back(to_insert);
  }

  // Perform the journaled rewrites.
  for (auto &entry : cfg_rewrite_entries) {
    MachineBasicBlock *mezzanine_MBB = MF.CreateMachineBasicBlock();

    // Deal with mezzanine to successor succession.
    BuildMI(mezzanine_MBB, DebugLoc(), TII.get(AMDGPU::S_BRANCH)).addMBB(entry.succ_MBB);
    mezzanine_MBB->addSuccessor(entry.succ_MBB);

    // Move instructions to mezzanine block.
    move_body(entry.body, *mezzanine_MBB);

    for (MachineBasicBlock *pred_MBB : entry.pred_MBBs) {
      //Deal with predecessor to mezzanine succession.
      MachineInstr &branch_ins =
          get_branch_with_dest(*pred_MBB, *entry.succ_MBB);
      assert(branch_ins.getOperand(0).isMBB() && "Branch instruction isn't.");
      branch_ins.getOperand(0).setMBB(mezzanine_MBB);
      pred_MBB->replaceSuccessor(entry.succ_MBB, mezzanine_MBB);

      // Delete instructions that were lowered from epilog
      auto epilog_it = get_epilog_for_successor(*pred_MBB, *entry.succ_MBB);
      while (!epilog_it.isEnd())
        epilog_it++->eraseFromBundle();
    }
  }
}
