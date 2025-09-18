#pragma once

#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
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
  auto &TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  auto& MRI = MF.getRegInfo();

  bool phi_seen = false;
  MachineBasicBlock::iterator first_phi;
  for (first_phi = MBB.begin(); first_phi != MBB.end(); first_phi++)
    if (first_phi->getOpcode() == AMDGPU::PHI) {
      phi_seen = true;
      break;
    }
  
  if (!phi_seen) {
    MI.removeFromParent();
    MBB.insert(MBB.begin(), &MI);
  } else {
    auto phi = BuildMI(MBB, first_phi, MI.getDebugLoc(), TII.get(AMDGPU::PHI),
            MI.getOperand(0).getReg());
    for (auto *pred_MBB : MBB.predecessors()) {
      Register cloned_reg = MRI.cloneVirtualRegister(MI.getOperand(0).getReg());
      MachineInstr& branch_MI = get_branch_with_dest(*pred_MBB,MBB);
      MachineInstr *cloned_MI = MF.CloneMachineInstr(&MI);
      cloned_MI->getOperand(0).setReg(cloned_reg);
      phi.addReg(cloned_reg).addMBB(pred_MBB);
      pred_MBB->insertAfterBundle(branch_MI.getIterator(), cloned_MI);
      cloned_MI->bundleWithPred();
    }
    MI.eraseFromParent();
  }
}

struct Epilog_Iterator {
  MachineBasicBlock::instr_iterator internal_it;
  Epilog_Iterator(MachineBasicBlock::instr_iterator i) : internal_it(i) {}

  bool operator==(const Epilog_Iterator &other) {
    return internal_it == other.internal_it;
  }
  bool isEnd() { return internal_it.isEnd(); }
  MachineInstr &operator*() { return *internal_it; }
  MachineBasicBlock::instr_iterator operator->() { return internal_it; }
  Epilog_Iterator &operator++() {
    ++internal_it;
    if (!internal_it.isEnd() && internal_it->isBranch())
      internal_it = internal_it->getParent()->instr_end();
    return *this;
  }
  Epilog_Iterator operator++(int ignored) {
    Epilog_Iterator to_return = *this;
    ++*this;
    return to_return;
  }
  
};

static inline Epilog_Iterator
get_epilog_for_successor(MachineBasicBlock& pred_MBB, MachineBasicBlock& succ_MBB) {
  MachineFunction& MF = *pred_MBB.getParent();
  auto& TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  for (MachineInstr &branch_MI : reverse(pred_MBB.instrs()))
    if (branch_MI.isBranch() && TII.getBranchDestBlock(branch_MI) == &succ_MBB)
      return ++Epilog_Iterator(branch_MI.getIterator());

  llvm_unreachable("There should always be a branch to succ_MBB.");
}

static inline bool epilogs_are_identical(const vector<MachineInstr *> left,
                                         const vector<MachineInstr *> right,
                                         const MachineBasicBlock &succ_MBB) {
  if (left.size() != right.size())
    return false;
  
  for (unsigned i = 0; i < left.size(); i++)
    if (!left[i]->isIdenticalTo(*right[i]))
      return false;
  return true;
}

static inline void move_body(vector<MachineInstr *> &body,
                      MachineBasicBlock &dest_MBB) {
  for (auto rev_it = body.rbegin(); rev_it != body.rend(); rev_it++) {
    MachineInstr &body_ins = **rev_it;
    body_ins.removeFromBundle();
    dest_MBB.insert(dest_MBB.begin(), &body_ins);
  }
}

static inline void normalize_ir_post_phi_elimination(MachineFunction &MF) {
  auto& TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  struct CFG_Rewrite_Entry {
    unordered_set<MachineBasicBlock *> pred_MBBs;
    MachineBasicBlock *succ_MBB;
    vector<MachineInstr*> body;
  };

  vector<CFG_Rewrite_Entry> cfg_rewrite_entries;
  for (MachineBasicBlock &MBB : MF) {
    CFG_Rewrite_Entry to_insert = {{}, &MBB, {}};
    for (MachineBasicBlock *pred_MBB : MBB.predecessors()) {
      Epilog_Iterator ep_it =
          get_epilog_for_successor(*pred_MBB, MBB);

      vector<MachineInstr *> epilog;
      while (!ep_it.isEnd())
        epilog.push_back(&*ep_it++);

      if (!epilogs_are_identical(to_insert.body, epilog, MBB)) {
        if (to_insert.pred_MBBs.size() && to_insert.body.size()) {
          // Potentially, we need to insert a new entry.  But first see if we
          // can find an existing entry with the same epilog.
          bool existing_entry_found = false;
          for (auto rev_it = cfg_rewrite_entries.rbegin();
               rev_it != cfg_rewrite_entries.rend() && rev_it->succ_MBB == &MBB;
               rev_it++)
            if (epilogs_are_identical(rev_it->body, epilog, MBB)) {
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
    if (to_insert.pred_MBBs.size() == MBB.pred_size()) {
      move_body(to_insert.body, MBB);
      for (MachineBasicBlock *pred_MBB : to_insert.pred_MBBs) {
        // Delete instructions that were lowered from epilog
        MachineInstr &branch_ins =
          get_branch_with_dest(*pred_MBB, *to_insert.succ_MBB);
        auto epilog_it = ++Epilog_Iterator(branch_ins.getIterator());
        while (!epilog_it.isEnd())
          epilog_it++->eraseFromBundle();
      }

    }
    else if (to_insert.body.size())
      cfg_rewrite_entries.push_back(to_insert);
  }

  // Perform the journaled rewrites.
  for (auto &entry : cfg_rewrite_entries) {
    MachineBasicBlock *mezzanine_MBB = MF.CreateMachineBasicBlock();
    MF.insert(MF.end(),mezzanine_MBB);

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
      auto epilog_it = ++Epilog_Iterator(branch_ins.getIterator());
      while (!epilog_it.isEnd())
        epilog_it++->eraseFromBundle();
    }
  }
}

namespace std {
  template <>
  struct hash<Register>
  {
    std::size_t operator()(const Register& r) const
    {
         return hash<unsigned>()(r);
    }
  };
}

static inline void hoist_unrelated_copies(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &branch_MI : MBB) {
      if (!branch_MI.isBranch())
        continue;

      unordered_set<Register> related_copy_sources;
      Epilog_Iterator epilog_it = branch_MI.getIterator();
      Epilog_Iterator copy_move_it = ++epilog_it;
      while (!epilog_it.isEnd()) {
        if (epilog_it->getOpcode() != AMDGPU::COPY)
          related_copy_sources.insert(epilog_it->getOperand(0).getReg());
        ++epilog_it;
      }

      while (!copy_move_it.isEnd()) {
        Epilog_Iterator next = copy_move_it; ++next;
        if (copy_move_it->getOpcode() == AMDGPU::COPY &&
            !related_copy_sources.count(copy_move_it->getOperand(1).getReg())
            || copy_move_it->getOpcode() == AMDGPU::IMPLICIT_DEF) {
          MachineInstr &MI_to_move = *copy_move_it;
          MI_to_move.removeFromBundle();
          MBB.insert(branch_MI.getIterator(),&MI_to_move);
        }
        
        copy_move_it = next;
      }
    }
}
