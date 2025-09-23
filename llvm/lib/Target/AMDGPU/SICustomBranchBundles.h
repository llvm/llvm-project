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

static inline MachineInstr &getBranchWithDest(MachineBasicBlock &BranchingMBB,
                                              MachineBasicBlock &DestMBB) {
  auto &TII =
      *BranchingMBB.getParent()->getSubtarget<GCNSubtarget>().getInstrInfo();
  for (MachineInstr &BranchMI : reverse(BranchingMBB.instrs()))
    if (BranchMI.isBranch() && TII.getBranchDestBlock(BranchMI) == &DestMBB)
      return BranchMI;

  llvm_unreachable("Don't call this if there's no branch to the destination.");
}

static inline void moveInsBeforePhis(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  auto &TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  auto &MRI = MF.getRegInfo();

  bool PhiSeen = false;
  MachineBasicBlock::iterator FirstPhi;
  for (FirstPhi = MBB.begin(); FirstPhi != MBB.end(); FirstPhi++)
    if (FirstPhi->getOpcode() == AMDGPU::PHI) {
      PhiSeen = true;
      break;
    }

  if (!PhiSeen) {
    MI.removeFromParent();
    MBB.insert(MBB.begin(), &MI);
  } else {
    auto Phi = BuildMI(MBB, FirstPhi, MI.getDebugLoc(), TII.get(AMDGPU::PHI),
                       MI.getOperand(0).getReg());
    for (auto *PredMBB : MBB.predecessors()) {
      Register ClonedReg = MRI.cloneVirtualRegister(MI.getOperand(0).getReg());
      MachineInstr &BranchMI = getBranchWithDest(*PredMBB, MBB);
      MachineInstr *ClonedMI = MF.CloneMachineInstr(&MI);
      ClonedMI->getOperand(0).setReg(ClonedReg);
      Phi.addReg(ClonedReg).addMBB(PredMBB);
      PredMBB->insertAfterBundle(BranchMI.getIterator(), ClonedMI);
      ClonedMI->bundleWithPred();
    }
    MI.eraseFromParent();
  }
}

struct EpilogIterator {
  MachineBasicBlock::instr_iterator InternalIt;
  EpilogIterator(MachineBasicBlock::instr_iterator I) : InternalIt(I) {}

  bool operator==(const EpilogIterator &Other) {
    return InternalIt == Other.InternalIt;
  }
  bool isEnd() { return InternalIt.isEnd(); }
  MachineInstr &operator*() { return *InternalIt; }
  MachineBasicBlock::instr_iterator operator->() { return InternalIt; }
  EpilogIterator &operator++() {
    ++InternalIt;
    if (!InternalIt.isEnd() && InternalIt->isBranch())
      InternalIt = InternalIt->getParent()->instr_end();
    return *this;
  }
  EpilogIterator operator++(int Ignored) {
    EpilogIterator ToReturn = *this;
    ++*this;
    return ToReturn;
  }
};

static inline EpilogIterator getEpilogForSuccessor(MachineBasicBlock &PredMBB,
                                                   MachineBasicBlock &SuccMBB) {
  MachineFunction &MF = *PredMBB.getParent();
  auto &TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  for (MachineInstr &BranchMI : reverse(PredMBB.instrs()))
    if (BranchMI.isBranch() && TII.getBranchDestBlock(BranchMI) == &SuccMBB)
      return ++EpilogIterator(BranchMI.getIterator());

  llvm_unreachable("There should always be a branch to succ_MBB.");
}

static inline bool epilogsAreIdentical(const vector<MachineInstr *> Left,
                                       const vector<MachineInstr *> Right,
                                       const MachineBasicBlock &SuccMBB) {
  if (Left.size() != Right.size())
    return false;

  for (unsigned I = 0; I < Left.size(); I++)
    if (!Left[I]->isIdenticalTo(*Right[I]))
      return false;
  return true;
}

static inline void moveBody(vector<MachineInstr *> &Body,
                            MachineBasicBlock &DestMBB) {
  for (auto RevIt = Body.rbegin(); RevIt != Body.rend(); RevIt++) {
    MachineInstr &BodyIns = **RevIt;
    BodyIns.removeFromBundle();
    DestMBB.insert(DestMBB.begin(), &BodyIns);
  }
}

static inline void normalizeIrPostPhiElimination(MachineFunction &MF) {
  auto &TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  struct CFGRewriteEntry {
    unordered_set<MachineBasicBlock *> PredMBBs;
    MachineBasicBlock *SuccMBB;
    vector<MachineInstr *> Body;
  };

  vector<CFGRewriteEntry> CfgRewriteEntries;
  for (MachineBasicBlock &MBB : MF) {
    CFGRewriteEntry ToInsert = {{}, &MBB, {}};
    for (MachineBasicBlock *PredMBB : MBB.predecessors()) {
      EpilogIterator EpIt = getEpilogForSuccessor(*PredMBB, MBB);

      vector<MachineInstr *> Epilog;
      while (!EpIt.isEnd())
        Epilog.push_back(&*EpIt++);

      if (!epilogsAreIdentical(ToInsert.Body, Epilog, MBB)) {
        if (ToInsert.PredMBBs.size() && ToInsert.Body.size()) {
          // Potentially, we need to insert a new entry.  But first see if we
          // can find an existing entry with the same epilog.
          bool ExistingEntryFound = false;
          for (auto RevIt = CfgRewriteEntries.rbegin();
               RevIt != CfgRewriteEntries.rend() && RevIt->SuccMBB == &MBB;
               RevIt++)
            if (epilogsAreIdentical(RevIt->Body, Epilog, MBB)) {
              RevIt->PredMBBs.insert(PredMBB);
              ExistingEntryFound = true;
              break;
            }

          if (!ExistingEntryFound)
            CfgRewriteEntries.push_back(ToInsert);
        }
        ToInsert.PredMBBs.clear();
        ToInsert.Body = Epilog;
      }

      ToInsert.PredMBBs.insert(PredMBB);
    }

    // Handle the last potential rewrite entry.  Lower instead of journaling a
    // rewrite entry if all predecessor MBBs are in this single entry.
    if (ToInsert.PredMBBs.size() == MBB.pred_size()) {
      moveBody(ToInsert.Body, MBB);
      for (MachineBasicBlock *PredMBB : ToInsert.PredMBBs) {
        // Delete instructions that were lowered from epilog
        MachineInstr &BranchIns =
            getBranchWithDest(*PredMBB, *ToInsert.SuccMBB);
        auto EpilogIt = ++EpilogIterator(BranchIns.getIterator());
        while (!EpilogIt.isEnd())
          EpilogIt++->eraseFromBundle();
      }

    } else if (ToInsert.Body.size())
      CfgRewriteEntries.push_back(ToInsert);
  }

  // Perform the journaled rewrites.
  for (auto &Entry : CfgRewriteEntries) {
    MachineBasicBlock *MezzanineMBB = MF.CreateMachineBasicBlock();
    MF.insert(MF.end(), MezzanineMBB);

    // Deal with mezzanine to successor succession.
    BuildMI(MezzanineMBB, DebugLoc(), TII.get(AMDGPU::S_BRANCH))
        .addMBB(Entry.SuccMBB);
    MezzanineMBB->addSuccessor(Entry.SuccMBB);

    // Move instructions to mezzanine block.
    moveBody(Entry.Body, *MezzanineMBB);

    for (MachineBasicBlock *PredMBB : Entry.PredMBBs) {
      // Deal with predecessor to mezzanine succession.
      MachineInstr &BranchIns = getBranchWithDest(*PredMBB, *Entry.SuccMBB);
      assert(BranchIns.getOperand(0).isMBB() && "Branch instruction isn't.");
      BranchIns.getOperand(0).setMBB(MezzanineMBB);
      PredMBB->replaceSuccessor(Entry.SuccMBB, MezzanineMBB);

      // Delete instructions that were lowered from epilog
      auto EpilogIt = ++EpilogIterator(BranchIns.getIterator());
      while (!EpilogIt.isEnd())
        EpilogIt++->eraseFromBundle();
    }
  }
}

namespace std {
template <> struct hash<Register> {
  std::size_t operator()(const Register &R) const {
    return hash<unsigned>()(R);
  }
};
} // namespace std

static inline void hoistUnrelatedCopies(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &BranchMI : MBB) {
      if (!BranchMI.isBranch())
        continue;

      unordered_set<Register> RelatedCopySources;
      EpilogIterator EpilogIt = BranchMI.getIterator();
      EpilogIterator CopyMoveIt = ++EpilogIt;
      while (!EpilogIt.isEnd()) {
        if (EpilogIt->getOpcode() != AMDGPU::COPY)
          RelatedCopySources.insert(EpilogIt->getOperand(0).getReg());
        ++EpilogIt;
      }

      while (!CopyMoveIt.isEnd()) {
        EpilogIterator Next = CopyMoveIt;
        ++Next;
        if (CopyMoveIt->getOpcode() == AMDGPU::COPY &&
                !RelatedCopySources.count(CopyMoveIt->getOperand(1).getReg()) ||
            CopyMoveIt->getOpcode() == AMDGPU::IMPLICIT_DEF) {
          MachineInstr &MIToMove = *CopyMoveIt;
          MIToMove.removeFromBundle();
          MBB.insert(BranchMI.getIterator(), &MIToMove);
        }

        CopyMoveIt = Next;
      }
    }
}
