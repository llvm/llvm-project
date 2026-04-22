//===-- SparseLiveVariables.cpp - Sparse Live Variable Analysis -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SparseLiveVariables analysis pass.
//
// This pass computes block-level live-in and live-out sets using a memory
// efficient SparseBitVector representation. Unlike the legacy LiveVariables
// pass, this analysis is completely stateless at the instruction level. It
// relies on a fixed-point dataflow iteration over the control flow graph to
// establish block boundary conditions. Instruction-level queries are evaluated
// dynamically via the LivenessTracker by traversing backwards from the end
// of a block.
//
// This modern, target-independent pass is designed to handle very large
// virtual register sets without the massive memory overhead traditionally
// associated with dense bit-vectors and heavily cached liveness states.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SparseLiveVariables.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "sparse-live-variables"

char SparseLiveVariablesWrapperPass::ID = 0;
INITIALIZE_PASS(SparseLiveVariablesWrapperPass, DEBUG_TYPE,
                "Sparse Live Variable Analysis", false, false)

AnalysisKey SparseLiveVariablesAnalysis::Key;

SparseLiveVariables
SparseLiveVariablesAnalysis::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &) {
  SparseLiveVariables LV;
  LV.analyze(MF);
  return LV;
}

void SparseLiveVariables::analyze(MachineFunction &MF) {
  if (MF.empty())
    return;

  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget().getRegisterInfo();

  BlockLiveness.assign(MF.getNumBlockIDs(), BlockInfo());

  SmallPtrSet<MachineBasicBlock *, 16> Reachable;
  for (MachineBasicBlock *MBB : llvm::depth_first(&MF))
    Reachable.insert(MBB);

  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (MachineBasicBlock *MBB : llvm::post_order(&MF)) {
      if (!Reachable.count(MBB))
        continue;

      SparseBitVector<> OldLiveIn = BlockLiveness[MBB->getNumber()].LiveIn;
      SparseBitVector<> OldLiveOut = BlockLiveness[MBB->getNumber()].LiveOut;

      SparseBitVector<> LiveOut;
      for (const MachineBasicBlock *Succ : MBB->successors()) {
        if (!Reachable.count(Succ))
          continue;
        LiveOut |= BlockLiveness[Succ->getNumber()].LiveIn;

        // PHI nodes have special liveness semantics: their uses are evaluated
        // conditionally on the incoming edge, meaning they are NOT Live-In to
        // the block containing the PHI node. Therefore, we cannot rely solely
        // on unioning the successor's LiveIn set. We must actively scan the
        // successor's PHI nodes and add any registers explicitly tied to the
        // incoming edge from this MBB directly into our LiveOut set.
        for (const MachineInstr &MI : *Succ) {
          if (!MI.isPHI())
            break;
          for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
            if (MI.getOperand(i).isReg() &&
                MI.getOperand(i + 1).getMBB() == MBB) {
              Register Reg = MI.getOperand(i).getReg();
              if (Reg.isValid() && Reg.isVirtual())
                LiveOut.set(Reg.id());
            }
          }
        }
      }
      BlockLiveness[MBB->getNumber()].LiveOut = LiveOut;

      SparseBitVector<> LiveIn = BlockLiveness[MBB->getNumber()].LiveOut;
      LivenessTracker Tracker(LiveIn, MRI);

      for (MachineInstr &MI : llvm::reverse(*MBB))
        Tracker.stepBackward(MI);

      BlockLiveness[MBB->getNumber()].LiveIn = Tracker.getLiveSet();

      if (BlockLiveness[MBB->getNumber()].LiveIn != OldLiveIn ||
          BlockLiveness[MBB->getNumber()].LiveOut != OldLiveOut)
        Changed = true;
    }
  }
}

void SparseLiveVariables::verifyLiveness(const MachineFunction &MF) const {
  for (const MachineBasicBlock &MBB : MF) {
    if (!hasAnalyzed(&MBB))
      continue;

    const SparseBitVector<> &LiveIn = BlockLiveness[MBB.getNumber()].LiveIn;
    for (const auto &LI : MBB.liveins()) {
      if (!LiveIn.test(LI.PhysReg.id())) {
        LLVM_DEBUG(dbgs() << "Warning: Live-in register "
                          << printReg(LI.PhysReg, TRI)
                          << " missing from computed live-in set of block "
                          << printMBBReference(MBB) << "\n");
      }
    }
  }
}
void SparseLiveVariables::updateLiveIns(MachineFunction &MF) const {
  for (MachineBasicBlock &MBB : MF) {
    if (!hasAnalyzed(&MBB))
      continue;

    MBB.clearLiveIns();
    const SparseBitVector<> &LiveIn = BlockLiveness[MBB.getNumber()].LiveIn;
    for (unsigned RegID : LiveIn) {
      Register Reg(RegID);
      // MBB.addLiveIn only takes physical registers.
      if (Reg.isPhysical())
        MBB.addLiveIn(Reg);
    }
    MBB.sortUniqueLiveIns();
  }
}

void SparseLiveVariables::recomputeRegisterLiveness(Register Reg,
                                                    MachineInstr *IgnoreMI) {
  if (!Reg.isVirtual())
    return;

  // 1. Clear Reg from all blocks
  for (unsigned i = 0, e = BlockLiveness.size(); i != e; ++i) {
    BlockLiveness[i].LiveIn.reset(Reg.id());
    BlockLiveness[i].LiveOut.reset(Reg.id());
  }

  // 2. Propagate from all uses
  SmallVector<MachineBasicBlock *, 8> WorkList;

  for (MachineInstr &UseMI : MRI->use_instructions(Reg)) {
    if (&UseMI == IgnoreMI)
      continue;

    MachineBasicBlock *MBB = UseMI.getParent();

    if (UseMI.isPHI()) {
      // For PHI nodes, the register is live-out of the corresponding
      // predecessor
      for (unsigned i = 1; i < UseMI.getNumOperands(); i += 2) {
        if (UseMI.getOperand(i).isReg() &&
            UseMI.getOperand(i).getReg() == Reg) {
          MachineBasicBlock *Pred = UseMI.getOperand(i + 1).getMBB();
          if (hasAnalyzed(Pred)) {
            if (!BlockLiveness[Pred->getNumber()].LiveOut.test(Reg.id())) {
              BlockLiveness[Pred->getNumber()].LiveOut.set(Reg.id());
              WorkList.push_back(Pred);
            }
          }
        }
      }
      continue;
    }

    bool FoundDef = false;
    for (auto I = MachineBasicBlock::reverse_iterator(&UseMI), E = MBB->rend();
         I != E; ++I) {
      if (&*I == IgnoreMI)
        continue;
      if (I->definesRegister(Reg, TRI)) {
        FoundDef = true;
        break;
      }
    }

    if (!FoundDef) {
      if (hasAnalyzed(MBB)) {
        if (!BlockLiveness[MBB->getNumber()].LiveIn.test(Reg.id())) {
          BlockLiveness[MBB->getNumber()].LiveIn.set(Reg.id());
          WorkList.push_back(MBB);
        }
      }
    }
  }

  // 3. Propagate backwards
  while (!WorkList.empty()) {
    MachineBasicBlock *Curr = WorkList.pop_back_val();

    for (MachineBasicBlock *Pred : Curr->predecessors()) {
      if (!hasAnalyzed(Pred))
        continue;

      if (!BlockLiveness[Pred->getNumber()].LiveOut.test(Reg.id())) {
        BlockLiveness[Pred->getNumber()].LiveOut.set(Reg.id());

        bool FoundDef = false;
        for (const MachineInstr &MI : llvm::reverse(*Pred)) {
          if (&MI == IgnoreMI)
            continue;
          if (MI.definesRegister(Reg, TRI)) {
            FoundDef = true;
            break;
          }
        }

        if (!FoundDef) {
          if (!BlockLiveness[Pred->getNumber()].LiveIn.test(Reg.id())) {
            BlockLiveness[Pred->getNumber()].LiveIn.set(Reg.id());
            WorkList.push_back(Pred);
          }
        }
      }
    }
  }
}

bool SparseLiveVariables::evaluateLiveIn(Register Reg, MachineBasicBlock *MBB,
                                         MachineInstr *IgnoreMI) const {
  for (const MachineInstr &BlockMI : *MBB) {
    if (&BlockMI == IgnoreMI)
      continue;
    if (BlockMI.isPHI()) {
      if (BlockMI.definesRegister(Reg, TRI))
        return false;
      continue;
    }
    if (BlockMI.readsRegister(Reg, TRI))
      return true;
    if (BlockMI.definesRegister(Reg, TRI))
      return false;
  }
  if (hasAnalyzed(MBB))
    return BlockLiveness[MBB->getNumber()].LiveOut.test(Reg.id());
  return false;
}

bool SparseLiveVariables::isLiveOut(Register Reg, MachineBasicBlock *MBB,
                                    MachineInstr *IgnoreMI) const {
  for (MachineBasicBlock *Succ : MBB->successors()) {
    if (hasAnalyzed(Succ) &&
        BlockLiveness[Succ->getNumber()].LiveIn.test(Reg.id()))
      return true;

    for (const MachineInstr &OtherPHI : *Succ) {
      if (!OtherPHI.isPHI())
        break;
      if (&OtherPHI == IgnoreMI)
        continue;
      for (unsigned i = 1; i < OtherPHI.getNumOperands(); i += 2) {
        if (OtherPHI.getOperand(i).isReg() &&
            OtherPHI.getOperand(i).getReg() == Reg &&
            OtherPHI.getOperand(i + 1).getMBB() == MBB) {
          return true;
        }
      }
    }
  }
  return false;
}

void SparseLiveVariables::reevaluateLiveIn(Register Reg, MachineBasicBlock *MBB,
                                           MachineInstr *IgnoreMI) {
  if (!hasAnalyzed(MBB))
    return;

  bool OldLiveIn = BlockLiveness[MBB->getNumber()].LiveIn.test(Reg.id());
  bool NewLiveIn = evaluateLiveIn(Reg, MBB, IgnoreMI);

  if (OldLiveIn != NewLiveIn) {
    if (NewLiveIn) {
      BlockLiveness[MBB->getNumber()].LiveIn.set(Reg.id());
      propagateGrowth(Reg, MBB, IgnoreMI);
    } else {
      BlockLiveness[MBB->getNumber()].LiveIn.reset(Reg.id());
      propagateShrinkage(Reg, MBB, IgnoreMI);
    }
  }
}

void SparseLiveVariables::propagateGrowth(Register Reg,
                                          MachineBasicBlock *StartBB,
                                          MachineInstr *IgnoreMI) {
  SmallVector<MachineBasicBlock *, 8> WorkList;
  WorkList.push_back(StartBB);

  while (!WorkList.empty()) {
    MachineBasicBlock *Curr = WorkList.pop_back_val();
    for (MachineBasicBlock *Pred : Curr->predecessors()) {
      if (!hasAnalyzed(Pred))
        continue;

      if (!BlockLiveness[Pred->getNumber()].LiveOut.test(Reg.id())) {
        BlockLiveness[Pred->getNumber()].LiveOut.set(Reg.id());

        if (!BlockLiveness[Pred->getNumber()].LiveIn.test(Reg.id())) {
          if (evaluateLiveIn(Reg, Pred, IgnoreMI)) {
            BlockLiveness[Pred->getNumber()].LiveIn.set(Reg.id());
            WorkList.push_back(Pred);
          }
        }
      }
    }
  }
}

void SparseLiveVariables::propagateShrinkage(Register Reg,
                                             MachineBasicBlock *StartBB,
                                             MachineInstr *IgnoreMI) {
  SmallVector<MachineBasicBlock *, 8> WorkList;
  WorkList.push_back(StartBB);

  while (!WorkList.empty()) {
    MachineBasicBlock *Curr = WorkList.pop_back_val();
    for (MachineBasicBlock *Pred : Curr->predecessors()) {
      if (!hasAnalyzed(Pred))
        continue;

      if (!BlockLiveness[Pred->getNumber()].LiveOut.test(Reg.id()))
        continue;

      if (!isLiveOut(Reg, Pred, IgnoreMI)) {
        BlockLiveness[Pred->getNumber()].LiveOut.reset(Reg.id());

        if (BlockLiveness[Pred->getNumber()].LiveIn.test(Reg.id())) {
          if (!evaluateLiveIn(Reg, Pred, IgnoreMI)) {
            BlockLiveness[Pred->getNumber()].LiveIn.reset(Reg.id());
            WorkList.push_back(Pred);
          }
        }
      }
    }
  }
}

void SparseLiveVariables::addInstruction(MachineInstr &MI,
                                         MachineBasicBlock *MBB) {
  if (!MBB)
    MBB = MI.getParent();

  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      continue;

    Register Reg = MO.getReg();

    if (MO.isUse() && MI.isPHI()) {
      MachineBasicBlock *Pred = nullptr;
      for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
        if (&MI.getOperand(i) == &MO) {
          Pred = MI.getOperand(i + 1).getMBB();
          break;
        }
      }

      if (Pred) {
        if (hasAnalyzed(Pred) &&
            !BlockLiveness[Pred->getNumber()].LiveOut.test(Reg.id())) {
          BlockLiveness[Pred->getNumber()].LiveOut.set(Reg.id());
          reevaluateLiveIn(Reg, Pred);
        }
      }
      continue;
    }

    reevaluateLiveIn(Reg, MBB);
  }
}

void SparseLiveVariables::removeInstruction(MachineInstr &MI) {
  MachineBasicBlock *MBB = MI.getParent();

  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      continue;

    Register Reg = MO.getReg();

    if (MO.isUse() && MI.isPHI()) {
      MachineBasicBlock *Pred = nullptr;
      for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
        if (&MI.getOperand(i) == &MO) {
          Pred = MI.getOperand(i + 1).getMBB();
          break;
        }
      }

      if (Pred) {
        if (hasAnalyzed(Pred) &&
            BlockLiveness[Pred->getNumber()].LiveOut.test(Reg.id())) {
          if (!isLiveOut(Reg, Pred, &MI)) {
            BlockLiveness[Pred->getNumber()].LiveOut.reset(Reg.id());
            reevaluateLiveIn(Reg, Pred, &MI);
          }
        }
      }
      continue;
    }

    reevaluateLiveIn(Reg, MBB, &MI);
  }
}

void SparseLiveVariables::handleMove(MachineInstr &MI,
                                     MachineBasicBlock * /*OldBB*/,
                                     MachineBasicBlock * /*NewBB*/) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.getReg().isVirtual()) {
      recomputeRegisterLiveness(MO.getReg());
    }
  }
}

char &llvm::SparseLiveVariablesID = SparseLiveVariablesWrapperPass::ID;
