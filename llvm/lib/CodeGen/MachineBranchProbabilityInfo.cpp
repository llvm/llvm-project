//===- MachineBranchProbabilityInfo.cpp - Machine Branch Probability Info -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This analysis uses probability info stored in Machine Basic Blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

INITIALIZE_PASS_BEGIN(MachineBranchProbabilityInfo, "machine-branch-prob",
                      "Machine Branch Probability Analysis", false, true)
INITIALIZE_PASS_END(MachineBranchProbabilityInfo, "machine-branch-prob",
                    "Machine Branch Probability Analysis", false, true)

char MachineBranchProbabilityInfo::ID = 0;

void MachineBranchProbabilityInfo::anchor() { }

uint32_t MachineBranchProbabilityInfo::getEdgeWeight(
    const MachineBasicBlock *Src,
    MachineBasicBlock::const_succ_iterator Dst) const {
  return Src->getSuccProbability(Dst).getNumerator();
}

uint32_t MachineBranchProbabilityInfo::getEdgeWeight(
    const MachineBasicBlock *Src, const MachineBasicBlock *Dst) const {
  // This is a linear search. Try to use the const_succ_iterator version when
  // possible.
  return getEdgeWeight(Src, std::find(Src->succ_begin(), Src->succ_end(), Dst));
}

BranchProbability MachineBranchProbabilityInfo::getEdgeProbability(
    const MachineBasicBlock *Src,
    MachineBasicBlock::const_succ_iterator Dst) const {
  return Src->getSuccProbability(Dst);
}

BranchProbability MachineBranchProbabilityInfo::getEdgeProbability(
    const MachineBasicBlock *Src, const MachineBasicBlock *Dst) const {
  // This is a linear search. Try to use the const_succ_iterator version when
  // possible.
  return getEdgeProbability(Src,
                            std::find(Src->succ_begin(), Src->succ_end(), Dst));
}

bool
MachineBranchProbabilityInfo::isEdgeHot(const MachineBasicBlock *Src,
                                        const MachineBasicBlock *Dst) const {
  // Hot probability is at least 4/5 = 80%
  static BranchProbability HotProb(4, 5);
  return getEdgeProbability(Src, Dst) > HotProb;
}

MachineBasicBlock *
MachineBranchProbabilityInfo::getHotSucc(MachineBasicBlock *MBB) const {
  auto MaxProb = BranchProbability::getZero();
  MachineBasicBlock *MaxSucc = nullptr;
  for (MachineBasicBlock::const_succ_iterator I = MBB->succ_begin(),
       E = MBB->succ_end(); I != E; ++I) {
    auto Prob = getEdgeProbability(MBB, I);
    if (Prob > MaxProb) {
      MaxProb = Prob;
      MaxSucc = *I;
    }
  }

  static BranchProbability HotProb(4, 5);
  if (getEdgeProbability(MBB, MaxSucc) >= HotProb)
    return MaxSucc;

  return nullptr;
}

raw_ostream &MachineBranchProbabilityInfo::printEdgeProbability(
    raw_ostream &OS, const MachineBasicBlock *Src,
    const MachineBasicBlock *Dst) const {

  const BranchProbability Prob = getEdgeProbability(Src, Dst);
  OS << "edge MBB#" << Src->getNumber() << " -> MBB#" << Dst->getNumber()
     << " probability is " << Prob
     << (isEdgeHot(Src, Dst) ? " [HOT edge]\n" : "\n");

  return OS;
}
