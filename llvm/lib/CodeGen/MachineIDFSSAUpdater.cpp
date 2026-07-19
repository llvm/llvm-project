//===- MachineIDFSSAUpdater.cpp - Unstructured SSA Update Tool ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MachineIDFSSAUpdater class, which provides an
// efficient SSA form maintenance utility for machine-level IR. It uses the
// iterated dominance frontier (IDF) algorithm via MachineForwardIDFCalculator
// to compute phi-function placement, offering better performance than the
// incremental MachineSSAUpdater approach. The updater requires a single call
// to calculate() after all definitions and uses have been registered.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineIDFSSAUpdater.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/Debug.h"

namespace llvm {

template <bool IsPostDom>
class MachineIDFCalculator final
    : public IDFCalculatorBase<MachineBasicBlock, IsPostDom> {
public:
  using IDFCalculatorBase =
      typename llvm::IDFCalculatorBase<MachineBasicBlock, IsPostDom>;
  using ChildrenGetterTy = typename IDFCalculatorBase::ChildrenGetterTy;

  MachineIDFCalculator(DominatorTreeBase<MachineBasicBlock, IsPostDom> &DT)
      : IDFCalculatorBase(DT) {}
};

using MachineForwardIDFCalculator = MachineIDFCalculator<false>;
using MachineReverseIDFCalculator = MachineIDFCalculator<true>;

} // namespace llvm

using namespace llvm;

/// Given sets of UsingBlocks and DefBlocks, compute the set of LiveInBlocks.
/// This is basically a subgraph limited by DefBlocks and UsingBlocks.
static void
computeLiveInBlocks(const SmallPtrSetImpl<MachineBasicBlock *> &UsingBlocks,
                    const SmallPtrSetImpl<MachineBasicBlock *> &DefBlocks,
                    SmallPtrSetImpl<MachineBasicBlock *> &LiveInBlocks) {
  // To determine liveness, we must iterate through the predecessors of blocks
  // where the def is live.  Blocks are added to the worklist if we need to
  // check their predecessors.  Start with all the using blocks.
  SmallVector<MachineBasicBlock *, 64> LiveInBlockWorklist(UsingBlocks.begin(),
                                                           UsingBlocks.end());

  // Now that we have a set of blocks where the phi is live-in, recursively add
  // their predecessors until we find the full region the value is live.
  while (!LiveInBlockWorklist.empty()) {
    MachineBasicBlock *BB = LiveInBlockWorklist.pop_back_val();

    // The block really is live in here, insert it into the set.  If already in
    // the set, then it has already been processed.
    if (!LiveInBlocks.insert(BB).second)
      continue;

    // Since the value is live into BB, it is either defined in a predecessor or
    // live into it to.  Add the preds to the worklist unless they are a
    // defining block.
    for (MachineBasicBlock *P : BB->predecessors()) {
      // The value is not live into a predecessor if it defines the value.
      if (DefBlocks.count(P))
        continue;

      // Otherwise it is, add to the worklist.
      LiveInBlockWorklist.push_back(P);
    }
  }
}

MachineInstrBuilder
MachineIDFSSAUpdater::createInst(unsigned Opc, MachineBasicBlock *BB,
                                 MachineBasicBlock::iterator I) {
  return BuildMI(*BB, I, DebugLoc(), TII.get(Opc),
                 MRI.createVirtualRegister(RegAttrs));
}

// IsLiveOut indicates whether we are computing live-out values (true) or
// live-in values (false).
Register MachineIDFSSAUpdater::computeValue(MachineBasicBlock *BB,
                                            bool IsLiveOut) {
  BBValueInfo *BBInfo = &BBInfos[BB];

  if (IsLiveOut && BBInfo->LiveOutValue)
    return BBInfo->LiveOutValue;

  if (BBInfo->LiveInValue)
    return BBInfo->LiveInValue;

  SmallVector<BBValueInfo *, 4> DomPath = {BBInfo};
  MachineBasicBlock *DomBB = BB, *TopDomBB = BB;
  Register V;

  while (DT.isReachableFromEntry(DomBB) && !DomBB->pred_empty() &&
         (DomBB = DT.getNode(DomBB)->getIDom()->getBlock())) {
    BBInfo = &BBInfos[DomBB];
    if (BBInfo->LiveOutValue) {
      V = BBInfo->LiveOutValue;
      break;
    }
    if (BBInfo->LiveInValue) {
      V = BBInfo->LiveInValue;
      break;
    }
    TopDomBB = DomBB;
    DomPath.emplace_back(BBInfo);
  }

  if (!V) {
    V = createInst(TargetOpcode::IMPLICIT_DEF, TopDomBB,
                   TopDomBB->getFirstNonPHI())
            .getReg(0);
  }

  for (BBValueInfo *BBInfo : DomPath) {
    // Loop above can insert new entries into the BBInfos map: assume the
    // map shouldn't grow as the caller should have been allocated enough
    // buckets, see [1].
    BBInfo->LiveInValue = V;
  }

  return V;
}

/// Perform all the necessary updates, including new PHI-nodes insertion and the
/// requested uses update.
void MachineIDFSSAUpdater::calculate() {
  MachineForwardIDFCalculator IDF(DT);

  SmallPtrSet<MachineBasicBlock *, 2> DefBlocks;
  for (auto [BB, V] : Defines)
    DefBlocks.insert(BB);
  IDF.setDefiningBlocks(DefBlocks);

  SmallPtrSet<MachineBasicBlock *, 2> UsingBlocks(UseBlocks.begin(),
                                                  UseBlocks.end());
  SmallVector<MachineBasicBlock *, 4> IDFBlocks;
  SmallPtrSet<MachineBasicBlock *, 4> LiveInBlocks;
  computeLiveInBlocks(UsingBlocks, DefBlocks, LiveInBlocks);
  IDF.setLiveInBlocks(LiveInBlocks);
  IDF.calculate(IDFBlocks);

  // Reserve sufficient buckets to prevent map growth. [1]
  BBInfos.reserve(LiveInBlocks.size() + DefBlocks.size());

  for (auto [BB, V] : Defines)
    BBInfos[BB].LiveOutValue = V;

  for (MachineBasicBlock *FrontierBB : IDFBlocks) {
    Register NewVR =
        createInst(TargetOpcode::PHI, FrontierBB, FrontierBB->begin())
            .getReg(0);
    BBInfos[FrontierBB].LiveInValue = NewVR;
  }

  for (MachineBasicBlock *BB : IDFBlocks) {
    auto *PHI = &BB->front();
    assert(PHI->isPHI());
    MachineInstrBuilder MIB(*BB->getParent(), PHI);
    for (MachineBasicBlock *Pred : BB->predecessors())
      MIB.addReg(computeValue(Pred, /*IsLiveOut=*/true)).addMBB(Pred);
  }
}

Register MachineIDFSSAUpdater::getValueInMiddleOfBlock(MachineBasicBlock *BB) {
  return computeValue(BB, /*IsLiveOut=*/false);
}
