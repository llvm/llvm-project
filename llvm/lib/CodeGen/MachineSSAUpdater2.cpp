//===- MachineSSAUpdater2.cpp - Unstructured SSA Update Tool
//------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MachineSSAUpdater2 class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineSSAUpdater2.h"
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
ComputeLiveInBlocks(const SmallPtrSetImpl<MachineBasicBlock *> &UsingBlocks,
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
MachineSSAUpdater2::CreateInst(unsigned Opc, MachineBasicBlock *BB,
                               MachineBasicBlock::iterator I) {
  return BuildMI(*BB, I, DebugLoc(), TII.get(Opc),
                 MRI.createVirtualRegister(RegAttrs));
}

// IsLiveOut indicates whether we are computing live-out values (true) or
// live-in values (false).
Register MachineSSAUpdater2::ComputeValue(MachineBasicBlock *BB,
                                          bool IsLiveOut) {
  auto *BBInfo = &BBInfos[BB];

  if (IsLiveOut && BBInfo->LiveOutValue)
    return BBInfo->LiveOutValue;

  if (BBInfo->LiveInValue)
    return BBInfo->LiveInValue;

  SmallVector<BBValueInfo *, 4> Stack = {BBInfo};
  MachineBasicBlock *DomBB = BB;
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
    Stack.emplace_back(BBInfo);
  }

  for (auto *BBInfo : Stack)
    // Loop above can insert new entries into the BBInfos map: assume the
    // map shouldn't grow due to [1] and BBInfo references are valid.
    BBInfo->LiveInValue = V;

  if (!V) {
    V = CreateInst(TargetOpcode::IMPLICIT_DEF, BB,
                   IsLiveOut ? BB->getFirstTerminator() : BB->getFirstNonPHI())
            .getReg(0);
    if (IsLiveOut)
      BBInfos[BB].LiveOutValue = V;
    else
      BBInfos[BB].LiveInValue = V;
  }

  return V;
}

/// Perform all the necessary updates, including new PHI-nodes insertion and the
/// requested uses update.
void MachineSSAUpdater2::Calculate() {
  MachineForwardIDFCalculator IDF(DT);

  SmallPtrSet<MachineBasicBlock *, 2> DefBlocks;
  for (auto [BB, V] : Defines)
    DefBlocks.insert(BB);
  IDF.setDefiningBlocks(DefBlocks);

  SmallPtrSet<MachineBasicBlock *, 2> UsingBlocks;
  for (MachineBasicBlock *BB : UseBlocks)
    UsingBlocks.insert(BB);

  SmallVector<MachineBasicBlock *, 32> IDFBlocks;
  SmallPtrSet<MachineBasicBlock *, 32> LiveInBlocks;
  ComputeLiveInBlocks(UsingBlocks, DefBlocks, LiveInBlocks);
  IDF.setLiveInBlocks(LiveInBlocks);
  IDF.calculate(IDFBlocks);

  // Reserve sufficient buckets to prevent map growth. [1]
  BBInfos.reserve(LiveInBlocks.size() + DefBlocks.size());

  for (auto [BB, V] : Defines)
    BBInfos[BB].LiveOutValue = V;

  for (auto *FrontierBB : IDFBlocks) {
    Register NewVR =
        CreateInst(TargetOpcode::PHI, FrontierBB, FrontierBB->begin())
            .getReg(0);
    BBInfos[FrontierBB].LiveInValue = NewVR;
  }

  for (auto *BB : IDFBlocks) {
    auto *PHI = &BB->front();
    assert(PHI->isPHI());
    MachineInstrBuilder MIB(*BB->getParent(), PHI);
    for (MachineBasicBlock *Pred : BB->predecessors())
      MIB.addReg(ComputeValue(Pred, /*IsLiveOut=*/true)).addMBB(Pred);
  }
}

Register MachineSSAUpdater2::GetValueInMiddleOfBlock(MachineBasicBlock *BB) {
  return ComputeValue(BB, /*IsLiveOut=*/false);
}
