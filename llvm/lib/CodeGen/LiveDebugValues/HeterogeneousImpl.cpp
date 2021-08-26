//===- HeterogeneousImpl.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file HeterogeneousImpl.cpp
///
//===----------------------------------------------------------------------===//

#include "LiveDebugValues.h"

#include "LiveDebugValues.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundleIterator.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <deque>
#include <utility>
#include <vector>

namespace llvm {
class TargetPassConfig;
} // namespace llvm

using namespace llvm;

#define DEBUG_TYPE "livedebugvalues"

STATISTIC(NumInserted, "Number of DBG_DEF instructions inserted");

namespace {

class HeterogeneousLDV : public LDVImpl {
private:
  bool ExtendRanges(MachineFunction &MF, MachineDominatorTree *DomTree,
                    TargetPassConfig *TPC, unsigned InputBBLimit,
                    unsigned InputDbgValLimit) override;

public:
  HeterogeneousLDV();
  ~HeterogeneousLDV();
};

} // end anonymous namespace

HeterogeneousLDV::HeterogeneousLDV() {}

HeterogeneousLDV::~HeterogeneousLDV() {}

bool HeterogeneousLDV::ExtendRanges(MachineFunction &MF,
                                    MachineDominatorTree *DomTree,
                                    TargetPassConfig *TPC,
                                    unsigned InputBBLimit,
                                    unsigned InputDbgValLimit) {
  LLVM_DEBUG(dbgs() << "\nDebug Range Extension\n");

  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  bool Changed = false;

// FIXME: All of these asserts should be graceful errors, but I don't know how
// to achieve that in a function pass?
#ifndef NDEBUG
  // To diagnose kills which are not dominated by a corresponding def we record
  // kills which were successfully reached by a def.
  SmallPtrSet<MachineInstr *, 16> ReachedDbgKills;
#endif

  std::vector<MachineInstr *> OriginalDbgDefs;

  // We will insert additional DBG_DEFs which we do not want to consider, so we
  // record the set present on entry to the pass.
  for (auto &&MBB : MF)
    for (auto &&MI : MBB)
      if (MI.isDebugDef())
        OriginalDbgDefs.push_back(&MI);

#ifndef NDEBUG
  SmallPtrSet<DILifetime *, 16> ReachedLifetimes;
  // To diagnose multiple def intrinsics referring to the same lifetime before
  // this pass we record the lifetimes as we encounter them.
  for (auto &&DbgDef : OriginalDbgDefs)
    assert(ReachedLifetimes.insert(DbgDef->getDebugLifetime()).second &&
           "Bounded lifetime referenced by more than a single def");
#endif


  for (auto &&DbgDef : OriginalDbgDefs) {
    std::deque<MachineBasicBlock *> PendingMBBs{DbgDef->getParent()};
    SmallPtrSet<MachineBasicBlock *, 16> SeenMBBs{DbgDef->getParent()};
    while (!PendingMBBs.empty()) {
      MachineBasicBlock *MBB = PendingMBBs.front();
      PendingMBBs.pop_front();

      // First scan if this block kills the lifetime of the def we are
      // currently processing. If so, we don't want to consider any successors
      // of MBB.
      auto MBBI = find_if(MBB->instrs(), [&](auto &&MI) {
        return MI.isDebugKill() &&
               MI.getDebugLifetime() == DbgDef->getDebugLifetime();
      });
      if (MBBI != MBB->end()) {
#ifndef NDEBUG
        bool Inserted = ReachedDbgKills.insert(&*MBBI).second;
        (void)Inserted;
        assert(Inserted &&
               "Should never revisit a DBG_KILL in livedebugvalues");
#endif
        continue;
      }

      // If the def is live through the block, we can propogate defs into the
      // beginning of each unvisited successor of MBB, and then add them to the
      // worklist.
      for (auto &&Successor : MBB->successors()) {
        if (SeenMBBs.insert(Successor).second) {
          Changed = true;
          NumInserted++;
          auto MIB = BuildMI(*Successor, Successor->begin(), DebugLoc(),
                             TII->get(TargetOpcode::DBG_DEF));
          for (auto &&O : DbgDef->operands())
            MIB.add(O);
          PendingMBBs.push_back(Successor);
        }
      }
    }
  }

#ifndef NDEBUG
  for (auto &&MBB : MF)
    for (auto &&MI : MBB)
      assert(!MI.isDebugKill() ||
             ReachedDbgKills.count(&MI) && "Orphaned DBG_KILL");
#endif

  return Changed;
}

LDVImpl *llvm::makeHeterogeneousLiveDebugValues() {
  return new HeterogeneousLDV();
}
