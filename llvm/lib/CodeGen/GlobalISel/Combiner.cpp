//===-- lib/CodeGen/GlobalISel/GICombiner.cpp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file constains common code to combine machine functions at generic
// level.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/GlobalISel/GISelWorkList.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gi-combiner"

using namespace llvm;

Combiner::Combiner(CombinerInfo &Info, const TargetPassConfig *TPC)
    : CInfo(Info), TPC(TPC) {
  (void)this->TPC; // FIXME: Remove when used.
}

bool Combiner::combineMachineInstrs(MachineFunction &MF) {
  // If the ISel pipeline failed, do not bother running this pass.
  // FIXME: Should this be here or in individual combiner passes.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;

  MRI = &MF.getRegInfo();
  Builder.setMF(MF);

  DEBUG(dbgs() << "Generic MI Combiner for: " << MF.getName() << '\n');

  MachineOptimizationRemarkEmitter MORE(MF, /*MBFI=*/nullptr);

  bool MFChanged = false;
  bool Changed;

  do {
    // Collect all instructions. Do a post order traversal for basic blocks and
    // insert with list bottom up, so while we pop_back_val, we'll traverse top
    // down RPOT.
    Changed = false;
    GISelWorkList<512> WorkList;
    for (MachineBasicBlock *MBB : post_order(&MF)) {
      if (MBB->empty())
        continue;
      for (auto MII = MBB->rbegin(), MIE = MBB->rend(); MII != MIE;) {
        MachineInstr *CurMI = &*MII;
        ++MII;
        // Erase dead insts before even adding to the list.
        if (isTriviallyDead(*CurMI, *MRI)) {
          DEBUG(dbgs() << *CurMI << "Is dead; erasing.\n");
          CurMI->eraseFromParentAndMarkDBGValuesForRemoval();
          continue;
        }
        WorkList.insert(CurMI);
      }
    }
    // Main Loop. Process the instructions here.
    while (!WorkList.empty()) {
      MachineInstr *CurrInst = WorkList.pop_back_val();
      DEBUG(dbgs() << "Try combining " << *CurrInst << "\n";);
      Changed |= CInfo.combine(*CurrInst, Builder);
    }
    MFChanged |= Changed;
  } while (Changed);

  return MFChanged;
}
