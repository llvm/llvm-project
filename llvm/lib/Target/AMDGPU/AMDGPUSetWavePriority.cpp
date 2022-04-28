//===- AMDGPUSetWavePriority.cpp - Set wave priority ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Pass to temporarily raise the wave priority beginning the start of
/// the shader function until its last VMEM instructions to allow younger
/// waves to issue their VMEM instructions as well.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Allocator.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-set-wave-priority"

namespace {

struct MBBInfo {
  MBBInfo() = default;
  bool MayReachVMEMLoad = false;
};

using MBBInfoSet = DenseMap<const MachineBasicBlock *, MBBInfo>;

class AMDGPUSetWavePriority : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUSetWavePriority() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "Set wave priority"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  MachineInstr *BuildSetprioMI(MachineFunction &MF, unsigned priority) const;

  const SIInstrInfo *TII;
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUSetWavePriority, DEBUG_TYPE, "Set wave priority", false,
                false)

char AMDGPUSetWavePriority::ID = 0;

FunctionPass *llvm::createAMDGPUSetWavePriorityPass() {
  return new AMDGPUSetWavePriority();
}

MachineInstr *AMDGPUSetWavePriority::BuildSetprioMI(MachineFunction &MF,
                                                    unsigned priority) const {
  return BuildMI(MF, DebugLoc(), TII->get(AMDGPU::S_SETPRIO)).addImm(priority);
}

// Checks that for every predecessor Pred that can reach a VMEM load,
// none of Pred's successors can reach a VMEM load.
static bool CanLowerPriorityDirectlyInPredecessors(const MachineBasicBlock &MBB,
                                                   MBBInfoSet &MBBInfos) {
  for (const MachineBasicBlock *Pred : MBB.predecessors()) {
    if (!MBBInfos[Pred].MayReachVMEMLoad)
      continue;
    for (const MachineBasicBlock *Succ : Pred->successors()) {
      if (MBBInfos[Succ].MayReachVMEMLoad)
        return false;
    }
  }
  return true;
}

static bool isVMEMLoad(const MachineInstr &MI) {
  return SIInstrInfo::isVMEM(MI) && MI.mayLoad();
}

bool AMDGPUSetWavePriority::runOnMachineFunction(MachineFunction &MF) {
  const unsigned HighPriority = 3;
  const unsigned LowPriority = 0;

  Function &F = MF.getFunction();
  if (skipFunction(F) || !AMDGPU::isEntryFunctionCC(F.getCallingConv()))
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();

  MBBInfoSet MBBInfos;
  SmallVector<const MachineBasicBlock *, 16> Worklist;
  for (MachineBasicBlock &MBB : MF) {
    if (any_of(MBB, isVMEMLoad))
      Worklist.push_back(&MBB);
  }

  // Mark blocks from which control may reach VMEM loads.
  while (!Worklist.empty()) {
    const MachineBasicBlock *MBB = Worklist.pop_back_val();
    MBBInfo &Info = MBBInfos[MBB];
    if (!Info.MayReachVMEMLoad) {
      Info.MayReachVMEMLoad = true;
      Worklist.append(MBB->pred_begin(), MBB->pred_end());
    }
  }

  MachineBasicBlock &Entry = MF.front();
  if (!MBBInfos[&Entry].MayReachVMEMLoad)
    return false;

  // Raise the priority at the beginning of the shader.
  MachineBasicBlock::iterator I = Entry.begin(), E = Entry.end();
  while (I != E && !SIInstrInfo::isVALU(*I) && !I->isTerminator())
    ++I;
  Entry.insert(I, BuildSetprioMI(MF, HighPriority));

  // Lower the priority on edges where control leaves blocks from which
  // VMEM loads are reachable.
  SmallSet<MachineBasicBlock *, 16> PriorityLoweringBlocks;
  for (MachineBasicBlock &MBB : MF) {
    if (MBBInfos[&MBB].MayReachVMEMLoad) {
      if (MBB.succ_empty())
        PriorityLoweringBlocks.insert(&MBB);
      continue;
    }

    if (CanLowerPriorityDirectlyInPredecessors(MBB, MBBInfos)) {
      for (MachineBasicBlock *Pred : MBB.predecessors()) {
        if (MBBInfos[Pred].MayReachVMEMLoad)
          PriorityLoweringBlocks.insert(Pred);
      }
      continue;
    }

    // Where lowering the priority in predecessors is not possible, the
    // block receiving control either was not part of a loop in the first
    // place or the loop simplification/canonicalization pass should have
    // already tried to split the edge and insert a preheader, and if for
    // whatever reason it failed to do so, then this leaves us with the
    // only option of lowering the priority within the loop.
    PriorityLoweringBlocks.insert(&MBB);
  }

  for (MachineBasicBlock *MBB : PriorityLoweringBlocks) {
    MachineBasicBlock::iterator I = MBB->end(), B = MBB->begin();
    while (I != B) {
      if (isVMEMLoad(*--I)) {
        ++I;
        break;
      }
    }
    MBB->insert(I, BuildSetprioMI(MF, LowPriority));
  }

  return true;
}
