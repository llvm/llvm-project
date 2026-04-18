#include "llvm/CodeGen/SparseLiveVariables.h"
#include "llvm/InitializePasses.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "sparse-live-variables"

char SparseLiveVariables::ID = 0;
INITIALIZE_PASS(SparseLiveVariables, DEBUG_TYPE, "Sparse Live Variable Analysis", false, false)

bool SparseLiveVariables::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()) || MF.empty())
    return false;

  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  
  BlockLiveness.clear();

  SmallPtrSet<MachineBasicBlock *, 16> Reachable;
  for (MachineBasicBlock *MBB : llvm::depth_first(&MF))
    Reachable.insert(MBB);

  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (MachineBasicBlock *MBB : llvm::post_order(&MF)) {
      if (!Reachable.count(MBB)) continue;

      SparseBitVector<> OldLiveIn = BlockLiveness[MBB].LiveIn;
      SparseBitVector<> OldLiveOut = BlockLiveness[MBB].LiveOut;

      for (const MachineBasicBlock *Succ : MBB->successors())
        if (Reachable.count(Succ))
          BlockLiveness[MBB].LiveOut |= BlockLiveness[Succ].LiveIn;

      SparseBitVector<> LiveIn = BlockLiveness[MBB].LiveOut;
      LivenessTracker Tracker(LiveIn, MRI);

      for (MachineInstr &MI : llvm::reverse(*MBB))
        Tracker.stepBackward(MI);

      BlockLiveness[MBB].LiveIn = Tracker.getLiveSet();

      if (BlockLiveness[MBB].LiveIn != OldLiveIn ||
          BlockLiveness[MBB].LiveOut != OldLiveOut)
        Changed = true;
    }
  }



  return false;
}






void SparseLiveVariables::verifyLiveness(const MachineFunction &MF) const {
  for (const MachineBasicBlock &MBB : MF) {
    auto It = BlockLiveness.find(&MBB);
    if (It == BlockLiveness.end()) continue;

    const SparseBitVector<> &LiveIn = It->second.LiveIn;
    for (const auto &LI : MBB.liveins()) {
      if (!LiveIn.test(LI.PhysReg.id())) {
        LLVM_DEBUG(dbgs() << "Warning: Live-in register " << printReg(LI.PhysReg, TRI)
                          << " missing from computed live-in set of block "
                          << printMBBReference(MBB) << "\n");
      }
    }
  }
}
char &llvm::SparseLiveVariablesID = SparseLiveVariables::ID;
