//===-- AGXHoistFromElseFlow.cpp - Keep else flow blocks free of code -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The structurizer's flow block between the two sides of an if / else ends
// in AGX_ELSE, which AGXLowerControlFlow expands into else_fcmp: the
// instructions before it run with the execution mask of the then side only.
// (SILowerControlFlow can restore the union of both sides at the start of
// the block; the AGX nesting counter has no such operation.) The flow block
// dominates the else side and the join, so the machine SSA optimizations
// put code there that the else side or the join reads: MachineCSE's PRE
// hoists computations common to two blocks below it into it, and
// MachineSink sinks values from the if block into it. Such code would leave
// stale registers in the threads that take the else side.
//
// This pass moves these instructions to the end of the block's immediate
// dominator, the if block, which runs with the execution mask of both sides,
// when they only read values available there and have no memory or other
// side effects (which rules out every move that could change a result).
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "agx-hoist-from-else-flow"

namespace {

class AGXHoistFromElseFlow : public MachineFunctionPass {
public:
  static char ID;
  AGXHoistFromElseFlow() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AGX Hoist From Else Flow Blocks";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXHoistFromElseFlow::ID = 0;

INITIALIZE_PASS_BEGIN(AGXHoistFromElseFlow, DEBUG_TYPE,
                      "AGX Hoist From Else Flow Blocks", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AGXHoistFromElseFlow, DEBUG_TYPE,
                    "AGX Hoist From Else Flow Blocks", false, false)

FunctionPass *llvm::createAGXHoistFromElseFlowPass() {
  return new AGXHoistFromElseFlow();
}

static bool endsInOpcode(const MachineBasicBlock &MBB, unsigned Opcode) {
  for (const MachineInstr &MI : MBB.terminators())
    if (MI.getOpcode() == Opcode)
      return true;
  return false;
}

bool AGXHoistFromElseFlow::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    if (!endsInOpcode(MBB, AGX::AGX_ELSE))
      continue;
    MachineDomTreeNode *Node = MDT.getNode(&MBB);
    if (!Node || !Node->getIDom())
      continue;
    MachineBasicBlock *If = Node->getIDom()->getBlock();
    if (!endsInOpcode(*If, AGX::AGX_IF))
      continue;

    for (MachineInstr &MI : llvm::make_early_inc_range(
             llvm::make_range(MBB.getFirstNonPHI(), MBB.getFirstTerminator()))) {
      if (MI.isDebugInstr() || MI.mayLoadOrStore() ||
          MI.hasUnmodeledSideEffects() || MI.isConvergent())
        continue;
      bool Movable = true;
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg())
          continue;
        if (!MO.getReg().isVirtual()) {
          Movable = false;
          break;
        }
        if (MO.isDef())
          continue;
        MachineInstr *Def = MRI.getVRegDef(MO.getReg());
        if (!Def || !MDT.properlyDominates(Def->getParent(), &MBB)) {
          Movable = false;
          break;
        }
      }
      if (!Movable) {
        LLVM_DEBUG(dbgs() << "cannot hoist from " << printMBBReference(MBB)
                          << ": " << MI);
        continue;
      }
      LLVM_DEBUG(dbgs() << "hoisting from " << printMBBReference(MBB)
                        << " to " << printMBBReference(*If) << ": " << MI);
      If->splice(If->getFirstTerminator(), &MBB, MI.getIterator());
      Changed = true;
    }
  }
  return Changed;
}
