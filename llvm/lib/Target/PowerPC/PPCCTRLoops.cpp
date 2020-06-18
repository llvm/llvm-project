//===-- PPCCTRLoops.cpp - Identify and generate CTR loops -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass identifies loops where we can generate the PPC branch instructions
// that decrement and test the count register (CTR) (bdnz and friends).
//
// The pattern that defines the induction variable can changed depending on
// prior optimizations.  For example, the IndVarSimplify phase run by 'opt'
// normalizes induction variables, and the Loop Strength Reduction pass
// run by 'llc' may also make changes to the induction variable.
//
// Criteria for CTR loops:
//  - Countable loops (w/ ind. var for a trip count)
//  - Try inner-most loops first
//  - No nested CTR loops.
//  - No function calls in loops.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCSubtarget.h"
#include "PPCTargetMachine.h"
#include "PPCTargetTransformInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#ifndef NDEBUG
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#endif

using namespace llvm;

#define DEBUG_TYPE "ctrloops"

#ifndef NDEBUG
static cl::opt<int> CTRLoopLimit("ppc-max-ctrloop", cl::Hidden, cl::init(-1));
#endif

namespace {

#ifndef NDEBUG
  struct PPCCTRLoopsVerify : public MachineFunctionPass {
  public:
    static char ID;

    PPCCTRLoopsVerify() : MachineFunctionPass(ID) {
      initializePPCCTRLoopsVerifyPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

  private:
    MachineDominatorTree *MDT;
  };

  char PPCCTRLoopsVerify::ID = 0;
#endif // NDEBUG
} // end anonymous namespace

#ifndef NDEBUG
INITIALIZE_PASS_BEGIN(PPCCTRLoopsVerify, "ppc-ctr-loops-verify",
                      "PowerPC CTR Loops Verify", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(PPCCTRLoopsVerify, "ppc-ctr-loops-verify",
                    "PowerPC CTR Loops Verify", false, false)

FunctionPass *llvm::createPPCCTRLoopsVerify() {
  return new PPCCTRLoopsVerify();
}
#endif // NDEBUG

#ifndef NDEBUG
static bool clobbersCTR(const MachineInstr &MI) {
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (MO.isReg()) {
      if (MO.isDef() && (MO.getReg() == PPC::CTR || MO.getReg() == PPC::CTR8))
        return true;
    } else if (MO.isRegMask()) {
      if (MO.clobbersPhysReg(PPC::CTR) || MO.clobbersPhysReg(PPC::CTR8))
        return true;
    }
  }

  return false;
}

static bool verifyCTRBranch(MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator I) {
  MachineBasicBlock::iterator BI = I;
  SmallSet<MachineBasicBlock *, 16>   Visited;
  SmallVector<MachineBasicBlock *, 8> Preds;
  bool CheckPreds;

  if (I == MBB->begin()) {
    Visited.insert(MBB);
    goto queue_preds;
  } else
    --I;

check_block:
  Visited.insert(MBB);
  if (I == MBB->end())
    goto queue_preds;

  CheckPreds = true;
  for (MachineBasicBlock::iterator IE = MBB->begin();; --I) {
    unsigned Opc = I->getOpcode();
    if (Opc == PPC::MTCTRloop || Opc == PPC::MTCTR8loop) {
      CheckPreds = false;
      break;
    }

    if (I != BI && clobbersCTR(*I)) {
      LLVM_DEBUG(dbgs() << printMBBReference(*MBB) << " (" << MBB->getFullName()
                        << ") instruction " << *I
                        << " clobbers CTR, invalidating "
                        << printMBBReference(*BI->getParent()) << " ("
                        << BI->getParent()->getFullName() << ") instruction "
                        << *BI << "\n");
      return false;
    }

    if (I == IE)
      break;
  }

  if (!CheckPreds && Preds.empty())
    return true;

  if (CheckPreds) {
queue_preds:
    if (MachineFunction::iterator(MBB) == MBB->getParent()->begin()) {
      LLVM_DEBUG(dbgs() << "Unable to find a MTCTR instruction for "
                        << printMBBReference(*BI->getParent()) << " ("
                        << BI->getParent()->getFullName() << ") instruction "
                        << *BI << "\n");
      return false;
    }

    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PIE = MBB->pred_end(); PI != PIE; ++PI)
      Preds.push_back(*PI);
  }

  do {
    MBB = Preds.pop_back_val();
    if (!Visited.count(MBB)) {
      I = MBB->getLastNonDebugInstr();
      goto check_block;
    }
  } while (!Preds.empty());

  return true;
}

bool PPCCTRLoopsVerify::runOnMachineFunction(MachineFunction &MF) {
  MDT = &getAnalysis<MachineDominatorTree>();

  // Verify that all bdnz/bdz instructions are dominated by a loop mtctr before
  // any other instructions that might clobber the ctr register.
  for (MachineFunction::iterator I = MF.begin(), IE = MF.end();
       I != IE; ++I) {
    MachineBasicBlock *MBB = &*I;
    if (!MDT->isReachableFromEntry(MBB))
      continue;

    for (MachineBasicBlock::iterator MII = MBB->getFirstTerminator(),
      MIIE = MBB->end(); MII != MIIE; ++MII) {
      unsigned Opc = MII->getOpcode();
      if (Opc == PPC::BDNZ8 || Opc == PPC::BDNZ ||
          Opc == PPC::BDZ8  || Opc == PPC::BDZ)
        if (!verifyCTRBranch(MBB, MII))
          llvm_unreachable("Invalid PPC CTR loop!");
    }
  }

  return false;
}
#endif // NDEBUG
