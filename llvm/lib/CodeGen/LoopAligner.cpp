//===-- LoopAligner.cpp - Loop aligner pass. ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass that align loop headers to target specific
// alignment boundary.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loopalign"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

namespace {
  class LoopAligner : public MachineFunctionPass {
  public:
    static char ID;
    LoopAligner() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const { return "Loop aligner"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };

  char LoopAligner::ID = 0;
} // end anonymous namespace

FunctionPass *llvm::createLoopAlignerPass() { return new LoopAligner(); }

bool LoopAligner::runOnMachineFunction(MachineFunction &MF) {
  const MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfo>();

  if (MLI->empty())
    return false;  // No loops.

  const TargetLowering *TLI = MF.getTarget().getTargetLowering();
  if (!TLI)
    return false;

  unsigned Align = TLI->getPrefLoopAlignment();
  if (!Align)
    return false;  // Don't care about loop alignment.

  const Function *F = MF.getFunction();
  if (F->hasFnAttr(Attribute::OptimizeForSize))
    return false;

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = I;
    if (MLI->isLoopHeader(MBB)) {
      MachineBasicBlock *PredBB = prior(I);
      if (MLI->getLoopFor(MBB) == MLI->getLoopFor(PredBB))
        // If previously BB is in the same loop, don't align this BB. We want
        // to prevent adding noop's inside a loop.
        continue;
      MBB->setAlignment(Align);
    }
  }

  return true;
}
