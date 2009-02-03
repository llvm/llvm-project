//===- MachineLoopInfo.cpp - Natural Loop Calculator ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachineLoopInfo class that is used to identify natural
// loops and determine the loop depth of various nodes of the CFG.  Note that
// the loops identified may actually be several natural loops that share the 
// same header node... not just a single natural loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/Passes.h"
using namespace llvm;

TEMPLATE_INSTANTIATION(class LoopBase<MachineBasicBlock>);
TEMPLATE_INSTANTIATION(class LoopInfoBase<MachineBasicBlock>);

char MachineLoopInfo::ID = 0;
static RegisterPass<MachineLoopInfo>
X("machine-loops", "Machine Natural Loop Construction", true);

const PassInfo *const llvm::MachineLoopInfoID = &X;

bool MachineLoopInfo::runOnMachineFunction(MachineFunction &) {
  releaseMemory();
  LI->Calculate(getAnalysis<MachineDominatorTree>().getBase());    // Update
  return false;
}

void MachineLoopInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineDominatorTree>();
}
