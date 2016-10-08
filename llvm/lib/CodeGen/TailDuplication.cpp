//===-- TailDuplication.cpp - Duplicate blocks into predecessors' tails ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass duplicates basic blocks ending in unconditional branches into
// the tails of their predecessors, using the TailDuplicator utility class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TailDuplicator.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "tailduplication"

namespace {
/// Perform tail duplication. Delegates to TailDuplicator
class TailDuplicatePass : public MachineFunctionPass {
  TailDuplicator Duplicator;

public:
  static char ID;
  explicit TailDuplicatePass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

char TailDuplicatePass::ID = 0;
}

char &llvm::TailDuplicateID = TailDuplicatePass::ID;

INITIALIZE_PASS(TailDuplicatePass, "tailduplication", "Tail Duplication", false,
                false)

bool TailDuplicatePass::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  auto MBPI = &getAnalysis<MachineBranchProbabilityInfo>();

  Duplicator.initMF(MF, MBPI);

  bool MadeChange = false;
  while (Duplicator.tailDuplicateBlocks())
    MadeChange = true;

  return MadeChange;
}

void TailDuplicatePass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
}
