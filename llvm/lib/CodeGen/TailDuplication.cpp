//===- TailDuplication.cpp - Duplicate blocks into predecessors' tails ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass duplicates basic blocks ending in unconditional branches
/// into the tails of their predecessors, using the TailDuplicator utility
/// class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TailDuplicator.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "tailduplication"

namespace {

class TailDuplicateBase : public MachineFunctionPass {
  TailDuplicator Duplicator;
  bool PreRegAlloc;
public:
  TailDuplicateBase(char &PassID, bool PreRegAlloc)
    : MachineFunctionPass(PassID), PreRegAlloc(PreRegAlloc) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineBranchProbabilityInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class TailDuplicate : public TailDuplicateBase {
public:
  static char ID;
  TailDuplicate() : TailDuplicateBase(ID, false) {
    initializeTailDuplicatePass(*PassRegistry::getPassRegistry());
  }
};

class EarlyTailDuplicate : public TailDuplicateBase {
public:
  static char ID;
  EarlyTailDuplicate() : TailDuplicateBase(ID, true) {
    initializeEarlyTailDuplicatePass(*PassRegistry::getPassRegistry());
  }
};

} // end anonymous namespace

char TailDuplicate::ID;
char EarlyTailDuplicate::ID;

char &llvm::TailDuplicateID = TailDuplicate::ID;
char &llvm::EarlyTailDuplicateID = EarlyTailDuplicate::ID;

INITIALIZE_PASS(TailDuplicate, DEBUG_TYPE, "Tail Duplication", false, false)
INITIALIZE_PASS(EarlyTailDuplicate, "early-tailduplication",
                "Early Tail Duplication", false, false)

bool TailDuplicateBase::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto MBPI = &getAnalysis<MachineBranchProbabilityInfo>();
  Duplicator.initMF(MF, PreRegAlloc, MBPI, /*LayoutMode=*/false);

  bool MadeChange = false;
  while (Duplicator.tailDuplicateBlocks())
    MadeChange = true;

  return MadeChange;
}
