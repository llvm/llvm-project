//===- CountInstructions.cpp
//----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CountInstructions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Casting.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "count-instructions"

STATISTIC(TotalBasicBlocks, "Number of basic blocks");
STATISTIC(TotalInstructions, "Number of total instructions");
STATISTIC(TotalBranchInstructions, "Number of branch instructions");
STATISTIC(TotalSwitchInstructions, "Number of switch instructions");
STATISTIC(TotalSuccessors, "Number of basic block successors");
STATISTIC(TotalBranchSuccessors, "Number of branch successors");
STATISTIC(TotalSwitchSuccessors, "Number of switch successors");
PreservedAnalyses CountInstructionsPass::run(Function &F,
                                             FunctionAnalysisManager &) {
  uint32_t CountBasicBlocks = 0;
  uint32_t CountInstructions = 0;
  uint32_t CountBranchInstructions = 0;
  uint32_t CountSwitchInstructions = 0;
  uint32_t CountSuccessors = 0;
  uint32_t CountBranchSuccessors = 0;
  uint32_t CountSwitchSuccessors = 0;

  for (BasicBlock &BB : F) {
    CountBasicBlocks++;
    Instruction *I = BB.getTerminator();
    CountSuccessors += I->getNumSuccessors();
    if (isa<BranchInst>(I)) {
      CountBranchInstructions++;
      CountBranchSuccessors += I->getNumSuccessors();
    } else if (isa<SwitchInst>(I)) {
      CountSwitchInstructions++;
      CountSwitchSuccessors += I->getNumSuccessors();
    }
    CountInstructions += BB.size();
  }
  TotalInstructions += CountInstructions;
  TotalBasicBlocks += CountBasicBlocks;
  TotalBranchInstructions += CountBranchInstructions;
  TotalSwitchInstructions += CountSwitchInstructions;
  TotalSuccessors += CountSuccessors;
  TotalBranchSuccessors += CountBranchSuccessors;
  TotalSwitchSuccessors += CountSwitchSuccessors;

  return PreservedAnalyses::all();
}