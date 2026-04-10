//===-- InstCount.cpp - Collects the count of all instructions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass collects the count of all instructions and reports them
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InstCount.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "instcount"

STATISTIC(TotalInstsPrePasses,
          "Number of instructions of all types (before passes)");
STATISTIC(TotalInsts, "Number of instructions of all types");
STATISTIC(TotalBlocksPrePasses, "Number of basic blocks (before passes)");
STATISTIC(TotalBlocks, "Number of basic blocks");
STATISTIC(TotalFuncsPrePasses,
          "Number of non-external functions (before passes)");
STATISTIC(TotalFuncs, "Number of non-external functions");
STATISTIC(LargestFunctionSizePrePasses,
          "Largest number of instructions in a single function (before "
          "passes)");
STATISTIC(LargestFunctionSize,
          "Largest number of instructions in a single function");
STATISTIC(LargestFunctionBBCountPrePasses,
          "Largest number of basic blocks in a single function (before "
          "passes)");
STATISTIC(LargestFunctionBBCount,
          "Largest number of basic blocks in a single function");

#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  STATISTIC(Num##OPCODE##InstPrePasses,                                        \
            "Number of " #OPCODE " insts (before passes)");                    \
  STATISTIC(Num##OPCODE##Inst, "Number of " #OPCODE " insts");

#include "llvm/IR/Instruction.def"

namespace {
class InstCount : public InstVisitor<InstCount> {
  friend class InstVisitor<InstCount>;
  bool IsPrePasses;

public:
  InstCount(bool IsPrePasses) : IsPrePasses(IsPrePasses) {}

  void visitFunction(Function &F) {
    if (IsPrePasses) {
      ++TotalFuncsPrePasses;
      LargestFunctionSizePrePasses.updateMax(F.getInstructionCount());
      LargestFunctionBBCountPrePasses.updateMax(F.size());
    } else {
      ++TotalFuncs;
      LargestFunctionSize.updateMax(F.getInstructionCount());
      LargestFunctionBBCount.updateMax(F.size());
    }
  }

  void visitBasicBlock(BasicBlock &BB) {
    if (IsPrePasses)
      ++TotalBlocksPrePasses;
    else
      ++TotalBlocks;
  }

#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  void visit##OPCODE(CLASS &) {                                                \
    if (IsPrePasses) {                                                         \
      ++Num##OPCODE##InstPrePasses;                                            \
      ++TotalInstsPrePasses;                                                   \
    } else {                                                                   \
      ++Num##OPCODE##Inst;                                                     \
      ++TotalInsts;                                                            \
    }                                                                          \
  }

#include "llvm/IR/Instruction.def"

  void visitInstruction(Instruction &I) {
    errs() << "Instruction Count does not know about " << I;
    llvm_unreachable(nullptr);
  }
};
} // namespace

PreservedAnalyses InstCountPass::run(Function &F,
                                     FunctionAnalysisManager &FAM) {
  LLVM_DEBUG(dbgs() << "INSTCOUNT: running on function " << F.getName()
                    << "\n");
  InstCount(this->IsPrePasses).visit(F);

  return PreservedAnalyses::all();
}
