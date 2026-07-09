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

STATISTIC(TotalInsts, "Number of instructions of all types");
STATISTIC(TotalBlocks, "Number of basic blocks");
STATISTIC(TotalFuncs, "Number of non-external functions");

STATISTIC(LargestFunctionSize,
          "Largest number of instructions in a single function");
STATISTIC(LargestFunctionBBCount,
          "Largest number of basic blocks in a single function");

STATISTIC(TotalInstsPreOptimization,
          "Number of instructions of all types (before optimizations)");
STATISTIC(TotalBlocksPreOptimization,
          "Number of basic blocks (before optimizations)");
STATISTIC(TotalFuncsPreOptimization,
          "Number of non-external functions (before optimizations)");

STATISTIC(LargestFunctionSizePreOptimization,
          "Largest number of instructions in a single function (before "
          "optimizations)");
STATISTIC(LargestFunctionBBCountPreOptimization,
          "Largest number of basic blocks in a single function (before "
          "optimizations)");

#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  STATISTIC(Num##OPCODE##Inst, "Number of " #OPCODE " insts");                 \
  STATISTIC(Num##OPCODE##InstPreOptimization,                                  \
            "Number of " #OPCODE " insts (before optimizations)");

#include "llvm/IR/Instruction.def"

namespace {
class InstCount : public InstVisitor<InstCount> {
  friend class InstVisitor<InstCount>;

  bool IsPreOptimization;

  void visitFunction(Function &F) {
    if (IsPreOptimization) {
      ++TotalFuncsPreOptimization;
      LargestFunctionSizePreOptimization.updateMax(F.getInstructionCount());
      LargestFunctionBBCountPreOptimization.updateMax(F.size());
    } else {
      ++TotalFuncs;
      LargestFunctionSize.updateMax(F.getInstructionCount());
      LargestFunctionBBCount.updateMax(F.size());
    }
  }

  void visitBasicBlock(BasicBlock &BB) {
    if (IsPreOptimization)
      ++TotalBlocksPreOptimization;
    else
      ++TotalBlocks;
  }

#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  void visit##OPCODE(CLASS &) {                                                \
    if (IsPreOptimization) {                                                   \
      ++Num##OPCODE##InstPreOptimization;                                      \
      ++TotalInstsPreOptimization;                                             \
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

public:
  InstCount(bool IsPreOptimization = false)
      : IsPreOptimization(IsPreOptimization) {}
};
} // namespace

PreservedAnalyses InstCountPass::run(Function &F,
                                     FunctionAnalysisManager &FAM) {
  LLVM_DEBUG(dbgs() << "INSTCOUNT: running on function " << F.getName()
                    << "\n");
  InstCount(this->IsPreOptimization).visit(F);

  return PreservedAnalyses::all();
}
