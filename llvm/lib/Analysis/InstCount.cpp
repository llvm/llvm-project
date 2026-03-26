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

STATISTIC(TotalInstsBeforeOptimization,
          "Number of instructions of all types (before optimizations)");
STATISTIC(TotalInstsAfterOptimization,
          "Number of instructions of all types (after optimizations)");
STATISTIC(TotalBlocksBeforeOptimization,
          "Number of basic blocks (before optimizations)");
STATISTIC(TotalBlocksAfterOptimization,
          "Number of basic blocks (after optimizations)");
STATISTIC(TotalFuncsBeforeOptimization,
          "Number of non-external functions (before optimizations)");
STATISTIC(TotalFuncsAfterOptimization,
          "Number of non-external functions (after optimizations)");
STATISTIC(LargestFunctionSizeBeforeOptimization,
          "Largest number of instructions in a single function (before "
          "optimizations)");
STATISTIC(LargestFunctionSizeAfterOptimization,
          "Largest number of instructions in a single function (after "
          "optimizations)");
STATISTIC(LargestFunctionBBCountBeforeOptimization,
          "Largest number of basic blocks in a single function (before "
          "optimizations)");
STATISTIC(LargestFunctionBBCountAfterOptimization,
          "Largest number of basic blocks in a single function (after "
          "optimizations)");

#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  STATISTIC(Num##OPCODE##InstBeforeOptimization,                               \
            "Number of " #OPCODE " insts (before optimizations)");             \
  STATISTIC(Num##OPCODE##InstAfterOptimization,                                \
            "Number of " #OPCODE " insts (after optimizations)");

#include "llvm/IR/Instruction.def"

namespace {
class InstCount : public InstVisitor<InstCount> {
  friend class InstVisitor<InstCount>;
  bool IsBeforeOptimization;

public:
  InstCount(bool IsBeforeOptimization)
      : IsBeforeOptimization(IsBeforeOptimization) {}

  void visitFunction(Function &F) {
    if (IsBeforeOptimization) {
      ++TotalFuncsBeforeOptimization;
      LargestFunctionSizeBeforeOptimization.updateMax(F.getInstructionCount());
      LargestFunctionBBCountBeforeOptimization.updateMax(F.size());
    } else {
      ++TotalFuncsAfterOptimization;
      LargestFunctionSizeAfterOptimization.updateMax(F.getInstructionCount());
      LargestFunctionBBCountAfterOptimization.updateMax(F.size());
    }
  }

  void visitBasicBlock(BasicBlock &BB) {
    if (IsBeforeOptimization)
      ++TotalBlocksBeforeOptimization;
    else
      ++TotalBlocksAfterOptimization;
  }

#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  void visit##OPCODE(CLASS &) {                                                \
    if (IsBeforeOptimization) {                                                \
      ++Num##OPCODE##InstBeforeOptimization;                                   \
      ++TotalInstsBeforeOptimization;                                          \
    } else {                                                                   \
      ++Num##OPCODE##InstAfterOptimization;                                    \
      ++TotalInstsAfterOptimization;                                           \
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
  InstCount(this->IsBeforeOptimization).visit(F);

  return PreservedAnalyses::all();
}