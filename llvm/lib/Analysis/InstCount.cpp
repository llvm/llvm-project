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

STATISTIC(TotalInstsPreOptimizations,
          "Number of instructions of all types (before optimizations)");
STATISTIC(TotalInsts,
          "Number of instructions of all types (after optimizations)");
STATISTIC(TotalBlocksPreOptimizations,
          "Number of basic blocks (before optimizations)");
STATISTIC(TotalBlocks, "Number of basic blocks (after optimizations)");
STATISTIC(TotalFuncsPreOptimizations,
          "Number of non-external functions (before optimizations)");
STATISTIC(TotalFuncs, "Number of non-external functions (after optimizations)");
STATISTIC(LargestFunctionSizePreOptimizations,
          "Largest number of instructions in a single function (before "
          "optimizations)");
STATISTIC(LargestFunctionSize,
          "Largest number of instructions in a single function (after "
          "optimizations)");
STATISTIC(LargestFunctionBBCountPreOptimizations,
          "Largest number of basic blocks in a single function (before "
          "optimizations)");
STATISTIC(LargestFunctionBBCount,
          "Largest number of basic blocks in a single function (after "
          "optimizations)");

#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  STATISTIC(Num##OPCODE##InstPreOptimizations,                                 \
            "Number of " #OPCODE " insts (before optimizations)");             \
  STATISTIC(Num##OPCODE##Inst,                                                 \
            "Number of " #OPCODE " insts (after optimizations)");

#include "llvm/IR/Instruction.def"

namespace {
class InstCount : public InstVisitor<InstCount> {
  friend class InstVisitor<InstCount>;
  bool IsPreOptimizations;

public:
  InstCount(bool IsPreOptimizations) : IsPreOptimizations(IsPreOptimizations) {}

  void visitFunction(Function &F) {
    if (IsPreOptimizations) {
      ++TotalFuncsPreOptimizations;
      LargestFunctionSizePreOptimizations.updateMax(F.getInstructionCount());
      LargestFunctionBBCountPreOptimizations.updateMax(F.size());
    } else {
      ++TotalFuncs;
      LargestFunctionSize.updateMax(F.getInstructionCount());
      LargestFunctionBBCount.updateMax(F.size());
    }
  }

  void visitBasicBlock(BasicBlock &BB) {
    if (IsPreOptimizations)
      ++TotalBlocksPreOptimizations;
    else
      ++TotalBlocks;
  }

#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  void visit##OPCODE(CLASS &) {                                                \
    if (IsPreOptimizations) {                                                  \
      ++Num##OPCODE##InstPreOptimizations;                                     \
      ++TotalInstsPreOptimizations;                                            \
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
  InstCount(this->IsPreOptimizations).visit(F);

  return PreservedAnalyses::all();
}