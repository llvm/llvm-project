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
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "instcount"

STATISTIC(TotalInsts, "Number of instructions (of all types)");
STATISTIC(TotalBlocks, "Number of basic blocks");
STATISTIC(TotalFuncs, "Number of non-external functions");

#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  STATISTIC(Num##OPCODE##Inst, "Number of " #OPCODE " insts");

#include "llvm/IR/Instruction.def"

namespace {
class InstCount : public InstVisitor<InstCount> {
  friend class InstVisitor<InstCount>;

  void visitFunction(Function &F) { ++TotalFuncs; }
  void visitBasicBlock(BasicBlock &BB) { ++TotalBlocks; }

#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  void visit##OPCODE(CLASS &) {                                                \
    ++Num##OPCODE##Inst;                                                       \
    ++TotalInsts;                                                              \
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
  InstCount().visit(F);

  return PreservedAnalyses::all();
}

namespace {
class InstCountLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  InstCountLegacyPass() : FunctionPass(ID) {
    initializeInstCountLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    LLVM_DEBUG(dbgs() << "INSTCOUNT: running on function " << F.getName()
                      << "\n");
    InstCount().visit(F);
    return false;
  };

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  void print(raw_ostream &O, const Module *M) const override {}
};
} // namespace

char InstCountLegacyPass::ID = 0;
INITIALIZE_PASS(InstCountLegacyPass, "instcount",
                "Counts the various types of Instructions", false, true)

FunctionPass *llvm::createInstCountPass() { return new InstCountLegacyPass(); }
