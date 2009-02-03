//===-- InstCount.cpp - Collects the count of all instructions ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass collects the count of all instructions and reports them
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instcount"
#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/Statistic.h"
#include <ostream>
using namespace llvm;

STATISTIC(TotalInsts , "Number of instructions (of all types)");
STATISTIC(TotalBlocks, "Number of basic blocks");
STATISTIC(TotalFuncs , "Number of non-external functions");
STATISTIC(TotalMemInst, "Number of memory instructions");

#define HANDLE_INST(N, OPCODE, CLASS) \
  STATISTIC(Num ## OPCODE ## Inst, "Number of " #OPCODE " insts");

#include "llvm/Instruction.def"


namespace {
  class VISIBILITY_HIDDEN InstCount 
      : public FunctionPass, public InstVisitor<InstCount> {
    friend class InstVisitor<InstCount>;

    void visitFunction  (Function &F) { ++TotalFuncs; }
    void visitBasicBlock(BasicBlock &BB) { ++TotalBlocks; }

#define HANDLE_INST(N, OPCODE, CLASS) \
    void visit##OPCODE(CLASS &) { ++Num##OPCODE##Inst; ++TotalInsts; }

#include "llvm/Instruction.def"

    void visitInstruction(Instruction &I) {
      cerr << "Instruction Count does not know about " << I;
      abort();
    }
  public:
    static char ID; // Pass identification, replacement for typeid
    InstCount() : FunctionPass(&ID) {}

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
    virtual void print(std::ostream &O, const Module *M) const {}

  };
}

char InstCount::ID = 0;
static RegisterPass<InstCount>
X("instcount", "Counts the various types of Instructions", false, true);

FunctionPass *llvm::createInstCountPass() { return new InstCount(); }

// InstCount::run - This is the main Analysis entry point for a
// function.
//
bool InstCount::runOnFunction(Function &F) {
  unsigned StartMemInsts =
    NumGetElementPtrInst + NumLoadInst + NumStoreInst + NumCallInst +
    NumInvokeInst + NumAllocaInst + NumMallocInst + NumFreeInst;
  visit(F);
  unsigned EndMemInsts =
    NumGetElementPtrInst + NumLoadInst + NumStoreInst + NumCallInst +
    NumInvokeInst + NumAllocaInst + NumMallocInst + NumFreeInst;
  TotalMemInst += EndMemInsts-StartMemInsts;
  return false;
}
