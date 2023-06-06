#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  struct P2Comp : public FunctionPass {
    static char ID;
    P2Comp() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
      unsigned int count = 0;

      for (Function::const_iterator BB = F.begin(); BB != F.end(); ++BB) {
        for (BasicBlock::const_iterator Ins = BB->begin(); Ins != BB->end(); ++Ins) {
          const Instruction *I = &*Ins;
          if (isa<CmpInst>(I)) {
            ++count;
          }
        }
      }
      errs() << "Number of Comparison instructions: " << count << "\n";
      return false;
    }
  };
}

char P2Comp::ID = 0;
static RegisterPass<P2Comp> X("p2comp", "Count Comparison Instructions");

namespace {
  struct P2Mul : public FunctionPass {
    static char ID;
    P2Mul() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
      unsigned int count = 0;

      for (Function::const_iterator BB = F.begin(); BB != F.end(); ++BB) {
        for (BasicBlock::const_iterator Ins = BB->begin(); Ins != BB->end(); ++Ins) {
          const Instruction *I = &*Ins;
          if (I->getOpcode() == Instruction::Mul) { // From: llvm/IR/Instruction.def
            ++count;
          }
        }
      }
      errs() << "Number of Multiply instructions: " << count << "\n";
      return false;
    }
  };
}

char P2Mul::ID = 0;
static RegisterPass<P2Mul> Y("p2mul", "Count Multiply Instructions");
