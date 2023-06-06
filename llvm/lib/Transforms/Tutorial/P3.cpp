#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  struct P3Comp : public FunctionPass {
    static char ID;
    P3Comp() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
      for (Function::const_iterator BB = F.begin(); BB != F.end(); ++BB) {
        for (BasicBlock::const_iterator Ins = BB->begin(); Ins != BB->end(); ++Ins) {
          const Instruction *I = &*Ins;

          // Better to dynamically cast and check types in one operation in LLVM
          if (const CmpInst *CI = dyn_cast<CmpInst>(I)) {
            // Print out the instruction itself
            errs() << "Ins: ";
            CI->print(errs());
            errs() << "\n";

            // Print first source operand
            errs() << "1st Operand: ";
            CI->getOperand(0)->print(errs());
            errs() << "\n";

            // Print second source operand
            errs() << "2nd Operand: ";
            CI->getOperand(1)->print(errs());
            errs() << "\n\n";
          }
        }
      }
      return false;
    }
  };
}

char P3Comp::ID = 0;
static RegisterPass<P3Comp> X("p3comp", "Print Comp Instruction Operands");
