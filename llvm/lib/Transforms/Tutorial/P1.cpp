#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  struct P1Function : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    P1Function() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
      errs() << "Hello: ";
      errs() << F.getName() << '\n';
      return false; // Return true for transformation passes; otherwise return false
    }
  };
}

char P1Function::ID = 0;
static RegisterPass<P1Function> X("p1function", "Hello World Function Pass");

namespace {
  struct P1Module : public ModulePass {
    static char ID;
    P1Module() : ModulePass(ID) {}

    bool runOnModule(Module &M) override {
      for (Module::const_iterator It = M.begin(); It != M.end(); ++It) {
        const Function *F = &*It;
        errs() << "Hello: ";
        errs() << F->getName() << '\n';
      }
      return false;
    }
  };
}

char P1Module::ID = 0;
static RegisterPass<P1Module> Y("p1module", "Hello World Module Pass");
