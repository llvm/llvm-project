//===- Linkage.cpp - Ajdust linkage for the Yk JIT -----------------===//
//
// The JIT relies upon the use of `dlsym()` at runtime in order to lookup any
// given function from its virtual address. For this to work the symbols for
// all functions must be in the dynamic symbol table.
//
// `yk-config` already provides the `--export-dynamic` flag in order to ensure
// that all *externally visible* symbols make it in to the dynamic symbol table,
// but that's not enough: functions marked for internal linkage (e.g. `static`
// functions in C) will be missed.
//
// This pass walks the functions of a module and flips any with internal linkage
// to external linkage.
//
// Note that whilst symbols with internal linkage can have the same name and be
// distinct, this is not so for symbols with external linkage. That's OK for
// us because Yk requires the use of LTO, and the LTO module merger will have
// already mangled the names for us so that symbol clashes can't occur.

#include "llvm/Transforms/Yk/Linkage.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "yk-linkage"

using namespace llvm;

namespace llvm {
void initializeYkLinkagePass(PassRegistry &);
} // namespace llvm

namespace {
class YkLinkage : public ModulePass {
public:
  static char ID;
  YkLinkage() : ModulePass(ID) {
    initializeYkLinkagePass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    for (Function &F : M) {
      if (F.hasInternalLinkage()) {
        F.setLinkage(GlobalVariable::ExternalLinkage);
      }
    }
    return true;
  }
};
} // namespace

char YkLinkage::ID = 0;
INITIALIZE_PASS(YkLinkage, DEBUG_TYPE, "yk-linkage", false, false)

ModulePass *llvm::createYkLinkagePass() { return new YkLinkage(); }
