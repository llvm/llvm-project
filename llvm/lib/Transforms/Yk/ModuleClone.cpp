//===- ModuleClone.cpp - Yk Module Cloning Pass ---------------------------===//
//
// This pass duplicates functions within the module, producing both the original
// and new versions of functions. the process is as follows:
//
// - **Cloning Criteria:**
//   - Functions **without** their address taken are cloned. This results in two
//     versions of such functions in the module: the original and the cloned
//     version.
//   - Functions **with** their address taken remain only in their original form
//     and are not cloned.
//
// - **Cloned Function Naming:**
//   - The cloned functions are renamed by adding the prefix `__yk_clone_` to
//     their original names. This distinguishes them from the original
//     functions.
//
// - **Clone Process:**
//   - 1. The original module is cloned, creating two copies of the module.
//   - 2. The functions in the cloned module that satisfy the cloning criteria
//     are renamed.
//   - 3. The cloned module is linked back into the original module.
//
// - **Optimisation Intent:**
//   - The **cloned functions** (those with the `__yk_clone_` prefix) are
//     intended to be the **optimised versions** of the functions.
//   - The **original functions** remain **unoptimised**.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Yk/ModuleClone.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"

#define DEBUG_TYPE "yk-module-clone-pass"

using namespace llvm;

namespace llvm {
void initializeYkModuleClonePass(PassRegistry &);
} // namespace llvm

namespace {
struct YkModuleClone : public ModulePass {
  static char ID;

  YkModuleClone() : ModulePass(ID) {
    initializeYkModuleClonePass(*PassRegistry::getPassRegistry());
  }
  void updateClonedFunctions(Module &M) {
    for (llvm::Function &F : M) {
      if (F.hasExternalLinkage() && F.isDeclaration()) {
        continue;
      }
      // Skip functions that are address taken
      if (!F.hasAddressTaken()) {
        F.setName(Twine(YK_CLONE_PREFIX) + F.getName());
      }
    }
  }

  bool runOnModule(Module &M) override {
    std::unique_ptr<Module> Cloned = CloneModule(M);
    if (!Cloned) {
      llvm::report_fatal_error("Error cloning the module");
      return false;
    }
    updateClonedFunctions(*Cloned);

    // The `OverrideFromSrc` flag instructs the linker to prioritise
    // definitions from the source module (the second argument) when
    // conflicts arise. This means that if two global variables, functions,
    // or constants have the same name in both the original and cloned modules,
    // the version from the cloned module will overwrite the original.
    if (Linker::linkModules(M, std::move(Cloned),
                            Linker::Flags::OverrideFromSrc)) {
      llvm::report_fatal_error("Error linking the modules");
      return false;
    }

    if (verifyModule(M, &errs())) {
      errs() << "Module verification failed!";
      return false;
    }

    return true;
  }
};
} // namespace

char YkModuleClone::ID = 0;

INITIALIZE_PASS(YkModuleClone, DEBUG_TYPE, "yk module clone", false, false)

ModulePass *llvm::createYkModuleClonePass() { return new YkModuleClone(); }
