//===- ModuleClone.cpp - Yk Module Cloning Pass ---------------------------===//
//
// This pass duplicates functions within the module, producing both the
// original (unoptimised) and cloned (optimised) versions of these
// functions. The process is as follows:
//
// - **Cloning Criteria:**
//   - Functions **without** their address taken are cloned. This results in two
//     versions of such functions in the module: the original and the cloned
//     version.
//   - Functions **with** their address taken remain only in their original form
//     and are not cloned.
//
// - **Cloned Function Naming:**
//   - The cloned functions are renamed by adding the prefix `__yk_unopt_` to
//     their original names. This distinguishes them from the original
//     functions.
//
// - **Optimisation Intent:**
//   - The **cloned functions** (those with the `__yk_unopt_` prefix) are
//     intended to be the **unoptimised versions** of the functions.
//   - The **original functions** remain **optimised**.
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

  /**
   * @brief Clones eligible functions within the given module.
   *
   * This function iterates over all functions in the provided LLVM module `M`
   * and clones those that meet the following criteria:
   *
   * - The function does **not** have external linkage and is **not** a
   * declaration.
   * - The function's address is **not** taken.
   *
   * @param M The LLVM module containing the functions to be cloned.
   * @return A map where the keys are the original function names and the
   *         values are pointers to the cloned `Function` objects. Returns
   *         a map if cloning succeeds, or `nullopt` if cloning fails.
   */
  std::optional<std::map<std::string, Function *>>
  cloneFunctionsInModule(Module &M) {
    LLVMContext &Context = M.getContext();
    std::map<std::string, Function *> ClonedFuncs;
    for (llvm::Function &F : M) {
      // Skip external declarations.
      if (F.hasExternalLinkage() && F.isDeclaration()) {
        continue;
      }
      // Skip already cloned functions or functions with address taken.
      if (F.hasAddressTaken() || F.getName().startswith(YK_UNOPT_PREFIX)) {
        continue;
      }
      ValueToValueMapTy VMap;
      Function *ClonedFunc = CloneFunction(&F, VMap);
      if (ClonedFunc == nullptr) {
        Context.emitError("Failed to clone function: " + F.getName());
        return std::nullopt;
      }
      // Copy arguments
      auto DestArgIt = ClonedFunc->arg_begin();
      for (const Argument &OrigArg : F.args()) {
        DestArgIt->setName(OrigArg.getName());
        VMap[&OrigArg] = &*DestArgIt++;
      }
      // Rename function
      auto originalName = F.getName().str();
      auto cloneName = Twine(YK_UNOPT_PREFIX) + originalName;
      ClonedFunc->setName(cloneName);
      ClonedFuncs[originalName] = ClonedFunc;
    }
    return ClonedFuncs;
  }

  /**
   * @brief Updates call instructions in cloned functions to reference
   * other cloned functions.
   *
   * @param M The LLVM module containing the functions.
   * @param ClonedFuncs A map of cloned function names to functions.
   */
  void
  updateClonedFunctionCalls(Module &M,
                            std::map<std::string, Function *> &ClonedFuncs) {
    for (auto &Entry : ClonedFuncs) {
      Function *ClonedFunc = Entry.second;
      for (BasicBlock &BB : *ClonedFunc) {
        for (Instruction &I : BB) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            Function *CalledFunc = Call->getCalledFunction();
            if (CalledFunc && !CalledFunc->isIntrinsic()) {
              std::string CalledName = CalledFunc->getName().str();
              auto It = ClonedFuncs.find(CalledName);
              if (It != ClonedFuncs.end()) {
                Call->setCalledFunction(It->second);
              }
            }
          }
        }
      }
    }
  }

  /**
   * This function iterates over all functions in the `FinalModule`.
   * If cloned function calls are identified within the original function
   * instructions, they are redirected to the original function instead.
   *
   * **Example Scenario:**
   * - Function `f` calls function `g`.
   * - Function `g` is cloned as `__yk_clone_g`.
   * - Function `f` is not cloned because its address is taken.
   * - As a result, function `f` calls `__yk_clone_g` instead of `g`.
   *
   * **Reasoning:**
   * In `YkIRWriter` we only serialise non-cloned functions.
   *
   * @param FinalModule The module containing both original and cloned
   * functions.
   */
  void updateFunctionCalls(Module &FinalModule) {
    for (Function &F : FinalModule) {
      if (F.getName().startswith(YK_CLONE_PREFIX)) {
        continue;
      }
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (CallInst *CI = dyn_cast<CallInst>(&I)) {
            Function *CalledFunc = CI->getCalledFunction();
            if (CalledFunc &&
                CalledFunc->getName().startswith(YK_CLONE_PREFIX)) {
              std::string OriginalName =
                  CalledFunc->getName().str().substr(strlen(YK_CLONE_PREFIX));
              Function *OriginalFunc = FinalModule.getFunction(OriginalName);
              if (OriginalFunc) {
                CI->setCalledFunction(OriginalFunc);
              }
            }
          }
        }
      }
    }
  }

  bool runOnModule(Module &M) override {
    LLVMContext &Context = M.getContext();
    auto clonedFunctions = cloneFunctionsInModule(M);
    if (!clonedFunctions) {
      Context.emitError("Failed to clone functions in module");
      return false;
    }
    updateClonedFunctionCalls(M, *clonedFunctions);

    if (verifyModule(M, &errs())) {
      Context.emitError("Module verification failed!");
      return false;
    }

    return true;
  }
};
} // namespace

char YkModuleClone::ID = 0;

INITIALIZE_PASS(YkModuleClone, DEBUG_TYPE, "yk module clone", false, false)

ModulePass *llvm::createYkModuleClonePass() { return new YkModuleClone(); }
