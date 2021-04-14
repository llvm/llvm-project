//===- InferFunctionAttrs.cpp - Infer implicit function attributes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
using namespace llvm;

#define DEBUG_TYPE "inferattrs"

/// If we can infer one attribute from another on the declaration of a
/// function, explicitly materialize the maximal set for readability in the IR.
/// Doing this also allows our CGSCC inference to avoid needing to duplicate
/// this logic on all calls to declarations (as declarations aren't explicitly
/// visited by CGSCC passes in the new pass manager.)
static bool inferAttributesFromOthers(Function &F) {
  // Note: We explicitly check for attributes rather than using cover functions
  // because some of the cover functions include the logic being implemented.

  bool Changed = false;
  // readnone + not convergent implies nosync
  if (!F.hasFnAttribute(Attribute::NoSync) &&
      F.doesNotAccessMemory() && !F.isConvergent()) {
    F.setNoSync();
    Changed = true;
  }

  // readonly implies nofree
  if (!F.hasFnAttribute(Attribute::NoFree) && F.onlyReadsMemory()) {
    F.setDoesNotFreeMemory();
    Changed = true;
  }

  // willreturn implies mustprogress
  if (!F.hasFnAttribute(Attribute::MustProgress) && F.willReturn()) {
    F.setMustProgress();
    Changed = true;
  }

  // TODO: There are a bunch of cases of restrictive memory effects we
  // can infer by inspecting arguments of argmemonly-ish functions.

  return Changed;
}

static bool inferAllPrototypeAttributes(
    Module &M, function_ref<TargetLibraryInfo &(Function &)> GetTLI) {
  bool Changed = false;

  for (Function &F : M.functions())
    // We only infer things using the prototype and the name; we don't need
    // definitions.
    if (F.isDeclaration() && !F.hasOptNone()) {
      Changed |= inferLibFuncAttributes(F, GetTLI(F));
      Changed |= inferAttributesFromOthers(F);
    }

  return Changed;
}

PreservedAnalyses InferFunctionAttrsPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTLI = [&FAM](Function &F) -> TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };

  if (!inferAllPrototypeAttributes(M, GetTLI))
    // If we didn't infer anything, preserve all analyses.
    return PreservedAnalyses::all();

  // Otherwise, we may have changed fundamental function attributes, so clear
  // out all the passes.
  return PreservedAnalyses::none();
}

namespace {
struct InferFunctionAttrsLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  InferFunctionAttrsLegacyPass() : ModulePass(ID) {
    initializeInferFunctionAttrsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    auto GetTLI = [this](Function &F) -> TargetLibraryInfo & {
      return this->getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    };
    return inferAllPrototypeAttributes(M, GetTLI);
  }
};
}

char InferFunctionAttrsLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(InferFunctionAttrsLegacyPass, "inferattrs",
                      "Infer set function attributes", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(InferFunctionAttrsLegacyPass, "inferattrs",
                    "Infer set function attributes", false, false)

Pass *llvm::createInferFunctionAttrsLegacyPass() {
  return new InferFunctionAttrsLegacyPass();
}
