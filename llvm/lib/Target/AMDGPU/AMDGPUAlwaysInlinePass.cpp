//===-- AMDGPUAlwaysInlinePass.cpp - Promote Allocas ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass marks all internal functions as always_inline and creates
/// duplicates of all other functions and marks the duplicates as always_inline.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

namespace {

static cl::opt<bool> StressCalls(
  "amdgpu-stress-function-calls",
  cl::Hidden,
  cl::desc("Force all functions to be noinline"),
  cl::init(false));

class AMDGPUAlwaysInline : public ModulePass {
  bool GlobalOpt;

public:
  static char ID;

  AMDGPUAlwaysInline(bool GlobalOpt = false) :
    ModulePass(ID), GlobalOpt(GlobalOpt) { }
  bool runOnModule(Module &M) override;
  StringRef getPassName() const override { return "AMDGPU Always Inline Pass"; }
};

} // End anonymous namespace

INITIALIZE_PASS(AMDGPUAlwaysInline, "amdgpu-always-inline",
                "AMDGPU Inline All Functions", false, false)

char AMDGPUAlwaysInline::ID = 0;

bool AMDGPUAlwaysInline::runOnModule(Module &M) {
  std::vector<GlobalAlias*> AliasesToRemove;
  std::vector<Function *> FuncsToClone;

  for (GlobalAlias &A : M.aliases()) {
    if (Function* F = dyn_cast<Function>(A.getAliasee())) {
      A.replaceAllUsesWith(F);
      AliasesToRemove.push_back(&A);
    }
  }

  if (GlobalOpt) {
    for (GlobalAlias* A : AliasesToRemove) {
      A->eraseFromParent();
    }
  }

  auto NewAttr = StressCalls ? Attribute::NoInline : Attribute::AlwaysInline;
  auto IncompatAttr
    = StressCalls ? Attribute::AlwaysInline : Attribute::NoInline;

  for (Function &F : M) {
    if (!F.hasLocalLinkage() && !F.isDeclaration() && !F.use_empty() &&
        !F.hasFnAttribute(IncompatAttr))
      FuncsToClone.push_back(&F);
  }

  for (Function *F : FuncsToClone) {
    ValueToValueMapTy VMap;
    Function *NewFunc = CloneFunction(F, VMap);
    NewFunc->setLinkage(GlobalValue::InternalLinkage);
    F->replaceAllUsesWith(NewFunc);
  }

  for (Function &F : M) {
    if (F.hasLocalLinkage() && !F.hasFnAttribute(IncompatAttr)) {
      F.addFnAttr(NewAttr);
    }
  }
  return false;
}

ModulePass *llvm::createAMDGPUAlwaysInlinePass(bool GlobalOpt) {
  return new AMDGPUAlwaysInline(GlobalOpt);
}
