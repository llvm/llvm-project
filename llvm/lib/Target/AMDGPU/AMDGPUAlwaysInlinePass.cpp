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
/// duplicates of all other functions a marks the duplicates as always_inline.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

namespace {

class AMDGPUAlwaysInline : public ModulePass {
  static char ID;

  bool GlobalOpt;

public:
  AMDGPUAlwaysInline(bool GlobalOpt) : ModulePass(ID), GlobalOpt(GlobalOpt) { }
  bool runOnModule(Module &M) override;
  StringRef getPassName() const override { return "AMDGPU Always Inline Pass"; }
};

} // End anonymous namespace

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

  for (Function &F : M) {
    if (!F.hasLocalLinkage() && !F.isDeclaration() && !F.use_empty() &&
        !F.hasFnAttribute(Attribute::NoInline))
      FuncsToClone.push_back(&F);
  }

  for (Function *F : FuncsToClone) {
    ValueToValueMapTy VMap;
    Function *NewFunc = CloneFunction(F, VMap);
    NewFunc->setLinkage(GlobalValue::InternalLinkage);
    F->replaceAllUsesWith(NewFunc);
  }

  for (Function &F : M) {
    if (F.hasLocalLinkage() && !F.hasFnAttribute(Attribute::NoInline)) {
      F.addFnAttr(Attribute::AlwaysInline);
    }
  }
  return false;
}

ModulePass *llvm::createAMDGPUAlwaysInlinePass(bool GlobalOpt) {
  return new AMDGPUAlwaysInline(GlobalOpt);
}
