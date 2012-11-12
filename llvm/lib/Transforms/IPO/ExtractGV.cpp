//===-- ExtractGV.cpp - Global Value extraction pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass extracts global values
//
//===----------------------------------------------------------------------===//

#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Constants.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>
using namespace llvm;

namespace {
  /// @brief A pass to extract specific functions and their dependencies.
  class GVExtractorPass : public ModulePass {
    SetVector<GlobalValue *> Named;
    bool deleteStuff;
  public:
    static char ID; // Pass identification, replacement for typeid

    /// FunctionExtractorPass - If deleteFn is true, this pass deletes as the
    /// specified function. Otherwise, it deletes as much of the module as
    /// possible, except for the function specified.
    ///
    explicit GVExtractorPass(std::vector<GlobalValue*>& GVs, bool deleteS = true)
      : ModulePass(ID), Named(GVs.begin(), GVs.end()), deleteStuff(deleteS) {}

    bool runOnModule(Module &M) {
      // Visit the global inline asm.
      if (!deleteStuff)
        M.setModuleInlineAsm("");

      // For simplicity, just give all GlobalValues ExternalLinkage. A trickier
      // implementation could figure out which GlobalValues are actually
      // referenced by the Named set, and which GlobalValues in the rest of
      // the module are referenced by the NamedSet, and get away with leaving
      // more internal and private things internal and private. But for now,
      // be conservative and simple.

      // Visit the GlobalVariables.
      for (Module::global_iterator I = M.global_begin(), E = M.global_end();
           I != E; ++I) {
        bool Delete =
          deleteStuff == (bool)Named.count(I) && !I->isDeclaration();
        if (!Delete) {
          if (I->hasAvailableExternallyLinkage())
            continue;
          if (I->getName() == "llvm.global_ctors")
            continue;
        }

        bool Local = I->hasLocalLinkage();
        if (Local)
          I->setVisibility(GlobalValue::HiddenVisibility);

        if (Local || Delete)
          I->setLinkage(GlobalValue::ExternalLinkage);

        if (Delete)
          I->setInitializer(0);
      }

      // Visit the Functions.
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
        bool Delete =
          deleteStuff == (bool)Named.count(I) && !I->isDeclaration();
        if (!Delete) {
          if (I->hasAvailableExternallyLinkage())
            continue;
        }

        bool Local = I->hasLocalLinkage();
        if (Local)
          I->setVisibility(GlobalValue::HiddenVisibility);

        if (Local || Delete)
          I->setLinkage(GlobalValue::ExternalLinkage);

        if (Delete)
          I->deleteBody();
      }

      // Visit the Aliases.
      for (Module::alias_iterator I = M.alias_begin(), E = M.alias_end();
           I != E;) {
        Module::alias_iterator CurI = I;
        ++I;

        if (CurI->hasLocalLinkage()) {
          CurI->setVisibility(GlobalValue::HiddenVisibility);
          CurI->setLinkage(GlobalValue::ExternalLinkage);
        }

        if (deleteStuff == (bool)Named.count(CurI)) {
          Type *Ty =  CurI->getType()->getElementType();

          CurI->removeFromParent();
          llvm::Value *Declaration;
          if (FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
            Declaration = Function::Create(FTy, GlobalValue::ExternalLinkage,
                                           CurI->getName(), &M);

          } else {
            Declaration =
              new GlobalVariable(M, Ty, false, GlobalValue::ExternalLinkage,
                                 0, CurI->getName());

          }
          CurI->replaceAllUsesWith(Declaration);
          delete CurI;
        }
      }

      return true;
    }
  };

  char GVExtractorPass::ID = 0;
}

ModulePass *llvm::createGVExtractionPass(std::vector<GlobalValue*>& GVs, 
                                         bool deleteFn) {
  return new GVExtractorPass(GVs, deleteFn);
}
