//===-- ElimAvailExtern.cpp - DCE unreachable internal functions
//----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transform is designed to eliminate available external global
// definitions from the program, turning them into declarations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Pass.h"
using namespace llvm;

#define DEBUG_TYPE "elim-avail-extern"

STATISTIC(NumFunctions, "Number of functions removed");
STATISTIC(NumVariables, "Number of global variables removed");

namespace {
struct EliminateAvailableExternally : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  EliminateAvailableExternally() : ModulePass(ID) {
    initializeEliminateAvailableExternallyPass(
        *PassRegistry::getPassRegistry());
  }

  // run - Do the EliminateAvailableExternally pass on the specified module,
  // optionally updating the specified callgraph to reflect the changes.
  //
  bool runOnModule(Module &M) override;
};
}

char EliminateAvailableExternally::ID = 0;
INITIALIZE_PASS(EliminateAvailableExternally, "elim-avail-extern",
                "Eliminate Available Externally Globals", false, false)

ModulePass *llvm::createEliminateAvailableExternallyPass() {
  return new EliminateAvailableExternally();
}

static void convertAliasToDeclaration(GlobalAlias &GA, Module &M) {
  GlobalValue *GVal = GA.getBaseObject();
  GlobalValue *NewGV;
  if (auto *GVar = dyn_cast<GlobalVariable>(GVal)) {
    GlobalVariable *NewGVar = new GlobalVariable(
        M, GVar->getType()->getElementType(), GVar->isConstant(),
        GVar->getLinkage(), /*init*/ nullptr, GA.getName(), GVar,
        GVar->getThreadLocalMode(), GVar->getType()->getAddressSpace());
    NewGV = NewGVar;
    NewGV->copyAttributesFrom(GVar);
  } else {
    auto *F = dyn_cast<Function>(GVal);
    assert(F);
    Function *NewF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                      GA.getName(), &M);
    NewGV = NewF;
    NewGV->copyAttributesFrom(F);
  }
  GA.replaceAllUsesWith(ConstantExpr::getBitCast(NewGV, GA.getType()));
  GA.eraseFromParent();
}

bool EliminateAvailableExternally::runOnModule(Module &M) {
  bool Changed = false;

  // Convert any aliases that alias with an available externally
  // value (which will be turned into declarations later on in this routine)
  // into declarations themselves. All aliases must be definitions, and
  // must alias with a definition. So this involves creating a declaration
  // equivalent to the alias's base object.
  for (Module::alias_iterator I = M.alias_begin(), E = M.alias_end(); I != E;) {
    // Increment the iterator first since we may delete the current alias.
    GlobalAlias &GA = *(I++);
    GlobalValue *GVal = GA.getBaseObject();
    if (!GVal->hasAvailableExternallyLinkage())
      continue;
    convertAliasToDeclaration(GA, M);
    Changed = true;
  }

  // Drop initializers of available externally global variables.
  for (GlobalVariable &GV : M.globals()) {
    if (!GV.hasAvailableExternallyLinkage())
      continue;
    if (GV.hasInitializer()) {
      Constant *Init = GV.getInitializer();
      GV.setInitializer(nullptr);
      if (isSafeToDestroyConstant(Init))
        Init->destroyConstant();
    }
    GV.removeDeadConstantUsers();
    GV.setLinkage(GlobalValue::ExternalLinkage);
    NumVariables++;
    Changed = true;
  }

  // Drop the bodies of available externally functions.
  for (Function &F : M) {
    if (!F.hasAvailableExternallyLinkage())
      continue;
    if (!F.isDeclaration())
      // This will set the linkage to external
      F.deleteBody();
    F.removeDeadConstantUsers();
    NumFunctions++;
    Changed = true;
  }

  return Changed;
}
