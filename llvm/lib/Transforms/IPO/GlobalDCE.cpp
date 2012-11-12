//===-- GlobalDCE.cpp - DCE unreachable internal functions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transform is designed to eliminate unreachable internal globals from the
// program.  It uses an aggressive algorithm, searching out globals that are
// known to be alive.  After it finds all of the globals which are needed, it
// deletes whatever is left over.  This allows it to delete recursive chunks of
// the program which are unreachable.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "globaldce"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumAliases  , "Number of global aliases removed");
STATISTIC(NumFunctions, "Number of functions removed");
STATISTIC(NumVariables, "Number of global variables removed");

namespace {
  struct GlobalDCE : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    GlobalDCE() : ModulePass(ID) {
      initializeGlobalDCEPass(*PassRegistry::getPassRegistry());
    }

    // run - Do the GlobalDCE pass on the specified module, optionally updating
    // the specified callgraph to reflect the changes.
    //
    bool runOnModule(Module &M);

  private:
    SmallPtrSet<GlobalValue*, 32> AliveGlobals;

    /// GlobalIsNeeded - mark the specific global value as needed, and
    /// recursively mark anything that it uses as also needed.
    void GlobalIsNeeded(GlobalValue *GV);
    void MarkUsedGlobalsAsNeeded(Constant *C);

    bool RemoveUnusedGlobalValue(GlobalValue &GV);
  };
}

char GlobalDCE::ID = 0;
INITIALIZE_PASS(GlobalDCE, "globaldce",
                "Dead Global Elimination", false, false)

ModulePass *llvm::createGlobalDCEPass() { return new GlobalDCE(); }

bool GlobalDCE::runOnModule(Module &M) {
  bool Changed = false;
  
  // Loop over the module, adding globals which are obviously necessary.
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    Changed |= RemoveUnusedGlobalValue(*I);
    // Functions with external linkage are needed if they have a body
    if (!I->isDiscardableIfUnused() &&
        !I->isDeclaration() && !I->hasAvailableExternallyLinkage())
      GlobalIsNeeded(I);
  }

  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    Changed |= RemoveUnusedGlobalValue(*I);
    // Externally visible & appending globals are needed, if they have an
    // initializer.
    if (!I->isDiscardableIfUnused() &&
        !I->isDeclaration() && !I->hasAvailableExternallyLinkage())
      GlobalIsNeeded(I);
  }

  for (Module::alias_iterator I = M.alias_begin(), E = M.alias_end();
       I != E; ++I) {
    Changed |= RemoveUnusedGlobalValue(*I);
    // Externally visible aliases are needed.
    if (!I->isDiscardableIfUnused())
      GlobalIsNeeded(I);
  }

  // Now that all globals which are needed are in the AliveGlobals set, we loop
  // through the program, deleting those which are not alive.
  //

  // The first pass is to drop initializers of global variables which are dead.
  std::vector<GlobalVariable*> DeadGlobalVars;   // Keep track of dead globals
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    if (!AliveGlobals.count(I)) {
      DeadGlobalVars.push_back(I);         // Keep track of dead globals
      I->setInitializer(0);
    }

  // The second pass drops the bodies of functions which are dead...
  std::vector<Function*> DeadFunctions;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!AliveGlobals.count(I)) {
      DeadFunctions.push_back(I);         // Keep track of dead globals
      if (!I->isDeclaration())
        I->deleteBody();
    }

  // The third pass drops targets of aliases which are dead...
  std::vector<GlobalAlias*> DeadAliases;
  for (Module::alias_iterator I = M.alias_begin(), E = M.alias_end(); I != E;
       ++I)
    if (!AliveGlobals.count(I)) {
      DeadAliases.push_back(I);
      I->setAliasee(0);
    }

  if (!DeadFunctions.empty()) {
    // Now that all interferences have been dropped, delete the actual objects
    // themselves.
    for (unsigned i = 0, e = DeadFunctions.size(); i != e; ++i) {
      RemoveUnusedGlobalValue(*DeadFunctions[i]);
      M.getFunctionList().erase(DeadFunctions[i]);
    }
    NumFunctions += DeadFunctions.size();
    Changed = true;
  }

  if (!DeadGlobalVars.empty()) {
    for (unsigned i = 0, e = DeadGlobalVars.size(); i != e; ++i) {
      RemoveUnusedGlobalValue(*DeadGlobalVars[i]);
      M.getGlobalList().erase(DeadGlobalVars[i]);
    }
    NumVariables += DeadGlobalVars.size();
    Changed = true;
  }

  // Now delete any dead aliases.
  if (!DeadAliases.empty()) {
    for (unsigned i = 0, e = DeadAliases.size(); i != e; ++i) {
      RemoveUnusedGlobalValue(*DeadAliases[i]);
      M.getAliasList().erase(DeadAliases[i]);
    }
    NumAliases += DeadAliases.size();
    Changed = true;
  }

  // Make sure that all memory is released
  AliveGlobals.clear();

  return Changed;
}

/// GlobalIsNeeded - the specific global value as needed, and
/// recursively mark anything that it uses as also needed.
void GlobalDCE::GlobalIsNeeded(GlobalValue *G) {
  // If the global is already in the set, no need to reprocess it.
  if (!AliveGlobals.insert(G))
    return;
  
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(G)) {
    // If this is a global variable, we must make sure to add any global values
    // referenced by the initializer to the alive set.
    if (GV->hasInitializer())
      MarkUsedGlobalsAsNeeded(GV->getInitializer());
  } else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(G)) {
    // The target of a global alias is needed.
    MarkUsedGlobalsAsNeeded(GA->getAliasee());
  } else {
    // Otherwise this must be a function object.  We have to scan the body of
    // the function looking for constants and global values which are used as
    // operands.  Any operands of these types must be processed to ensure that
    // any globals used will be marked as needed.
    Function *F = cast<Function>(G);

    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
        for (User::op_iterator U = I->op_begin(), E = I->op_end(); U != E; ++U)
          if (GlobalValue *GV = dyn_cast<GlobalValue>(*U))
            GlobalIsNeeded(GV);
          else if (Constant *C = dyn_cast<Constant>(*U))
            MarkUsedGlobalsAsNeeded(C);
  }
}

void GlobalDCE::MarkUsedGlobalsAsNeeded(Constant *C) {
  if (GlobalValue *GV = dyn_cast<GlobalValue>(C))
    return GlobalIsNeeded(GV);
  
  // Loop over all of the operands of the constant, adding any globals they
  // use to the list of needed globals.
  for (User::op_iterator I = C->op_begin(), E = C->op_end(); I != E; ++I)
    if (Constant *OpC = dyn_cast<Constant>(*I))
      MarkUsedGlobalsAsNeeded(OpC);
}

// RemoveUnusedGlobalValue - Loop over all of the uses of the specified
// GlobalValue, looking for the constant pointer ref that may be pointing to it.
// If found, check to see if the constant pointer ref is safe to destroy, and if
// so, nuke it.  This will reduce the reference count on the global value, which
// might make it deader.
//
bool GlobalDCE::RemoveUnusedGlobalValue(GlobalValue &GV) {
  if (GV.use_empty()) return false;
  GV.removeDeadConstantUsers();
  return GV.use_empty();
}
