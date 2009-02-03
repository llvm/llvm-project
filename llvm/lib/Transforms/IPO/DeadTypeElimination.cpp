//===- DeadTypeElimination.cpp - Eliminate unused types for symbol table --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is used to cleanup the output of GCC.  It eliminate names for types
// that are unused in the entire translation unit, using the FindUsedTypes pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "deadtypeelim"
#include "llvm/Transforms/IPO.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Module.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumKilled, "Number of unused typenames removed from symtab");

namespace {
  struct VISIBILITY_HIDDEN DTE : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    DTE() : ModulePass(&ID) {}

    // doPassInitialization - For this pass, it removes global symbol table
    // entries for primitive types.  These are never used for linking in GCC and
    // they make the output uglier to look at, so we nuke them.
    //
    // Also, initialize instance variables.
    //
    bool runOnModule(Module &M);

    // getAnalysisUsage - This function needs FindUsedTypes to do its job...
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<FindUsedTypes>();
    }
  };
}

char DTE::ID = 0;
static RegisterPass<DTE> X("deadtypeelim", "Dead Type Elimination");

ModulePass *llvm::createDeadTypeEliminationPass() {
  return new DTE();
}


// ShouldNukeSymtabEntry - Return true if this module level symbol table entry
// should be eliminated.
//
static inline bool ShouldNukeSymtabEntry(const Type *Ty){
  // Nuke all names for primitive types!
  if (Ty->isPrimitiveType() || Ty->isInteger()) 
    return true;

  // Nuke all pointers to primitive types as well...
  if (const PointerType *PT = dyn_cast<PointerType>(Ty))
    if (PT->getElementType()->isPrimitiveType() ||
        PT->getElementType()->isInteger()) 
      return true;

  return false;
}

// run - For this pass, it removes global symbol table entries for primitive
// types.  These are never used for linking in GCC and they make the output
// uglier to look at, so we nuke them.  Also eliminate types that are never used
// in the entire program as indicated by FindUsedTypes.
//
bool DTE::runOnModule(Module &M) {
  bool Changed = false;

  TypeSymbolTable &ST = M.getTypeSymbolTable();
  std::set<const Type *> UsedTypes = getAnalysis<FindUsedTypes>().getTypes();

  // Check the symbol table for superfluous type entries...
  //
  // Grab the 'type' plane of the module symbol...
  TypeSymbolTable::iterator TI = ST.begin();
  TypeSymbolTable::iterator TE = ST.end();
  while ( TI != TE ) {
    // If this entry should be unconditionally removed, or if we detect that
    // the type is not used, remove it.
    const Type *RHS = TI->second;
    if (ShouldNukeSymtabEntry(RHS) || !UsedTypes.count(RHS)) {
      ST.remove(TI++);
      ++NumKilled;
      Changed = true;
    } else {
      ++TI;
      // We only need to leave one name for each type.
      UsedTypes.erase(RHS);
    }
  }

  return Changed;
}

// vim: sw=2
