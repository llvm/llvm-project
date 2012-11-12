//===- NoAliasAnalysis.cpp - Minimal Alias Analysis Impl ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the default implementation of the Alias Analysis interface
// that simply returns "I don't know" for all queries.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
#include "llvm/DataLayout.h"
using namespace llvm;

namespace {
  /// NoAA - This class implements the -no-aa pass, which always returns "I
  /// don't know" for alias queries.  NoAA is unlike other alias analysis
  /// implementations, in that it does not chain to a previous analysis.  As
  /// such it doesn't follow many of the rules that other alias analyses must.
  ///
  struct NoAA : public ImmutablePass, public AliasAnalysis {
    static char ID; // Class identification, replacement for typeinfo
    NoAA() : ImmutablePass(ID) {
      initializeNoAAPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    }

    virtual void initializePass() {
      // Note: NoAA does not call InitializeAliasAnalysis because it's
      // special and does not support chaining.
      TD = getAnalysisIfAvailable<DataLayout>();
    }

    virtual AliasResult alias(const Location &LocA, const Location &LocB) {
      return MayAlias;
    }

    virtual ModRefBehavior getModRefBehavior(ImmutableCallSite CS) {
      return UnknownModRefBehavior;
    }
    virtual ModRefBehavior getModRefBehavior(const Function *F) {
      return UnknownModRefBehavior;
    }

    virtual bool pointsToConstantMemory(const Location &Loc,
                                        bool OrLocal) {
      return false;
    }
    virtual ModRefResult getModRefInfo(ImmutableCallSite CS,
                                       const Location &Loc) {
      return ModRef;
    }
    virtual ModRefResult getModRefInfo(ImmutableCallSite CS1,
                                       ImmutableCallSite CS2) {
      return ModRef;
    }

    virtual void deleteValue(Value *V) {}
    virtual void copyValue(Value *From, Value *To) {}
    virtual void addEscapingUse(Use &U) {}
    
    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    virtual void *getAdjustedAnalysisPointer(const void *ID) {
      if (ID == &AliasAnalysis::ID)
        return (AliasAnalysis*)this;
      return this;
    }
  };
}  // End of anonymous namespace

// Register this pass...
char NoAA::ID = 0;
INITIALIZE_AG_PASS(NoAA, AliasAnalysis, "no-aa",
                   "No Alias Analysis (always returns 'may' alias)",
                   true, true, true)

ImmutablePass *llvm::createNoAAPass() { return new NoAA(); }
