//===- InlineAlways.cpp - Code to inline always_inline functions ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a custom inliner that handles only functions that
// are marked as "always inline".
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "inline"
#include "llvm/CallingConv.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/DataLayout.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;

namespace {

  // AlwaysInliner only inlines functions that are mark as "always inline".
  class AlwaysInliner : public Inliner {
  public:
    // Use extremely low threshold.
    AlwaysInliner() : Inliner(ID, -2000000000, /*InsertLifetime*/true) {
      initializeAlwaysInlinerPass(*PassRegistry::getPassRegistry());
    }
    AlwaysInliner(bool InsertLifetime) : Inliner(ID, -2000000000,
                                                 InsertLifetime) {
      initializeAlwaysInlinerPass(*PassRegistry::getPassRegistry());
    }
    static char ID; // Pass identification, replacement for typeid
    virtual InlineCost getInlineCost(CallSite CS);
    virtual bool doFinalization(CallGraph &CG) {
      return removeDeadFunctions(CG, /*AlwaysInlineOnly=*/true);
    }
    virtual bool doInitialization(CallGraph &CG);
  };
}

char AlwaysInliner::ID = 0;
INITIALIZE_PASS_BEGIN(AlwaysInliner, "always-inline",
                "Inliner for always_inline functions", false, false)
INITIALIZE_AG_DEPENDENCY(CallGraph)
INITIALIZE_PASS_END(AlwaysInliner, "always-inline",
                "Inliner for always_inline functions", false, false)

Pass *llvm::createAlwaysInlinerPass() { return new AlwaysInliner(); }

Pass *llvm::createAlwaysInlinerPass(bool InsertLifetime) {
  return new AlwaysInliner(InsertLifetime);
}

/// \brief Minimal filter to detect invalid constructs for inlining.
static bool isInlineViable(Function &F) {
  bool ReturnsTwice =F.getFnAttributes().hasAttribute(Attributes::ReturnsTwice);
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI) {
    // Disallow inlining of functions which contain an indirect branch.
    if (isa<IndirectBrInst>(BI->getTerminator()))
      return false;

    for (BasicBlock::iterator II = BI->begin(), IE = BI->end(); II != IE;
         ++II) {
      CallSite CS(II);
      if (!CS)
        continue;

      // Disallow recursive calls.
      if (&F == CS.getCalledFunction())
        return false;

      // Disallow calls which expose returns-twice to a function not previously
      // attributed as such.
      if (!ReturnsTwice && CS.isCall() &&
          cast<CallInst>(CS.getInstruction())->canReturnTwice())
        return false;
    }
  }

  return true;
}

/// \brief Get the inline cost for the always-inliner.
///
/// The always inliner *only* handles functions which are marked with the
/// attribute to force inlining. As such, it is dramatically simpler and avoids
/// using the powerful (but expensive) inline cost analysis. Instead it uses
/// a very simple and boring direct walk of the instructions looking for
/// impossible-to-inline constructs.
///
/// Note, it would be possible to go to some lengths to cache the information
/// computed here, but as we only expect to do this for relatively few and
/// small functions which have the explicit attribute to force inlining, it is
/// likely not worth it in practice.
InlineCost AlwaysInliner::getInlineCost(CallSite CS) {
  Function *Callee = CS.getCalledFunction();
  // We assume indirect calls aren't calling an always-inline function.
  if (!Callee) return InlineCost::getNever();

  // We can't inline calls to external functions.
  // FIXME: We shouldn't even get here.
  if (Callee->isDeclaration()) return InlineCost::getNever();

  // Return never for anything not marked as always inline.
  if (!Callee->getFnAttributes().hasAttribute(Attributes::AlwaysInline))
    return InlineCost::getNever();

  // Do some minimal analysis to preclude non-viable functions.
  if (!isInlineViable(*Callee))
    return InlineCost::getNever();

  // Otherwise, force inlining.
  return InlineCost::getAlways();
}

// doInitialization - Initializes the vector of functions that have not
// been annotated with the "always inline" attribute.
bool AlwaysInliner::doInitialization(CallGraph &CG) {
  return false;
}
