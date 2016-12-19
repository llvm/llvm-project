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

#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "inline"

PreservedAnalyses AlwaysInlinerPass::run(Module &M, ModuleAnalysisManager &) {
  InlineFunctionInfo IFI;
  SmallSetVector<CallSite, 16> Calls;
  bool Changed = false;
  for (Function &F : M)
    if (!F.isDeclaration() && F.hasFnAttribute(Attribute::AlwaysInline) &&
        isInlineViable(F)) {
      Calls.clear();

      for (User *U : F.users())
        if (auto CS = CallSite(U))
          if (CS.getCalledFunction() == &F)
            Calls.insert(CS);

      for (CallSite CS : Calls)
        // FIXME: We really shouldn't be able to fail to inline at this point!
        // We should do something to log or check the inline failures here.
        Changed |= InlineFunction(CS, IFI);
    }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

namespace {

/// Inliner pass which only handles "always inline" functions.
///
/// Unlike the \c AlwaysInlinerPass, this uses the more heavyweight \c Inliner
/// base class to provide several facilities such as array alloca merging.
class AlwaysInlinerLegacyPass : public Inliner {

public:
  AlwaysInlinerLegacyPass() : Inliner(ID, /*InsertLifetime*/ true) {
    initializeAlwaysInlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  AlwaysInlinerLegacyPass(bool InsertLifetime) : Inliner(ID, InsertLifetime) {
    initializeAlwaysInlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  /// Main run interface method.  We override here to avoid calling skipSCC().
  bool runOnSCC(CallGraphSCC &SCC) override { return inlineCalls(SCC); }

  static char ID; // Pass identification, replacement for typeid

  InlineCost getInlineCost(CallSite CS) override;

  using llvm::Pass::doFinalization;
  bool doFinalization(CallGraph &CG) override {
    return removeDeadFunctions(CG, /*AlwaysInlineOnly=*/true);
  }
};
}

char AlwaysInlinerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(AlwaysInlinerLegacyPass, "always-inline",
                      "Inliner for always_inline functions", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(AlwaysInlinerLegacyPass, "always-inline",
                    "Inliner for always_inline functions", false, false)

Pass *llvm::createAlwaysInlinerLegacyPass(bool InsertLifetime) {
  return new AlwaysInlinerLegacyPass(InsertLifetime);
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
InlineCost AlwaysInlinerLegacyPass::getInlineCost(CallSite CS) {
  Function *Callee = CS.getCalledFunction();

  // Only inline direct calls to functions with always-inline attributes
  // that are viable for inlining. FIXME: We shouldn't even get here for
  // declarations.
  if (Callee && !Callee->isDeclaration() &&
      CS.hasFnAttr(Attribute::AlwaysInline) && isInlineViable(*Callee))
    return InlineCost::getAlways();

  return InlineCost::getNever();
}
