//===- InlinerPass.h - Code common to all inliners --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a simple policy-based bottom-up inliner.  This file
// implements all of the boring mechanics of the bottom-up inlining, while the
// subclass determines WHAT to inline, which is the much more interesting
// component.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INLINERPASS_H
#define LLVM_TRANSFORMS_IPO_INLINERPASS_H

#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Utils/ImportedFunctionsInliningStatistics.h"

namespace llvm {
class AssumptionCacheTracker;
class CallSite;
class DataLayout;
class InlineCost;
class OptimizationRemarkEmitter;
class ProfileSummaryInfo;
template <class PtrType, unsigned SmallSize> class SmallPtrSet;

/// This class contains all of the helper code which is used to perform the
/// inlining operations that do not depend on the policy.
struct Inliner : public CallGraphSCCPass {
  explicit Inliner(char &ID);
  explicit Inliner(char &ID, bool InsertLifetime);

  /// For this class, we declare that we require and preserve the call graph.
  /// If the derived class implements this method, it should always explicitly
  /// call the implementation here.
  void getAnalysisUsage(AnalysisUsage &Info) const override;

  bool doInitialization(CallGraph &CG) override;

  /// Main run interface method, this implements the interface required by the
  /// Pass class.
  bool runOnSCC(CallGraphSCC &SCC) override;

  using llvm::Pass::doFinalization;
  /// Remove now-dead linkonce functions at the end of processing to avoid
  /// breaking the SCC traversal.
  bool doFinalization(CallGraph &CG) override;

  /// This method must be implemented by the subclass to determine the cost of
  /// inlining the specified call site.  If the cost returned is greater than
  /// the current inline threshold, the call site is not inlined.
  virtual InlineCost getInlineCost(CallSite CS) = 0;

  /// Remove dead functions.
  ///
  /// This also includes a hack in the form of the 'AlwaysInlineOnly' flag
  /// which restricts it to deleting functions with an 'AlwaysInline'
  /// attribute. This is useful for the InlineAlways pass that only wants to
  /// deal with that subset of the functions.
  bool removeDeadFunctions(CallGraph &CG, bool AlwaysInlineOnly = false);

  /// This function performs the main work of the pass.  The default of
  /// Inlinter::runOnSCC() calls skipSCC() before calling this method, but
  /// derived classes which cannot be skipped can override that method and call
  /// this function unconditionally.
  bool inlineCalls(CallGraphSCC &SCC);

private:
  // Insert @llvm.lifetime intrinsics.
  bool InsertLifetime;

protected:
  AssumptionCacheTracker *ACT;
  ProfileSummaryInfo *PSI;
  ImportedFunctionsInliningStatistics ImportedFunctionsStats;
};

} // End llvm namespace

#endif
