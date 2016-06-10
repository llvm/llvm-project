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

namespace llvm {
class AssumptionCacheTracker;
class CallSite;
class DataLayout;
class InlineCost;
class ProfileSummaryInfo;
template <class PtrType, unsigned SmallSize> class SmallPtrSet;

/// Inliner - This class contains all of the helper code which is used to
/// perform the inlining operations that do not depend on the policy.
///
struct Inliner : public CallGraphSCCPass {
  explicit Inliner(char &ID);
  explicit Inliner(char &ID, bool InsertLifetime);

  /// getAnalysisUsage - For this class, we declare that we require and preserve
  /// the call graph.  If the derived class implements this method, it should
  /// always explicitly call the implementation here.
  void getAnalysisUsage(AnalysisUsage &Info) const override;

  // Main run interface method, this implements the interface required by the
  // Pass class.
  bool runOnSCC(CallGraphSCC &SCC) override;

  using llvm::Pass::doFinalization;
  // doFinalization - Remove now-dead linkonce functions at the end of
  // processing to avoid breaking the SCC traversal.
  bool doFinalization(CallGraph &CG) override;

  /// getInlineCost - This method must be implemented by the subclass to
  /// determine the cost of inlining the specified call site.  If the cost
  /// returned is greater than the current inline threshold, the call site is
  /// not inlined.
  ///
  virtual InlineCost getInlineCost(CallSite CS) = 0;

  /// removeDeadFunctions - Remove dead functions.
  ///
  /// This also includes a hack in the form of the 'AlwaysInlineOnly' flag
  /// which restricts it to deleting functions with an 'AlwaysInline'
  /// attribute. This is useful for the InlineAlways pass that only wants to
  /// deal with that subset of the functions.
  bool removeDeadFunctions(CallGraph &CG, bool AlwaysInlineOnly = false);

  /// This function performs the main work of the pass.  The default
  /// of Inlinter::runOnSCC() calls skipSCC() before calling this method, but
  /// derived classes which cannot be skipped can override that method and
  /// call this function unconditionally.
  bool inlineCalls(CallGraphSCC &SCC);

private:
  // InsertLifetime - Insert @llvm.lifetime intrinsics.
  bool InsertLifetime;

  /// shouldInline - Return true if the inliner should attempt to
  /// inline at the given CallSite.
  bool shouldInline(CallSite CS);
  /// Return true if inlining of CS can block the caller from being
  /// inlined which is proved to be more beneficial. \p IC is the
  /// estimated inline cost associated with callsite \p CS.
  /// \p TotalAltCost will be set to the estimated cost of inlining the caller
  /// if \p CS is suppressed for inlining.
  bool shouldBeDeferred(Function *Caller, CallSite CS, InlineCost IC,
                        int &TotalAltCost);

protected:
  AssumptionCacheTracker *ACT;
  ProfileSummaryInfo *PSI;
};

} // End llvm namespace

#endif
