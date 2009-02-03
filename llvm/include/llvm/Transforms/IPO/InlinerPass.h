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

#ifndef INLINER_H
#define INLINER_H

#include "llvm/CallGraphSCCPass.h"
#include "llvm/Transforms/Utils/InlineCost.h"

namespace llvm {
  class CallSite;

/// Inliner - This class contains all of the helper code which is used to
/// perform the inlining operations that do not depend on the policy.
///
struct Inliner : public CallGraphSCCPass {
  explicit Inliner(void *ID);
  explicit Inliner(void *ID, int Threshold);

  /// getAnalysisUsage - For this class, we declare that we require and preserve
  /// the call graph.  If the derived class implements this method, it should
  /// always explicitly call the implementation here.
  virtual void getAnalysisUsage(AnalysisUsage &Info) const;

  // Main run interface method, this implements the interface required by the
  // Pass class.
  virtual bool runOnSCC(const std::vector<CallGraphNode *> &SCC);

  // doFinalization - Remove now-dead linkonce functions at the end of
  // processing to avoid breaking the SCC traversal.
  virtual bool doFinalization(CallGraph &CG);


  /// This method returns the value specified by the -inline-threshold value,
  /// specified on the command line.  This is typically not directly needed.
  ///
  unsigned getInlineThreshold() const { return InlineThreshold; }

  /// getInlineCost - This method must be implemented by the subclass to
  /// determine the cost of inlining the specified call site.  If the cost
  /// returned is greater than the current inline threshold, the call site is
  /// not inlined.
  ///
  virtual InlineCost getInlineCost(CallSite CS) = 0;

  // getInlineFudgeFactor - Return a > 1.0 factor if the inliner should use a
  // higher threshold to determine if the function call should be inlined.
  ///
  virtual float getInlineFudgeFactor(CallSite CS) = 0;

  /// resetCachedCostInfo - erase any cached cost data from the derived class.
  /// If the derived class has no such data this can be empty.
  /// 
  virtual void resetCachedCostInfo(Function* Caller) = 0;

  /// removeDeadFunctions - Remove dead functions that are not included in
  /// DNR (Do Not Remove) list.
  bool removeDeadFunctions(CallGraph &CG, 
                           SmallPtrSet<const Function *, 16> *DNR = NULL);
private:
  // InlineThreshold - Cache the value here for easy access.
  unsigned InlineThreshold;

  /// shouldInline - Return true if the inliner should attempt to
  /// inline at the given CallSite.
  bool shouldInline(CallSite CS);
};

} // End llvm namespace

#endif
