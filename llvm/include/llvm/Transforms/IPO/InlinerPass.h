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
class BlockFrequencyAnalysis;
template <class PtrType, unsigned SmallSize> class SmallPtrSet;

// Functor invoked when a block is cloned during inlining.
typedef std::function<void(const BasicBlock *, const BasicBlock *)>
    BlockCloningFunctor;
// Functor invoked when a function is inlined inside the basic block
// containing the call.
typedef std::function<void(BasicBlock *, Function *)> FunctionCloningFunctor;
// Functor invoked when a function gets deleted during inlining.
typedef std::function<void(Function *)> FunctionDeletedFunctor;

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

private:
  // InsertLifetime - Insert @llvm.lifetime intrinsics.
  bool InsertLifetime;

  /// shouldInline - Return true if the inliner should attempt to
  /// inline at the given CallSite.
  bool shouldInline(CallSite CS);
  /// Set the BFI of \p Dst to be the same as \p Src.
  void copyBlockFrequency(BasicBlock *Src, BasicBlock *Dst);
  /// Invalidates BFI for function \p F.
  void invalidateBFI(Function *F);
  /// Invalidates BFI for all functions in  \p SCC.
  void invalidateBFI(CallGraphSCC &SCC);
  /// Update function entry count for \p Callee which has been inlined into
  /// \p CallBB.
  void updateEntryCount(BasicBlock *CallBB, Function *Callee);
  /// \brief Update block frequency of an inlined block.
  /// This method updates the block frequency of \p NewBB which is a clone of
  /// \p OrigBB when the callsite \p CS gets inlined. The frequency of \p NewBB
  /// is computed as follows:
  /// Freq(NewBB) = Freq(OrigBB) * CallSiteFreq / CalleeEntryFreq.
  void updateBlockFreq(CallSite &CS, const BasicBlock *OrigBB,
                       const BasicBlock *NewBB);

protected:
  AssumptionCacheTracker *ACT;
  std::unique_ptr<BlockFrequencyAnalysis> BFA;
  /// Are we using profile guided optimization?
  bool HasProfileData;
};

} // End llvm namespace

#endif
