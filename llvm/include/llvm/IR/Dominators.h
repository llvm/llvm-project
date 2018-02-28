//===- Dominators.h - Dominator Info Calculation ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DominatorTree class, which provides fast and efficient
// dominance queries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DOMINATORS_H
#define LLVM_IR_DOMINATORS_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/GenericDomTree.h"
#include <utility>

namespace llvm {

class Function;
class Instruction;
class Module;
class raw_ostream;

extern template class DomTreeNodeBase<BasicBlock>;
extern template class DominatorTreeBase<BasicBlock, false>; // DomTree
extern template class DominatorTreeBase<BasicBlock, true>; // PostDomTree

namespace DomTreeBuilder {
using BBDomTree = DomTreeBase<BasicBlock>;
using BBPostDomTree = PostDomTreeBase<BasicBlock>;

extern template struct Update<BasicBlock *>;

using BBUpdates = ArrayRef<Update<BasicBlock *>>;

extern template void Calculate<BBDomTree>(BBDomTree &DT);
extern template void Calculate<BBPostDomTree>(BBPostDomTree &DT);

extern template void InsertEdge<BBDomTree>(BBDomTree &DT, BasicBlock *From,
                                           BasicBlock *To);
extern template void InsertEdge<BBPostDomTree>(BBPostDomTree &DT,
                                               BasicBlock *From,
                                               BasicBlock *To);

extern template void DeleteEdge<BBDomTree>(BBDomTree &DT, BasicBlock *From,
                                           BasicBlock *To);
extern template void DeleteEdge<BBPostDomTree>(BBPostDomTree &DT,
                                               BasicBlock *From,
                                               BasicBlock *To);

extern template void ApplyUpdates<BBDomTree>(BBDomTree &DT, BBUpdates);
extern template void ApplyUpdates<BBPostDomTree>(BBPostDomTree &DT, BBUpdates);

extern template bool Verify<BBDomTree>(const BBDomTree &DT,
                                       BBDomTree::VerificationLevel VL);
extern template bool Verify<BBPostDomTree>(const BBPostDomTree &DT,
                                           BBPostDomTree::VerificationLevel VL);
}  // namespace DomTreeBuilder

using DomTreeNode = DomTreeNodeBase<BasicBlock>;

class BasicBlockEdge {
  const BasicBlock *Start;
  const BasicBlock *End;

public:
  BasicBlockEdge(const BasicBlock *Start_, const BasicBlock *End_) :
    Start(Start_), End(End_) {}

  BasicBlockEdge(const std::pair<BasicBlock *, BasicBlock *> &Pair)
      : Start(Pair.first), End(Pair.second) {}

  BasicBlockEdge(const std::pair<const BasicBlock *, const BasicBlock *> &Pair)
      : Start(Pair.first), End(Pair.second) {}

  const BasicBlock *getStart() const {
    return Start;
  }

  const BasicBlock *getEnd() const {
    return End;
  }

  /// Check if this is the only edge between Start and End.
  bool isSingleEdge() const;
};

template <> struct DenseMapInfo<BasicBlockEdge> {
  using BBInfo = DenseMapInfo<const BasicBlock *>;

  static unsigned getHashValue(const BasicBlockEdge *V);

  static inline BasicBlockEdge getEmptyKey() {
    return BasicBlockEdge(BBInfo::getEmptyKey(), BBInfo::getEmptyKey());
  }

  static inline BasicBlockEdge getTombstoneKey() {
    return BasicBlockEdge(BBInfo::getTombstoneKey(), BBInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const BasicBlockEdge &Edge) {
    return hash_combine(BBInfo::getHashValue(Edge.getStart()),
                        BBInfo::getHashValue(Edge.getEnd()));
  }

  static bool isEqual(const BasicBlockEdge &LHS, const BasicBlockEdge &RHS) {
    return BBInfo::isEqual(LHS.getStart(), RHS.getStart()) &&
           BBInfo::isEqual(LHS.getEnd(), RHS.getEnd());
  }
};

/// \brief Concrete subclass of DominatorTreeBase that is used to compute a
/// normal dominator tree.
///
/// Definition: A block is said to be forward statically reachable if there is
/// a path from the entry of the function to the block.  A statically reachable
/// block may become statically unreachable during optimization.
///
/// A forward unreachable block may appear in the dominator tree, or it may
/// not.  If it does, dominance queries will return results as if all reachable
/// blocks dominate it.  When asking for a Node corresponding to a potentially
/// unreachable block, calling code must handle the case where the block was
/// unreachable and the result of getNode() is nullptr.
///
/// Generally, a block known to be unreachable when the dominator tree is
/// constructed will not be in the tree.  One which becomes unreachable after
/// the dominator tree is initially constructed may still exist in the tree,
/// even if the tree is properly updated. Calling code should not rely on the
/// preceding statements; this is stated only to assist human understanding.
class DominatorTree : public DominatorTreeBase<BasicBlock, false> {
 public:
  using Base = DominatorTreeBase<BasicBlock, false>;

  DominatorTree() = default;
  explicit DominatorTree(Function &F) { recalculate(F); }

  /// Handle invalidation explicitly.
  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &);

  // Ensure base-class overloads are visible.
  using Base::dominates;

  /// \brief Return true if Def dominates a use in User.
  ///
  /// This performs the special checks necessary if Def and User are in the same
  /// basic block. Note that Def doesn't dominate a use in Def itself!
  bool dominates(const Instruction *Def, const Use &U) const;
  bool dominates(const Instruction *Def, const Instruction *User) const;
  bool dominates(const Instruction *Def, const BasicBlock *BB) const;

  /// Return true if an edge dominates a use.
  ///
  /// If BBE is not a unique edge between start and end of the edge, it can
  /// never dominate the use.
  bool dominates(const BasicBlockEdge &BBE, const Use &U) const;
  bool dominates(const BasicBlockEdge &BBE, const BasicBlock *BB) const;

  // Ensure base class overloads are visible.
  using Base::isReachableFromEntry;

  /// \brief Provide an overload for a Use.
  bool isReachableFromEntry(const Use &U) const;

  // Pop up a GraphViz/gv window with the Dominator Tree rendered using `dot`.
  void viewGraph(const Twine &Name, const Twine &Title);
  void viewGraph();
};

//===-------------------------------------
// DominatorTree GraphTraits specializations so the DominatorTree can be
// iterable by generic graph iterators.

template <class Node, class ChildIterator> struct DomTreeGraphTraitsBase {
  using NodeRef = Node *;
  using ChildIteratorType = ChildIterator;
  using nodes_iterator = df_iterator<Node *, df_iterator_default_set<Node*>>;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->end(); }

  static nodes_iterator nodes_begin(NodeRef N) {
    return df_begin(getEntryNode(N));
  }

  static nodes_iterator nodes_end(NodeRef N) { return df_end(getEntryNode(N)); }
};

template <>
struct GraphTraits<DomTreeNode *>
    : public DomTreeGraphTraitsBase<DomTreeNode, DomTreeNode::iterator> {};

template <>
struct GraphTraits<const DomTreeNode *>
    : public DomTreeGraphTraitsBase<const DomTreeNode,
                                    DomTreeNode::const_iterator> {};

template <> struct GraphTraits<DominatorTree*>
  : public GraphTraits<DomTreeNode*> {
  static NodeRef getEntryNode(DominatorTree *DT) { return DT->getRootNode(); }

  static nodes_iterator nodes_begin(DominatorTree *N) {
    return df_begin(getEntryNode(N));
  }

  static nodes_iterator nodes_end(DominatorTree *N) {
    return df_end(getEntryNode(N));
  }
};

/// \brief Analysis pass which computes a \c DominatorTree.
class DominatorTreeAnalysis : public AnalysisInfoMixin<DominatorTreeAnalysis> {
  friend AnalysisInfoMixin<DominatorTreeAnalysis>;
  static AnalysisKey Key;

public:
  /// \brief Provide the result typedef for this analysis pass.
  using Result = DominatorTree;

  /// \brief Run the analysis pass over a function and produce a dominator tree.
  DominatorTree run(Function &F, FunctionAnalysisManager &);
};

/// \brief Printer pass for the \c DominatorTree.
class DominatorTreePrinterPass
    : public PassInfoMixin<DominatorTreePrinterPass> {
  raw_ostream &OS;

public:
  explicit DominatorTreePrinterPass(raw_ostream &OS);

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief Verifier pass for the \c DominatorTree.
struct DominatorTreeVerifierPass : PassInfoMixin<DominatorTreeVerifierPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief Legacy analysis pass which computes a \c DominatorTree.
class DominatorTreeWrapperPass : public FunctionPass {
  DominatorTree DT;

public:
  static char ID;

  DominatorTreeWrapperPass() : FunctionPass(ID) {
    initializeDominatorTreeWrapperPassPass(*PassRegistry::getPassRegistry());
  }

  DominatorTree &getDomTree() { return DT; }
  const DominatorTree &getDomTree() const { return DT; }

  bool runOnFunction(Function &F) override;

  void verifyAnalysis() const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  void releaseMemory() override { DT.releaseMemory(); }

  void print(raw_ostream &OS, const Module *M = nullptr) const override;
};

//===-------------------------------------
/// \brief Class to defer updates to a DominatorTree.
///
/// Definition: Applying updates to every edge insertion and deletion is
/// expensive and not necessary. When one needs the DominatorTree for analysis
/// they can request a flush() to perform a larger batch update. This has the
/// advantage of the DominatorTree inspecting the set of updates to find
/// duplicates or unnecessary subtree updates.
///
/// The scope of DeferredDominance operates at a Function level.
///
/// It is not necessary for the user to scrub the updates for duplicates or
/// updates that point to the same block (Delete, BB_A, BB_A). Performance
/// can be gained if the caller attempts to batch updates before submitting
/// to applyUpdates(ArrayRef) in cases where duplicate edge requests will
/// occur.
///
/// It is required for the state of the LLVM IR to be applied *before*
/// submitting updates. The update routines must analyze the current state
/// between a pair of (From, To) basic blocks to determine if the update
/// needs to be queued.
/// Example (good):
///     TerminatorInstructionBB->removeFromParent();
///     DDT->deleteEdge(BB, Successor);
/// Example (bad):
///     DDT->deleteEdge(BB, Successor);
///     TerminatorInstructionBB->removeFromParent();
class DeferredDominance {
public:
  DeferredDominance(DominatorTree &DT_) : DT(DT_) {}

  /// \brief Queues multiple updates and discards duplicates.
  void applyUpdates(ArrayRef<DominatorTree::UpdateType> Updates);

  /// \brief Helper method for a single edge insertion. It's almost always
  /// better to batch updates and call applyUpdates to quickly remove duplicate
  /// edges. This is best used when there is only a single insertion needed to
  /// update Dominators.
  void insertEdge(BasicBlock *From, BasicBlock *To);

  /// \brief Helper method for a single edge deletion. It's almost always better
  /// to batch updates and call applyUpdates to quickly remove duplicate edges.
  /// This is best used when there is only a single deletion needed to update
  /// Dominators.
  void deleteEdge(BasicBlock *From, BasicBlock *To);

  /// \brief Delays the deletion of a basic block until a flush() event.
  void deleteBB(BasicBlock *DelBB);

  /// \brief Returns true if DelBB is awaiting deletion at a flush() event.
  bool pendingDeletedBB(BasicBlock *DelBB);

  /// \brief Returns true if pending DT updates are queued for a flush() event.
  bool pending();

  /// \brief Flushes all pending updates and block deletions. Returns a
  /// correct DominatorTree reference to be used by the caller for analysis.
  DominatorTree &flush();

  /// \brief Drops all internal state and forces a (slow) recalculation of the
  /// DominatorTree based on the current state of the LLVM IR in F. This should
  /// only be used in corner cases such as the Entry block of F being deleted.
  void recalculate(Function &F);

  /// \brief Debug method to help view the state of pending updates.
  LLVM_DUMP_METHOD void dump() const;

private:
  DominatorTree &DT;
  SmallVector<DominatorTree::UpdateType, 16> PendUpdates;
  SmallPtrSet<BasicBlock *, 8> DeletedBBs;

  /// Apply an update (Kind, From, To) to the internal queued updates. The
  /// update is only added when determined to be necessary. Checks for
  /// self-domination, unnecessary updates, duplicate requests, and balanced
  /// pairs of requests are all performed. Returns true if the update is
  /// queued and false if it is discarded.
  bool applyUpdate(DominatorTree::UpdateKind Kind, BasicBlock *From,
                   BasicBlock *To);

  /// Performs all pending basic block deletions. We have to defer the deletion
  /// of these blocks until after the DominatorTree updates are applied. The
  /// internal workings of the DominatorTree code expect every update's From
  /// and To blocks to exist and to be a member of the same Function.
  bool flushDelBB();
};

} // end namespace llvm

#endif // LLVM_IR_DOMINATORS_H
