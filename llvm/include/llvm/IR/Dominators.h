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
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/GenericDomTree.h"

namespace llvm {

class Function;
class BasicBlock;
class raw_ostream;

extern template class DomTreeNodeBase<BasicBlock>;
extern template class DominatorTreeBase<BasicBlock>;

extern template void Calculate<Function, BasicBlock *>(
    DominatorTreeBase<GraphTraits<BasicBlock *>::NodeType> &DT, Function &F);
extern template void Calculate<Function, Inverse<BasicBlock *>>(
    DominatorTreeBase<GraphTraits<Inverse<BasicBlock *>>::NodeType> &DT,
    Function &F);

typedef DomTreeNodeBase<BasicBlock> DomTreeNode;

class BasicBlockEdge {
  const BasicBlock *Start;
  const BasicBlock *End;
public:
  BasicBlockEdge(const BasicBlock *Start_, const BasicBlock *End_) :
    Start(Start_), End(End_) { }
  const BasicBlock *getStart() const {
    return Start;
  }
  const BasicBlock *getEnd() const {
    return End;
  }
  bool isSingleEdge() const;
};

template <> struct DenseMapInfo<BasicBlockEdge> {
  static unsigned getHashValue(const BasicBlockEdge *V);
  typedef DenseMapInfo<const BasicBlock *> BBInfo;
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
class DominatorTree : public DominatorTreeBase<BasicBlock> {
public:
  typedef DominatorTreeBase<BasicBlock> Base;

  DominatorTree() : DominatorTreeBase<BasicBlock>(false) {}
  explicit DominatorTree(Function &F) : DominatorTreeBase<BasicBlock>(false) {
    recalculate(F);
  }

  DominatorTree(DominatorTree &&Arg)
      : Base(std::move(static_cast<Base &>(Arg))) {}
  DominatorTree &operator=(DominatorTree &&RHS) {
    Base::operator=(std::move(static_cast<Base &>(RHS)));
    return *this;
  }

  /// \brief Returns *false* if the other dominator tree matches this dominator
  /// tree.
  inline bool compare(const DominatorTree &Other) const {
    const DomTreeNode *R = getRootNode();
    const DomTreeNode *OtherR = Other.getRootNode();

    if (!R || !OtherR || R->getBlock() != OtherR->getBlock())
      return true;

    if (Base::compare(Other))
      return true;

    return false;
  }

  // Ensure base-class overloads are visible.
  using Base::dominates;

  /// \brief Return true if Def dominates a use in User.
  ///
  /// This performs the special checks necessary if Def and User are in the same
  /// basic block. Note that Def doesn't dominate a use in Def itself!
  bool dominates(const Instruction *Def, const Use &U) const;
  bool dominates(const Instruction *Def, const Instruction *User) const;
  bool dominates(const Instruction *Def, const BasicBlock *BB) const;
  bool dominates(const BasicBlockEdge &BBE, const Use &U) const;
  bool dominates(const BasicBlockEdge &BBE, const BasicBlock *BB) const;

  // Ensure base class overloads are visible.
  using Base::isReachableFromEntry;

  /// \brief Provide an overload for a Use.
  bool isReachableFromEntry(const Use &U) const;

  /// \brief Verify the correctness of the domtree by re-computing it.
  ///
  /// This should only be used for debugging as it aborts the program if the
  /// verification fails.
  void verifyDomTree() const;
};

//===-------------------------------------
// DominatorTree GraphTraits specializations so the DominatorTree can be
// iterable by generic graph iterators.

template <class Node, class ChildIterator> struct DomTreeGraphTraitsBase {
  typedef Node NodeType;
  typedef Node *NodeRef;
  typedef ChildIterator ChildIteratorType;
  typedef df_iterator<Node *, SmallPtrSet<NodeType *, 8>> nodes_iterator;

  static NodeType *getEntryNode(NodeType *N) { return N; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) { return N->end(); }

  static nodes_iterator nodes_begin(NodeType *N) {
    return df_begin(getEntryNode(N));
  }

  static nodes_iterator nodes_end(NodeType *N) {
    return df_end(getEntryNode(N));
  }
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
  static NodeType *getEntryNode(DominatorTree *DT) {
    return DT->getRootNode();
  }

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
  static char PassID;

public:
  /// \brief Provide the result typedef for this analysis pass.
  typedef DominatorTree Result;

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

} // End llvm namespace

#endif
