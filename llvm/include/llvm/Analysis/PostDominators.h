//=- llvm/Analysis/PostDominators.h - Post Dominator Calculation-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes interfaces to post dominance information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_POST_DOMINATORS_H
#define LLVM_ANALYSIS_POST_DOMINATORS_H

#include "llvm/Analysis/Dominators.h"

namespace llvm {

/// PostDominatorTree Class - Concrete subclass of DominatorTree that is used to
/// compute the a post-dominator tree.
///
struct PostDominatorTree : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  DominatorTreeBase<BasicBlock>* DT;

  PostDominatorTree() : FunctionPass(ID) {
    initializePostDominatorTreePass(*PassRegistry::getPassRegistry());
    DT = new DominatorTreeBase<BasicBlock>(true);
  }

  ~PostDominatorTree();

  virtual bool runOnFunction(Function &F);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  inline const std::vector<BasicBlock*> &getRoots() const {
    return DT->getRoots();
  }

  inline DomTreeNode *getRootNode() const {
    return DT->getRootNode();
  }

  inline DomTreeNode *operator[](BasicBlock *BB) const {
    return DT->getNode(BB);
  }

  inline DomTreeNode *getNode(BasicBlock *BB) const {
    return DT->getNode(BB);
  }

  inline bool dominates(DomTreeNode* A, DomTreeNode* B) const {
    return DT->dominates(A, B);
  }

  inline bool dominates(const BasicBlock* A, const BasicBlock* B) const {
    return DT->dominates(A, B);
  }

  inline bool properlyDominates(const DomTreeNode* A, DomTreeNode* B) const {
    return DT->properlyDominates(A, B);
  }

  inline bool properlyDominates(BasicBlock* A, BasicBlock* B) const {
    return DT->properlyDominates(A, B);
  }

  inline BasicBlock *findNearestCommonDominator(BasicBlock *A, BasicBlock *B) {
    return DT->findNearestCommonDominator(A, B);
  }

  virtual void releaseMemory() {
    DT->releaseMemory();
  }

  virtual void print(raw_ostream &OS, const Module*) const;
};

FunctionPass* createPostDomTree();

template <> struct GraphTraits<PostDominatorTree*>
  : public GraphTraits<DomTreeNode*> {
  static NodeType *getEntryNode(PostDominatorTree *DT) {
    return DT->getRootNode();
  }

  static nodes_iterator nodes_begin(PostDominatorTree *N) {
    if (getEntryNode(N))
      return df_begin(getEntryNode(N));
    else
      return df_end(getEntryNode(N));
  }

  static nodes_iterator nodes_end(PostDominatorTree *N) {
    return df_end(getEntryNode(N));
  }
};

} // End llvm namespace

#endif
