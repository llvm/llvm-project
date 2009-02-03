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

  PostDominatorTree() : FunctionPass(&ID) {
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
  
  inline bool properlyDominates(const DomTreeNode* A, DomTreeNode* B) const {
    return DT->properlyDominates(A, B);
  }
  
  inline bool properlyDominates(BasicBlock* A, BasicBlock* B) const {
    return DT->properlyDominates(A, B);
  }

  virtual void print(std::ostream &OS, const Module* M= 0) const {
    DT->print(OS, M);
  }
};

FunctionPass* createPostDomTree();

/// PostDominanceFrontier Class - Concrete subclass of DominanceFrontier that is
/// used to compute the a post-dominance frontier.
///
struct PostDominanceFrontier : public DominanceFrontierBase {
  static char ID;
  PostDominanceFrontier() 
    : DominanceFrontierBase(&ID, true) {}

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    PostDominatorTree &DT = getAnalysis<PostDominatorTree>();
    Roots = DT.getRoots();
    if (const DomTreeNode *Root = DT.getRootNode())
      calculate(DT, Root);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<PostDominatorTree>();
  }

private:
  const DomSetType &calculate(const PostDominatorTree &DT,
                              const DomTreeNode *Node);
};

FunctionPass* createPostDomFrontier();

} // End llvm namespace

#endif
