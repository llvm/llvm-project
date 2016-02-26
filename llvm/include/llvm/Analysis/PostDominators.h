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

#ifndef LLVM_ANALYSIS_POSTDOMINATORS_H
#define LLVM_ANALYSIS_POSTDOMINATORS_H

#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// PostDominatorTree Class - Concrete subclass of DominatorTree that is used to
/// compute the post-dominator tree.
///
struct PostDominatorTree : public DominatorTreeBase<BasicBlock> {
  typedef DominatorTreeBase<BasicBlock> Base;

  PostDominatorTree() : DominatorTreeBase<BasicBlock>(true) {}

  PostDominatorTree(PostDominatorTree &&Arg)
    : Base(std::move(static_cast<Base &>(Arg))) {}

  PostDominatorTree &operator=(PostDominatorTree &&RHS) {
    Base::operator=(std::move(static_cast<Base &>(RHS)));
    return *this;
  }
};

/// \brief Analysis pass which computes a \c PostDominatorTree.
class PostDominatorTreeAnalysis {
public:
  /// \brief Provide the result typedef for this analysis pass.
  typedef PostDominatorTree Result;

  /// \brief Opaque, unique identifier for this analysis pass.
  static void *ID() { return (void *)&PassID; }

  /// \brief Run the analysis pass over a function and produce a post dominator
  ///        tree.
  PostDominatorTree run(Function &F);

  /// \brief Provide access to a name for this pass for debugging purposes.
  static StringRef name() { return "PostDominatorTreeAnalysis"; }

private:
  static char PassID;
};

/// \brief Printer pass for the \c PostDominatorTree.
class PostDominatorTreePrinterPass {
  raw_ostream &OS;

public:
  explicit PostDominatorTreePrinterPass(raw_ostream &OS);
  PreservedAnalyses run(Function &F, AnalysisManager<Function> *AM);

  static StringRef name() { return "PostDominatorTreePrinterPass"; }
};

struct PostDominatorTreeWrapperPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  PostDominatorTree DT;

  PostDominatorTreeWrapperPass() : FunctionPass(ID) {
    initializePostDominatorTreeWrapperPassPass(*PassRegistry::getPassRegistry());
  }

  PostDominatorTree &getPostDomTree() { return DT; }
  const PostDominatorTree &getPostDomTree() const { return DT; }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  void releaseMemory() override {
    DT.releaseMemory();
  }

  void print(raw_ostream &OS, const Module*) const override;
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
