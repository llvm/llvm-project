//===- llvm/Analysis/DominanceFrontier.h - Dominator Frontiers --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DominanceFrontier class, which calculate and holds the
// dominance frontier for a function.
//
// This should be considered deprecated, don't add any more uses of this data
// structure.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMINANCEFRONTIER_H
#define LLVM_ANALYSIS_DOMINANCEFRONTIER_H

#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"
#include <map>
#include <set>

namespace llvm {

//===----------------------------------------------------------------------===//
/// DominanceFrontierBase - Common base class for computing forward and inverse
/// dominance frontiers for a function.
///
template <class BlockT>
class DominanceFrontierBase {
public:
  typedef std::set<BlockT *> DomSetType;                // Dom set for a bb
  typedef std::map<BlockT *, DomSetType> DomSetMapType; // Dom set map

protected:
  typedef GraphTraits<BlockT *> BlockTraits;

  DomSetMapType Frontiers;
  std::vector<BlockT *> Roots;
  const bool IsPostDominators;

public:
  DominanceFrontierBase(bool isPostDom) : IsPostDominators(isPostDom) {}

  /// getRoots - Return the root blocks of the current CFG.  This may include
  /// multiple blocks if we are computing post dominators.  For forward
  /// dominators, this will always be a single block (the entry node).
  ///
  inline const std::vector<BlockT *> &getRoots() const {
    return Roots;
  }

  BlockT *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }

  /// isPostDominator - Returns true if analysis based of postdoms
  ///
  bool isPostDominator() const {
    return IsPostDominators;
  }

  void releaseMemory() {
    Frontiers.clear();
  }

  // Accessor interface:
  typedef typename DomSetMapType::iterator iterator;
  typedef typename DomSetMapType::const_iterator const_iterator;
  iterator begin() { return Frontiers.begin(); }
  const_iterator begin() const { return Frontiers.begin(); }
  iterator end() { return Frontiers.end(); }
  const_iterator end() const { return Frontiers.end(); }
  iterator find(BlockT *B) { return Frontiers.find(B); }
  const_iterator find(BlockT *B) const { return Frontiers.find(B); }

  iterator addBasicBlock(BlockT *BB, const DomSetType &frontier) {
    assert(find(BB) == end() && "Block already in DominanceFrontier!");
    return Frontiers.insert(std::make_pair(BB, frontier)).first;
  }

  /// removeBlock - Remove basic block BB's frontier.
  void removeBlock(BlockT *BB);

  void addToFrontier(iterator I, BlockT *Node);

  void removeFromFrontier(iterator I, BlockT *Node);

  /// compareDomSet - Return false if two domsets match. Otherwise
  /// return true;
  bool compareDomSet(DomSetType &DS1, const DomSetType &DS2) const;

  /// compare - Return true if the other dominance frontier base matches
  /// this dominance frontier base. Otherwise return false.
  bool compare(DominanceFrontierBase<BlockT> &Other) const;

  /// print - Convert to human readable form
  ///
  void print(raw_ostream &OS) const;

  /// dump - Dump the dominance frontier to dbgs().
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump() const;
#endif
};

//===-------------------------------------
/// DominanceFrontier Class - Concrete subclass of DominanceFrontierBase that is
/// used to compute a forward dominator frontiers.
///
template <class BlockT>
class ForwardDominanceFrontierBase : public DominanceFrontierBase<BlockT> {
private:
  typedef GraphTraits<BlockT *> BlockTraits;

public:
  typedef DominatorTreeBase<BlockT> DomTreeT;
  typedef DomTreeNodeBase<BlockT> DomTreeNodeT;
  typedef typename DominanceFrontierBase<BlockT>::DomSetType DomSetType;

  ForwardDominanceFrontierBase() : DominanceFrontierBase<BlockT>(false) {}

  void analyze(DomTreeT &DT) {
    this->Roots = DT.getRoots();
    assert(this->Roots.size() == 1 &&
           "Only one entry block for forward domfronts!");
    calculate(DT, DT[this->Roots[0]]);
  }

  const DomSetType &calculate(const DomTreeT &DT, const DomTreeNodeT *Node);
};

class DominanceFrontier : public ForwardDominanceFrontierBase<BasicBlock> {
public:
  typedef DominatorTreeBase<BasicBlock> DomTreeT;
  typedef DomTreeNodeBase<BasicBlock> DomTreeNodeT;
  typedef DominanceFrontierBase<BasicBlock>::DomSetType DomSetType;
  typedef DominanceFrontierBase<BasicBlock>::iterator iterator;
  typedef DominanceFrontierBase<BasicBlock>::const_iterator const_iterator;
};

class DominanceFrontierWrapperPass : public FunctionPass {
  DominanceFrontier DF;
public:
  static char ID; // Pass ID, replacement for typeid

  DominanceFrontierWrapperPass();

  DominanceFrontier &getDominanceFrontier() { return DF; }
  const DominanceFrontier &getDominanceFrontier() const { return DF;  }

  void releaseMemory() override;

  bool runOnFunction(Function &) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void print(raw_ostream &OS, const Module * = nullptr) const override;

  void dump() const;
};

extern template class DominanceFrontierBase<BasicBlock>;
extern template class ForwardDominanceFrontierBase<BasicBlock>;

/// \brief Analysis pass which computes a \c DominanceFrontier.
class DominanceFrontierAnalysis
    : public AnalysisInfoMixin<DominanceFrontierAnalysis> {
  friend AnalysisInfoMixin<DominanceFrontierAnalysis>;
  static char PassID;

public:
  /// \brief Provide the result typedef for this analysis pass.
  typedef DominanceFrontier Result;

  /// \brief Run the analysis pass over a function and produce a dominator tree.
  DominanceFrontier run(Function &F, AnalysisManager<Function> &AM);
};

/// \brief Printer pass for the \c DominanceFrontier.
class DominanceFrontierPrinterPass
    : public PassInfoMixin<DominanceFrontierPrinterPass> {
  raw_ostream &OS;

public:
  explicit DominanceFrontierPrinterPass(raw_ostream &OS);
  PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);
};

} // End llvm namespace

#endif
