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

#include "llvm/Analysis/Dominators.h"
#include <map>
#include <set>

namespace llvm {
  
//===----------------------------------------------------------------------===//
/// DominanceFrontierBase - Common base class for computing forward and inverse
/// dominance frontiers for a function.
///
class DominanceFrontierBase : public FunctionPass {
public:
  typedef std::set<BasicBlock*>             DomSetType;    // Dom set for a bb
  typedef std::map<BasicBlock*, DomSetType> DomSetMapType; // Dom set map
protected:
  DomSetMapType Frontiers;
  std::vector<BasicBlock*> Roots;
  const bool IsPostDominators;

public:
  DominanceFrontierBase(char &ID, bool isPostDom)
    : FunctionPass(ID), IsPostDominators(isPostDom) {}

  /// getRoots - Return the root blocks of the current CFG.  This may include
  /// multiple blocks if we are computing post dominators.  For forward
  /// dominators, this will always be a single block (the entry node).
  ///
  inline const std::vector<BasicBlock*> &getRoots() const { return Roots; }

  /// isPostDominator - Returns true if analysis based of postdoms
  ///
  bool isPostDominator() const { return IsPostDominators; }

  virtual void releaseMemory() { Frontiers.clear(); }

  // Accessor interface:
  typedef DomSetMapType::iterator iterator;
  typedef DomSetMapType::const_iterator const_iterator;
  iterator       begin()       { return Frontiers.begin(); }
  const_iterator begin() const { return Frontiers.begin(); }
  iterator       end()         { return Frontiers.end(); }
  const_iterator end()   const { return Frontiers.end(); }
  iterator       find(BasicBlock *B)       { return Frontiers.find(B); }
  const_iterator find(BasicBlock *B) const { return Frontiers.find(B); }

  iterator addBasicBlock(BasicBlock *BB, const DomSetType &frontier) {
    assert(find(BB) == end() && "Block already in DominanceFrontier!");
    return Frontiers.insert(std::make_pair(BB, frontier)).first;
  }

  /// removeBlock - Remove basic block BB's frontier.
  void removeBlock(BasicBlock *BB) {
    assert(find(BB) != end() && "Block is not in DominanceFrontier!");
    for (iterator I = begin(), E = end(); I != E; ++I)
      I->second.erase(BB);
    Frontiers.erase(BB);
  }

  void addToFrontier(iterator I, BasicBlock *Node) {
    assert(I != end() && "BB is not in DominanceFrontier!");
    I->second.insert(Node);
  }

  void removeFromFrontier(iterator I, BasicBlock *Node) {
    assert(I != end() && "BB is not in DominanceFrontier!");
    assert(I->second.count(Node) && "Node is not in DominanceFrontier of BB");
    I->second.erase(Node);
  }

  /// compareDomSet - Return false if two domsets match. Otherwise
  /// return true;
  bool compareDomSet(DomSetType &DS1, const DomSetType &DS2) const {
    std::set<BasicBlock *> tmpSet;
    for (DomSetType::const_iterator I = DS2.begin(),
           E = DS2.end(); I != E; ++I)
      tmpSet.insert(*I);

    for (DomSetType::const_iterator I = DS1.begin(),
           E = DS1.end(); I != E; ) {
      BasicBlock *Node = *I++;

      if (tmpSet.erase(Node) == 0)
        // Node is in DS1 but not in DS2.
        return true;
    }

    if (!tmpSet.empty())
      // There are nodes that are in DS2 but not in DS1.
      return true;

    // DS1 and DS2 matches.
    return false;
  }

  /// compare - Return true if the other dominance frontier base matches
  /// this dominance frontier base. Otherwise return false.
  bool compare(DominanceFrontierBase &Other) const {
    DomSetMapType tmpFrontiers;
    for (DomSetMapType::const_iterator I = Other.begin(),
           E = Other.end(); I != E; ++I)
      tmpFrontiers.insert(std::make_pair(I->first, I->second));

    for (DomSetMapType::iterator I = tmpFrontiers.begin(),
           E = tmpFrontiers.end(); I != E; ) {
      BasicBlock *Node = I->first;
      const_iterator DFI = find(Node);
      if (DFI == end())
        return true;

      if (compareDomSet(I->second, DFI->second))
        return true;

      ++I;
      tmpFrontiers.erase(Node);
    }

    if (!tmpFrontiers.empty())
      return true;

    return false;
  }

  /// print - Convert to human readable form
  ///
  virtual void print(raw_ostream &OS, const Module* = 0) const;

  /// dump - Dump the dominance frontier to dbgs().
  void dump() const;
};


//===-------------------------------------
/// DominanceFrontier Class - Concrete subclass of DominanceFrontierBase that is
/// used to compute a forward dominator frontiers.
///
class DominanceFrontier : public DominanceFrontierBase {
  virtual void anchor();
public:
  static char ID; // Pass ID, replacement for typeid
  DominanceFrontier() :
    DominanceFrontierBase(ID, false) {
      initializeDominanceFrontierPass(*PassRegistry::getPassRegistry());
    }

  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    DominatorTree &DT = getAnalysis<DominatorTree>();
    Roots = DT.getRoots();
    assert(Roots.size() == 1 && "Only one entry block for forward domfronts!");
    calculate(DT, DT[Roots[0]]);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<DominatorTree>();
  }

  const DomSetType &calculate(const DominatorTree &DT,
                              const DomTreeNode *Node);
};

} // End llvm namespace

#endif
