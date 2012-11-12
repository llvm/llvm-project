//=- llvm/CodeGen/MachineDominators.h - Machine Dom Calculation --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes mirroring those in llvm/Analysis/Dominators.h,
// but for target-specific code rather than target-independent IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEDOMINATORS_H
#define LLVM_CODEGEN_MACHINEDOMINATORS_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/DominatorInternals.h"

namespace llvm {

template<>
inline void DominatorTreeBase<MachineBasicBlock>::addRoot(MachineBasicBlock* MBB) {
  this->Roots.push_back(MBB);
}

EXTERN_TEMPLATE_INSTANTIATION(class DomTreeNodeBase<MachineBasicBlock>);
EXTERN_TEMPLATE_INSTANTIATION(class DominatorTreeBase<MachineBasicBlock>);

typedef DomTreeNodeBase<MachineBasicBlock> MachineDomTreeNode;

//===-------------------------------------
/// DominatorTree Class - Concrete subclass of DominatorTreeBase that is used to
/// compute a normal dominator tree.
///
class MachineDominatorTree : public MachineFunctionPass {
public:
  static char ID; // Pass ID, replacement for typeid
  DominatorTreeBase<MachineBasicBlock>* DT;
  
  MachineDominatorTree();
  
  ~MachineDominatorTree();
  
  DominatorTreeBase<MachineBasicBlock>& getBase() { return *DT; }
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  
  /// getRoots -  Return the root blocks of the current CFG.  This may include
  /// multiple blocks if we are computing post dominators.  For forward
  /// dominators, this will always be a single block (the entry node).
  ///
  inline const std::vector<MachineBasicBlock*> &getRoots() const {
    return DT->getRoots();
  }
  
  inline MachineBasicBlock *getRoot() const {
    return DT->getRoot();
  }
  
  inline MachineDomTreeNode *getRootNode() const {
    return DT->getRootNode();
  }
  
  virtual bool runOnMachineFunction(MachineFunction &F);
  
  inline bool dominates(MachineDomTreeNode* A, MachineDomTreeNode* B) const {
    return DT->dominates(A, B);
  }
  
  inline bool dominates(MachineBasicBlock* A, MachineBasicBlock* B) const {
    return DT->dominates(A, B);
  }
  
  // dominates - Return true if A dominates B. This performs the
  // special checks necessary if A and B are in the same basic block.
  bool dominates(MachineInstr *A, MachineInstr *B) const {
    MachineBasicBlock *BBA = A->getParent(), *BBB = B->getParent();
    if (BBA != BBB) return DT->dominates(BBA, BBB);

    // Loop through the basic block until we find A or B.
    MachineBasicBlock::iterator I = BBA->begin();
    for (; &*I != A && &*I != B; ++I)
      /*empty*/ ;

    //if(!DT.IsPostDominators) {
      // A dominates B if it is found first in the basic block.
      return &*I == A;
    //} else {
    //  // A post-dominates B if B is found first in the basic block.
    //  return &*I == B;
    //}
  }
  
  inline bool properlyDominates(const MachineDomTreeNode* A,
                                MachineDomTreeNode* B) const {
    return DT->properlyDominates(A, B);
  }
  
  inline bool properlyDominates(MachineBasicBlock* A,
                                MachineBasicBlock* B) const {
    return DT->properlyDominates(A, B);
  }
  
  /// findNearestCommonDominator - Find nearest common dominator basic block
  /// for basic block A and B. If there is no such block then return NULL.
  inline MachineBasicBlock *findNearestCommonDominator(MachineBasicBlock *A,
                                                       MachineBasicBlock *B) {
    return DT->findNearestCommonDominator(A, B);
  }
  
  inline MachineDomTreeNode *operator[](MachineBasicBlock *BB) const {
    return DT->getNode(BB);
  }
  
  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.
  ///
  inline MachineDomTreeNode *getNode(MachineBasicBlock *BB) const {
    return DT->getNode(BB);
  }
  
  /// addNewBlock - Add a new node to the dominator tree information.  This
  /// creates a new node as a child of DomBB dominator node,linking it into 
  /// the children list of the immediate dominator.
  inline MachineDomTreeNode *addNewBlock(MachineBasicBlock *BB,
                                         MachineBasicBlock *DomBB) {
    return DT->addNewBlock(BB, DomBB);
  }
  
  /// changeImmediateDominator - This method is used to update the dominator
  /// tree information when a node's immediate dominator changes.
  ///
  inline void changeImmediateDominator(MachineBasicBlock *N,
                                       MachineBasicBlock* NewIDom) {
    DT->changeImmediateDominator(N, NewIDom);
  }
  
  inline void changeImmediateDominator(MachineDomTreeNode *N,
                                       MachineDomTreeNode* NewIDom) {
    DT->changeImmediateDominator(N, NewIDom);
  }
  
  /// eraseNode - Removes a node from  the dominator tree. Block must not
  /// dominate any other blocks. Removes node from its immediate dominator's
  /// children list. Deletes dominator node associated with basic block BB.
  inline void eraseNode(MachineBasicBlock *BB) {
    DT->eraseNode(BB);
  }
  
  /// splitBlock - BB is split and now it has one successor. Update dominator
  /// tree to reflect this change.
  inline void splitBlock(MachineBasicBlock* NewBB) {
    DT->splitBlock(NewBB);
  }

  /// isReachableFromEntry - Return true if A is dominated by the entry
  /// block of the function containing it.
  bool isReachableFromEntry(MachineBasicBlock *A) {
    return DT->isReachableFromEntry(A);
  }

  virtual void releaseMemory();
  
  virtual void print(raw_ostream &OS, const Module*) const;
};

//===-------------------------------------
/// DominatorTree GraphTraits specialization so the DominatorTree can be
/// iterable by generic graph iterators.
///

template<class T> struct GraphTraits;

template <> struct GraphTraits<MachineDomTreeNode *> {
  typedef MachineDomTreeNode NodeType;
  typedef NodeType::iterator  ChildIteratorType;
  
  static NodeType *getEntryNode(NodeType *N) {
    return N;
  }
  static inline ChildIteratorType child_begin(NodeType* N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType* N) {
    return N->end();
  }
};

template <> struct GraphTraits<MachineDominatorTree*>
  : public GraphTraits<MachineDomTreeNode *> {
  static NodeType *getEntryNode(MachineDominatorTree *DT) {
    return DT->getRootNode();
  }
};

}

#endif
