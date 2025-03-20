//==- llvm/CodeGen/MachineDominators.h - Machine Dom Calculation -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes mirroring those in llvm/Analysis/Dominators.h,
// but for target-specific code rather than target-independent IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEDOMINATORS_H
#define LLVM_CODEGEN_MACHINEDOMINATORS_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBundleIterator.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/GenericDomTree.h"
#include <cassert>
#include <memory>
#include <optional>

namespace llvm {
class AnalysisUsage;
class MachineFunction;
class Module;
class raw_ostream;

template <>
inline void DominatorTreeBase<MachineBasicBlock, false>::addRoot(
    MachineBasicBlock *MBB) {
  this->Roots.push_back(MBB);
}

extern template class DomTreeNodeBase<MachineBasicBlock>;
extern template class DominatorTreeBase<MachineBasicBlock, false>; // DomTree

using MachineDomTreeNode = DomTreeNodeBase<MachineBasicBlock>;

namespace DomTreeBuilder {
using MBBDomTree = DomTreeBase<MachineBasicBlock>;
using MBBUpdates = ArrayRef<llvm::cfg::Update<MachineBasicBlock *>>;
using MBBDomTreeGraphDiff = GraphDiff<MachineBasicBlock *, false>;

extern template void Calculate<MBBDomTree>(MBBDomTree &DT);
extern template void CalculateWithUpdates<MBBDomTree>(MBBDomTree &DT,
                                                      MBBUpdates U);

extern template void InsertEdge<MBBDomTree>(MBBDomTree &DT,
                                            MachineBasicBlock *From,
                                            MachineBasicBlock *To);

extern template void DeleteEdge<MBBDomTree>(MBBDomTree &DT,
                                            MachineBasicBlock *From,
                                            MachineBasicBlock *To);

extern template void ApplyUpdates<MBBDomTree>(MBBDomTree &DT,
                                              MBBDomTreeGraphDiff &,
                                              MBBDomTreeGraphDiff *);

extern template bool Verify<MBBDomTree>(const MBBDomTree &DT,
                                        MBBDomTree::VerificationLevel VL);
} // namespace DomTreeBuilder

//===-------------------------------------
/// DominatorTree Class - Concrete subclass of DominatorTreeBase that is used to
/// compute a normal dominator tree.
///
class MachineDominatorTree : public DomTreeBase<MachineBasicBlock> {

public:
  using Base = DomTreeBase<MachineBasicBlock>;

  MachineDominatorTree() = default;
  explicit MachineDominatorTree(MachineFunction &MF) { recalculate(MF); }

  /// Handle invalidation explicitly.
  bool invalidate(MachineFunction &, const PreservedAnalyses &PA,
                  MachineFunctionAnalysisManager::Invalidator &);

  using Base::dominates;

  // dominates - Return true if A dominates B. This performs the
  // special checks necessary if A and B are in the same basic block.
  bool dominates(const MachineInstr *A, const MachineInstr *B) const {
    const MachineBasicBlock *BBA = A->getParent(), *BBB = B->getParent();
    if (BBA != BBB)
      return Base::dominates(BBA, BBB);

    // Loop through the basic block until we find A or B.
    MachineBasicBlock::const_iterator I = BBA->begin();
    for (; &*I != A && &*I != B; ++I)
      /*empty*/ ;

    return &*I == A;
  }
};

/// \brief Analysis pass which computes a \c MachineDominatorTree.
class MachineDominatorTreeAnalysis
    : public AnalysisInfoMixin<MachineDominatorTreeAnalysis> {
  friend AnalysisInfoMixin<MachineDominatorTreeAnalysis>;

  static AnalysisKey Key;

public:
  using Result = MachineDominatorTree;

  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &);
};

/// \brief Machine function pass which print \c MachineDominatorTree.
class MachineDominatorTreePrinterPass
    : public PassInfoMixin<MachineDominatorTreePrinterPass> {
  raw_ostream &OS;

public:
  explicit MachineDominatorTreePrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

/// \brief Analysis pass which computes a \c MachineDominatorTree.
class MachineDominatorTreeWrapperPass : public MachineFunctionPass {
  // MachineFunctionPass may verify the analysis result without running pass,
  // e.g. when `F.hasAvailableExternallyLinkage` is true.
  std::optional<MachineDominatorTree> DT;

public:
  static char ID;

  MachineDominatorTreeWrapperPass();

  MachineDominatorTree &getDomTree() { return *DT; }
  const MachineDominatorTree &getDomTree() const { return *DT; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void verifyAnalysis() const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  void releaseMemory() override;

  void print(raw_ostream &OS, const Module *M = nullptr) const override;
};

//===-------------------------------------
/// DominatorTree GraphTraits specialization so the DominatorTree can be
/// iterable by generic graph iterators.
///

template <class Node, class ChildIterator>
struct MachineDomTreeGraphTraitsBase {
  using NodeRef = Node *;
  using ChildIteratorType = ChildIterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->end(); }
};

template <class T> struct GraphTraits;

template <>
struct GraphTraits<MachineDomTreeNode *>
    : public MachineDomTreeGraphTraitsBase<MachineDomTreeNode,
                                           MachineDomTreeNode::const_iterator> {
};

template <>
struct GraphTraits<const MachineDomTreeNode *>
    : public MachineDomTreeGraphTraitsBase<const MachineDomTreeNode,
                                           MachineDomTreeNode::const_iterator> {
};

template <> struct GraphTraits<MachineDominatorTree*>
  : public GraphTraits<MachineDomTreeNode *> {
  static NodeRef getEntryNode(MachineDominatorTree *DT) {
    return DT->getRootNode();
  }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEDOMINATORS_H
