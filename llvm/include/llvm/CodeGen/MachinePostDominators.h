//===- llvm/CodeGen/MachinePostDominators.h ----------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes interfaces to post dominance information for
// target-specific code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEPOSTDOMINATORS_H
#define LLVM_CODEGEN_MACHINEPOSTDOMINATORS_H

#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <memory>

namespace llvm {

///
/// MachinePostDominatorTree - an analysis pass wrapper for DominatorTree
/// used to compute the post-dominator tree for MachineFunctions.
///
class MachinePostDominatorTree : public MachineFunctionPass {
  using PostDomTreeT = PostDomTreeBase<MachineBasicBlock>;
  std::unique_ptr<PostDomTreeT> PDT;

public:
  static char ID;

  MachinePostDominatorTree();

  FunctionPass *createMachinePostDominatorTreePass();

  const SmallVectorImpl<MachineBasicBlock *> &getRoots() const {
    return PDT->getRoots();
  }

  MachineDomTreeNode *getRootNode() const { return PDT->getRootNode(); }

  MachineDomTreeNode *operator[](MachineBasicBlock *BB) const {
    return PDT->getNode(BB);
  }

  MachineDomTreeNode *getNode(MachineBasicBlock *BB) const {
    return PDT->getNode(BB);
  }

  bool dominates(const MachineDomTreeNode *A,
                 const MachineDomTreeNode *B) const {
    return PDT->dominates(A, B);
  }

  bool dominates(const MachineBasicBlock *A, const MachineBasicBlock *B) const {
    return PDT->dominates(A, B);
  }

  bool properlyDominates(const MachineDomTreeNode *A,
                         const MachineDomTreeNode *B) const {
    return PDT->properlyDominates(A, B);
  }

  bool properlyDominates(const MachineBasicBlock *A,
                         const MachineBasicBlock *B) const {
    return PDT->properlyDominates(A, B);
  }

  bool isVirtualRoot(const MachineDomTreeNode *Node) const {
    return PDT->isVirtualRoot(Node);
  }

  MachineBasicBlock *findNearestCommonDominator(MachineBasicBlock *A,
                                                MachineBasicBlock *B) const {
    return PDT->findNearestCommonDominator(A, B);
  }

  /// Returns the nearest common dominator of the given blocks.
  /// If that tree node is a virtual root, a nullptr will be returned.
  MachineBasicBlock *
  findNearestCommonDominator(ArrayRef<MachineBasicBlock *> Blocks) const;

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override { PDT.reset(nullptr); }
  void verifyAnalysis() const override;
  void print(llvm::raw_ostream &OS, const Module *M = nullptr) const override;
};
} //end of namespace llvm

#endif
