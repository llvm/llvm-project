//===- llvm/CodeGen/MachineDominanceFrontier.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEDOMINANCEFRONTIER_H
#define LLVM_CODEGEN_MACHINEDOMINANCEFRONTIER_H

#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/DominanceFrontierImpl.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/GenericDomTree.h"

namespace llvm {

class MachineDominanceFrontier {
  ForwardDominanceFrontierBase<MachineBasicBlock> Base;

public:
 using DomTreeT = DomTreeBase<MachineBasicBlock>;
 using DomTreeNodeT = DomTreeNodeBase<MachineBasicBlock>;
 using DomSetType = DominanceFrontierBase<MachineBasicBlock, false>::DomSetType;
 using iterator = DominanceFrontierBase<MachineBasicBlock, false>::iterator;
 using const_iterator =
     DominanceFrontierBase<MachineBasicBlock, false>::const_iterator;

 MachineDominanceFrontier() = default;

 ForwardDominanceFrontierBase<MachineBasicBlock> &getBase() { return Base; }

 const SmallVectorImpl<MachineBasicBlock *> &getRoots() const {
   return Base.getRoots();
  }

  MachineBasicBlock *getRoot() const {
    return Base.getRoot();
  }

  bool isPostDominator() const {
    return Base.isPostDominator();
  }

  iterator begin() {
    return Base.begin();
  }

  const_iterator begin() const {
    return Base.begin();
  }

  iterator end() {
    return Base.end();
  }

  const_iterator end() const {
    return Base.end();
  }

  iterator find(MachineBasicBlock *B) {
    return Base.find(B);
  }

  const_iterator find(MachineBasicBlock *B) const {
    return Base.find(B);
  }

  bool analyze(MachineDominatorTree &MDT);

  void releaseMemory();
};

class MachineDominanceFrontierWrapperPass : public MachineFunctionPass {
private:
  MachineDominanceFrontier MDF;

public:
  MachineDominanceFrontierWrapperPass();

  MachineDominanceFrontierWrapperPass(
      const MachineDominanceFrontierWrapperPass &) = delete;
  MachineDominanceFrontierWrapperPass &
  operator=(const MachineDominanceFrontierWrapperPass &) = delete;

  static char ID;

  bool runOnMachineFunction(MachineFunction &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void releaseMemory() override;

  MachineDominanceFrontier &getMDF() { return MDF; }
};

class MachineDominanceFrontierAnalysis
    : public AnalysisInfoMixin<MachineDominanceFrontierAnalysis> {
  friend AnalysisInfoMixin<MachineDominanceFrontierAnalysis>;
  static AnalysisKey Key;
  MachineDominanceFrontier MDF;

public:
  using Result = MachineDominanceFrontier;

  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEDOMINANCEFRONTIER_H
