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

class MachineDominanceFrontier
    : public DominanceFrontierBase<MachineBasicBlock, false> {
public:
 using DomTreeT = DomTreeBase<MachineBasicBlock>;
 using DomTreeNodeT = DomTreeNodeBase<MachineBasicBlock>;
 using DomSetType = MachineDominanceFrontier::DomSetType;
 using iterator = MachineDominanceFrontier::iterator;
 using const_iterator = MachineDominanceFrontier ::const_iterator;

 MachineDominanceFrontier() = default;

 bool invalidate(MachineFunction &F, const PreservedAnalyses &PA,
                 MachineFunctionAnalysisManager::Invalidator &);
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

public:
  using Result = MachineDominanceFrontier;

  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEDOMINANCEFRONTIER_H
