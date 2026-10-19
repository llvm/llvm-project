//===-- llvm/CodeGen/LiveVariables.h - Live Variable Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a defunct analysis and should not be used. This pass should be
// deleted. It's sole function is now to made adjustments to dead flags and
// should be deleted once later passes are fixed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEVARIABLES_H
#define LLVM_CODEGEN_LIVEVARIABLES_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MachineBasicBlock;
class MachineRegisterInfo;

class LiveVariables {
  friend class LiveVariablesWrapperPass;

  MachineRegisterInfo *MRI = nullptr;

  const TargetRegisterInfo *TRI = nullptr;

  // PhysRegInfo - Keep track of which instruction was the last def of a
  // physical register. This is a purely local property, because all physical
  // register references are presumed dead across basic blocks.
  std::vector<MachineInstr *> PhysRegDef;

  // PhysRegInfo - Keep track of which instruction was the last use of a
  // physical register. This is a purely local property, because all physical
  // register references are presumed dead across basic blocks.
  std::vector<MachineInstr *> PhysRegUse;

  /// Track physical registers referenced in the current block, used as a
  /// compile-time guard.
  BitVector TrackedRegs;

  // DistanceMap - Keep track the distance of a MI from the start of the
  // current basic block.
  DenseMap<MachineInstr*, unsigned> DistanceMap;

  // For legacy pass.
  LiveVariables() = default;

  LLVM_ABI void analyze(MachineFunction &MF);

  /// HandlePhysRegKill - Mark the last def of Reg and its sub-registers dead
  /// if they are not used. Pay special attention to the sub-register uses
  /// which may come below the last use of the whole register.
  void HandlePhysRegKill(Register Reg, MachineInstr *MI);

  /// HandleRegMask - Call HandlePhysRegKill for all registers clobbered by Mask.
  void HandleRegMask(const MachineOperand &, unsigned);

  void HandlePhysRegUse(Register Reg, MachineInstr &MI);
  void HandlePhysRegDef(Register Reg, MachineInstr *MI);
  void UpdatePhysRegDefs(MachineInstr &MI, ArrayRef<Register> Defs);

  /// FindLastRefOrPartRef - Return the last reference or partial reference of
  /// the specified register.
  MachineInstr *FindLastRefOrPartRef(Register Reg);

  /// FindLastPartialDef - Return the last partial def of the specified
  /// register.
  MachineInstr *FindLastPartialDef(Register Reg);

  void runOnInstr(MachineInstr &MI, unsigned NumRegs);

  void runOnBlock(MachineBasicBlock *MBB, unsigned NumRegs);

public:
  LLVM_ABI LiveVariables(MachineFunction &MF);
};

class LiveVariablesAnalysis : public AnalysisInfoMixin<LiveVariablesAnalysis> {
  friend AnalysisInfoMixin<LiveVariablesAnalysis>;
  LLVM_ABI static AnalysisKey Key;

public:
  using Result = LiveVariables;
  LLVM_ABI Result run(MachineFunction &MF, MachineFunctionAnalysisManager &);
};

class LLVM_ABI LiveVariablesWrapperPass : public MachineFunctionPass {
  LiveVariables LV;

public:
  static char ID; // Pass identification, replacement for typeid

  LiveVariablesWrapperPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    LV.analyze(MF);
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // End llvm namespace

#endif
