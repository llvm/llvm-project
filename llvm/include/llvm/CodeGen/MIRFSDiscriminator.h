//===----- MIRFSDiscriminator.h: MIR FS Discriminator Support --0-- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the supporting functions for adding Machine level IR
// Flow Sensitive discriminators to the instruction debug information. With
// this, a cloned machine instruction in a different MachineBasicBlock will
// have its own discriminator value. This is done in a MIRAddFSDiscriminators
// pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MIRFSDISCRIMINATOR_H
#define LLVM_CODEGEN_MIRFSDISCRIMINATOR_H

#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/ProfileData/SampleProfReader.h"

#include <cassert>

namespace llvm {

class MIRAddFSDiscriminators : public MachineFunctionPass {
  MachineFunction *MF;
  unsigned LowBit;
  unsigned HighBit;

public:
  static char ID;
  /// FS bits that will be used in this pass (numbers are 0 based and
  /// inclusive).
  MIRAddFSDiscriminators(unsigned LowBit = 0, unsigned HighBit = 0)
      : MachineFunctionPass(ID), LowBit(LowBit), HighBit(HighBit) {
    assert(LowBit < HighBit && "HighBit needs to be greater than Lowbit");
  }

  /// getNumFSBBs() - Return the number of machine BBs that have FS samples.
  unsigned getNumFSBBs();

  /// getNumFSSamples() - Return the number of samples that have flow sensitive
  /// values.
  uint64_t getNumFSSamples();

  /// getMachineFunction - Return the current machine function.
  const MachineFunction *getMachineFunction() const { return MF; }

private:
  bool runOnMachineFunction(MachineFunction &) override;
};

} // namespace llvm

#endif // LLVM_CODEGEN_MIRFSDISCRIMINATOR_H
