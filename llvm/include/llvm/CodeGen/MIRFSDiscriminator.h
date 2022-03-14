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

#include "llvm/ADT/StringRef.h"
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

using namespace sampleprof;
class MIRAddFSDiscriminators : public MachineFunctionPass {
  MachineFunction *MF;
  unsigned LowBit;
  unsigned HighBit;

public:
  static char ID;
  /// PassNum is the sequence number this pass is called, start from 1.
  MIRAddFSDiscriminators(FSDiscriminatorPass P = FSDiscriminatorPass::Pass1)
      : MachineFunctionPass(ID) {
    LowBit = getFSPassBitBegin(P);
    HighBit = getFSPassBitEnd(P);
    assert(LowBit < HighBit && "HighBit needs to be greater than Lowbit");
  }

  StringRef getPassName() const override {
    return "Add FS discriminators in MIR";
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
