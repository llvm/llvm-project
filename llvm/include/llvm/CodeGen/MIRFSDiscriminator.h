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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/Discriminator.h"

#include <cassert>
#include <cstdint>

namespace llvm {
class MachineFunction;

using namespace sampleprof;
class MIRAddFSDiscriminators {
  MachineFunction *MF = nullptr;
  FSDiscriminatorPass Pass;
  unsigned LowBit;
  unsigned HighBit;

public:
  /// PassNum is the sequence number this pass is called, start from 1.
  MIRAddFSDiscriminators(FSDiscriminatorPass P = FSDiscriminatorPass::Pass1)
      : Pass(P) {
    LowBit = getFSPassBitBegin(P);
    HighBit = getFSPassBitEnd(P);
    assert(LowBit < HighBit && "HighBit needs to be greater than Lowbit");
  }

  /// getNumFSBBs() - Return the number of machine BBs that have FS samples.
  unsigned getNumFSBBs();

  /// getNumFSSamples() - Return the number of samples that have flow sensitive
  /// values.
  uint64_t getNumFSSamples();

  /// getMachineFunction - Return the current machine function.
  const MachineFunction *getMachineFunction() const { return MF; }

  bool runOnMachineFunction(MachineFunction &);
};

class MIRAddFSDiscriminatorsLegacy : public MachineFunctionPass {
  FSDiscriminatorPass Pass;

public:
  static char ID;
  MIRAddFSDiscriminatorsLegacy(
      FSDiscriminatorPass P = FSDiscriminatorPass::Pass1)
      : MachineFunctionPass(ID), Pass(P) {}
  StringRef getPassName() const override {
    return "Add FS discriminators in MIR";
  }

private:
  bool runOnMachineFunction(MachineFunction &MF) override;
};

class MIRAddFSDiscriminatorsPass
    : public PassInfoMixin<MIRAddFSDiscriminatorsPass> {
  FSDiscriminatorPass Pass;

public:
  MIRAddFSDiscriminatorsPass(FSDiscriminatorPass P = FSDiscriminatorPass::Pass1)
      : Pass(P) {}
  PreservedAnalyses run(MachineFunction &F, MachineFunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_MIRFSDISCRIMINATOR_H
