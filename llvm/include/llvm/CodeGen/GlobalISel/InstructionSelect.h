//== llvm/CodeGen/GlobalISel/InstructionSelect.h -----------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file This file describes the interface of the MachineFunctionPass
/// responsible for selecting (possibly generic) machine instructions to
/// target-specific instructions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECT_H
#define LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

class InstructionSelector;
class GISelValueTracking;
class BlockFrequencyInfo;
class ProfileSummaryInfo;

/// This pass is responsible for selecting generic machine instructions to
/// target-specific instructions.  It relies on the InstructionSelector provided
/// by the target.
/// Selection is done by examining blocks in post-order, and instructions in
/// reverse order.
///
/// \post for all inst in MF: not isPreISelGenericOpcode(inst.opcode)
class InstructionSelect : public MachineFunctionPass {
public:
  static char ID;
  StringRef getPassName() const override { return "InstructionSelect"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties()
        .setIsSSA()
        .setLegalized()
        .setRegBankSelected();
  }

  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().setSelected();
  }

  InstructionSelect(CodeGenOptLevel OL = CodeGenOptLevel::Default,
                    char &PassID = ID);

  bool runOnMachineFunction(MachineFunction &MF) override;
  bool selectMachineFunction(MachineFunction &MF);
  void setInstructionSelector(InstructionSelector *NewISel) { ISel = NewISel; }

protected:
  class MIIteratorMaintainer;

  InstructionSelector *ISel = nullptr;
  GISelValueTracking *VT = nullptr;
  BlockFrequencyInfo *BFI = nullptr;
  ProfileSummaryInfo *PSI = nullptr;

  CodeGenOptLevel OptLevel = CodeGenOptLevel::None;

  bool selectInstr(MachineInstr &MI);
};
} // End namespace llvm.

#endif
