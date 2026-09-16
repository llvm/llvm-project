//== llvm/CodeGen/GlobalISel/Legalizer.h ---------------- -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file A pass to convert the target-illegal operations created by IR -> MIR
/// translation into ones the target expects to be able to select. This may
/// occur in multiple phases, for example G_ADD <2 x i8> -> G_ADD <2 x i16> ->
/// G_ADD <4 x i16>.
///
/// The LegalizeHelper class is where most of the work happens, and is designed
/// to be callable from other passes that find themselves with an illegal
/// instruction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_LEGALIZER_H
#define LLVM_CODEGEN_GLOBALISEL_LEGALIZER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class LegalizerInfo;
class MachineIRBuilder;
class MachineInstr;
class GISelChangeObserver;
class LibcallLoweringInfo;
class LostDebugLocObserver;

struct LegalizerMFResult {
  bool Changed;
  const MachineInstr *FailedOn;
};

LegalizerMFResult legalizeMachineFunction(
    MachineFunction &MF, const LegalizerInfo &LI,
    ArrayRef<GISelChangeObserver *> AuxObservers,
    LostDebugLocObserver &LocObserver, MachineIRBuilder &MIRBuilder,
    const LibcallLoweringInfo *Libcalls, GISelValueTracking *VT);

class LLVM_ABI LegalizerLegacy : public MachineFunctionPass {
public:
  static char ID;

public:
  // Ctor, nothing fancy.
  LegalizerLegacy();

  StringRef getPassName() const override { return "Legalizer"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().setLegalized();
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setNoPHIs().setNoVRegs();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

class LegalizerPass : public RequiredPassInfoMixin<LegalizerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }

  MachineFunctionProperties getSetProperties() const {
    return MachineFunctionProperties().setLegalized();
  }

  MachineFunctionProperties getClearedProperties() const {
    return MachineFunctionProperties().setNoPHIs().setNoVRegs();
  }
};

} // End namespace llvm.

#endif
