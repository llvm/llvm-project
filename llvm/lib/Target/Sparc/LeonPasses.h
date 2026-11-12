//===------- LeonPasses.h - Define passes specific to LEON ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPARC_LEON_PASSES_H
#define LLVM_LIB_TARGET_SPARC_LEON_PASSES_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Pass.h"

namespace llvm {
class SparcSubtarget;

class LLVM_LIBRARY_VISIBILITY SparcErrataWorkaroundLegacy
    : public MachineFunctionPass {
public:
  static char ID;

  SparcErrataWorkaroundLegacy();
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "Errata workaround pass"; };

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class LLVM_LIBRARY_VISIBILITY SparcInsertNOPLoadLegacy
    : public MachineFunctionPass {
public:
  static char ID;

  SparcInsertNOPLoadLegacy();
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "InsertNOPLoad: Erratum Fix LBR35: insert a NOP instruction after "
           "every single-cycle load instruction when the next instruction is "
           "another load/store instruction";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class LLVM_LIBRARY_VISIBILITY SparcDetectRoundChangeLegacy
    : public MachineFunctionPass {
public:
  static char ID;

  SparcDetectRoundChangeLegacy();
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "DetectRoundChange: Leon erratum detection: detect any rounding "
           "mode change request: use only the round-to-nearest rounding mode";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class LLVM_LIBRARY_VISIBILITY SparcFixAllFDIVSQRTLegacy
    : public MachineFunctionPass {
public:
  static char ID;

  SparcFixAllFDIVSQRTLegacy();
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "FixAllFDIVSQRT: Erratum Fix LBR34: fix FDIVS/FDIVD/FSQRTS/FSQRTD "
           "instructions with NOPs and floating-point store";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_SPARC_LEON_PASSES_H
