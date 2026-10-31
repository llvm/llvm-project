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

class LLVM_LIBRARY_VISIBILITY ErrataWorkaround : public MachineFunctionPass {
protected:
  const SparcSubtarget *ST;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;

  bool checkSeqTN0009A(MachineBasicBlock::iterator I);
  bool checkSeqTN0009B(MachineBasicBlock::iterator I);
  bool checkSeqTN0010First(MachineBasicBlock &MBB);
  bool checkSeqTN0010(MachineBasicBlock::iterator I);
  bool checkSeqTN0012(MachineBasicBlock::iterator I);
  bool checkSeqTN0013(MachineBasicBlock::iterator I);

  bool moveNext(MachineBasicBlock::iterator &I);
  bool isFloat(MachineBasicBlock::iterator I);
  bool isDivSqrt(MachineBasicBlock::iterator I);
  bool isAtomic(MachineBasicBlock::iterator I);
  void insertNop(MachineBasicBlock::iterator I);

public:
  static char ID;

  ErrataWorkaround();
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "Errata workaround pass"; };
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

class LLVM_LIBRARY_VISIBILITY DetectRoundChange : public MachineFunctionPass {
public:
  static char ID;

  DetectRoundChange();
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "DetectRoundChange: Leon erratum detection: detect any rounding "
           "mode change request: use only the round-to-nearest rounding mode";
  }
};

class LLVM_LIBRARY_VISIBILITY FixAllFDIVSQRT : public MachineFunctionPass {
public:
  static char ID;

  FixAllFDIVSQRT();
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "FixAllFDIVSQRT: Erratum Fix LBR34: fix FDIVS/FDIVD/FSQRTS/FSQRTD "
           "instructions with NOPs and floating-point store";
  }
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_SPARC_LEON_PASSES_H
