//===------- LeonPasses.h - Define passes specific to LEON ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPARC_LEON_PASSES_H
#define LLVM_LIB_TARGET_SPARC_LEON_PASSES_H

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineBasicBlock.h"

#include "Sparc.h"
#include "SparcSubtarget.h"

namespace llvm {
class LLVM_LIBRARY_VISIBILITY LEONMachineFunctionPass
    : public MachineFunctionPass {
protected:
  const SparcSubtarget *Subtarget;

protected:
  LEONMachineFunctionPass(TargetMachine &tm, char& ID);
  LEONMachineFunctionPass(char& ID);
};

class LLVM_LIBRARY_VISIBILITY InsertNOPLoad : public LEONMachineFunctionPass {
public:
  static char ID;

  InsertNOPLoad(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction& MF) override;

  const char *getPassName() const override {
    return "InsertNOPLoad: Erratum Fix LBR35: insert a NOP instruction after every single-cycle load instruction when the next instruction is another load/store instruction";
  }
};
} // namespace llvm

#endif
