//===- X86GlobalBaseReg.cpp - PIC Global Base Register Initialization -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the pass that initializes the PIC global base register
// for x86-32.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "x86-global-base-reg"

namespace {
class X86GlobalBaseRegLegacy : public MachineFunctionPass {
public:
  static char ID;
  X86GlobalBaseRegLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 PIC Global Base Reg Initialization";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

char X86GlobalBaseRegLegacy::ID = 0;

FunctionPass *llvm::createX86GlobalBaseRegLegacyPass() {
  return new X86GlobalBaseRegLegacy();
}

bool X86GlobalBaseRegLegacy::runOnMachineFunction(MachineFunction &MF) {
  return MF.getInfo<X86MachineFunctionInfo>()->initGlobalBaseReg(MF);
}

PreservedAnalyses
X86GlobalBaseRegPass::run(MachineFunction &MF,
                          MachineFunctionAnalysisManager &MFAM) {
  return MF.getInfo<X86MachineFunctionInfo>()->initGlobalBaseReg(MF)
             ? getMachineFunctionPassPreservedAnalyses()
                   .preserveSet<CFGAnalyses>()
             : PreservedAnalyses::all();
}
