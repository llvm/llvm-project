//===-- FuncletLayout.cpp - Contiguously lay out funclets -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic block placement transformations which result in
// funclets being contiguous.
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/FuncletLayout.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
using namespace llvm;

#define DEBUG_TYPE "funclet-layout"

static bool runFuncletLayout(MachineFunction &F) {
  // Even though this gets information from getEHScopeMembership(), this pass is
  // only necessary for funclet-based EH personalities, in which these EH scopes
  // are outlined at the end.
  DenseMap<const MachineBasicBlock *, int> FuncletMembership =
      getEHScopeMembership(F);
  if (FuncletMembership.empty())
    return false;

  F.sort([&](MachineBasicBlock &X, MachineBasicBlock &Y) {
    auto FuncletX = FuncletMembership.find(&X);
    auto FuncletY = FuncletMembership.find(&Y);
    assert(FuncletX != FuncletMembership.end());
    assert(FuncletY != FuncletMembership.end());
    return FuncletX->second < FuncletY->second;
  });

  // Conservatively assume we changed something.
  return true;
}

namespace {
class FuncletLayoutLegacy : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  FuncletLayoutLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &F) override {
    return runFuncletLayout(F);
  }
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
};
} // namespace

char FuncletLayoutLegacy::ID = 0;
char &llvm::FuncletLayoutID = FuncletLayoutLegacy::ID;
INITIALIZE_PASS(FuncletLayoutLegacy, DEBUG_TYPE,
                "Contiguously Lay Out Funclets", false, false)

PreservedAnalyses FuncletLayoutPass::run(MachineFunction &MF,
                                         MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);
  if (!runFuncletLayout(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses();
}
