//===- MachineRegisterClassInfo.cpp - Machine Register Class Info ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis calculates register class info via RegisterClassInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineRegisterClassInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

INITIALIZE_PASS(MachineRegisterClassInfoWrapperPass, "machine-reg-class-info",
                "Machine Register Class Info Analysis", true, true)

MachineRegisterClassInfoAnalysis::Result
MachineRegisterClassInfoAnalysis::run(MachineFunction &MF,
                                      MachineFunctionAnalysisManager &) {
  RegisterClassInfo RCI;
  RCI.runOnMachineFunction(MF);
  return RCI;
}

char MachineRegisterClassInfoWrapperPass::ID = 0;

MachineRegisterClassInfoWrapperPass::MachineRegisterClassInfoWrapperPass()
    : MachineFunctionPass(ID), RCI() {
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeMachineRegisterClassInfoWrapperPassPass(Registry);
}

bool MachineRegisterClassInfoWrapperPass::runOnMachineFunction(
    MachineFunction &MF) {
  RCI.runOnMachineFunction(MF);
  return false;
}

void MachineRegisterClassInfoWrapperPass::anchor() {}

AnalysisKey MachineRegisterClassInfoAnalysis::Key;
