//===-- EHContGuardTargets.cpp - EH continuation target symbols -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains a machine function pass to insert a symbol before each
/// valid target where the unwinder in Windows may continue exectution after an
/// exception is thrown and store this in the MachineFunction's EHContTargets
/// vector. This will be used to emit the table of valid targets used by Windows
/// EH Continuation Guard.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "ehcontguard-catchret"

STATISTIC(EHContGuardTargetsFound, "Number of EHCont Guard targets");

namespace {

/// MachineFunction pass to insert a symbol before each valid catchret target
/// and store these in the MachineFunction's CatchRetTargets vector.
class EHContGuardTargets : public MachineFunctionPass {
public:
  static char ID;

  EHContGuardTargets() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "EH Cont Guard catchret targets";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

char EHContGuardTargets::ID = 0;

INITIALIZE_PASS(EHContGuardTargets, "EHContGuardTargets",
                "Insert symbols at valid targets for /guard:ehcont", false,
                false)
FunctionPass *llvm::createEHContGuardTargetsPass() {
  return new EHContGuardTargets();
}

bool EHContGuardTargets::runOnMachineFunction(MachineFunction &MF) {

  // Skip modules for which the ehcontguard flag is not set.
  if (!MF.getFunction().getParent()->getModuleFlag("ehcontguard"))
    return false;

  // Skip functions that do not have targets
  if (!MF.hasEHContTarget())
    return false;

  bool Result = false;

  for (MachineBasicBlock &MBB : MF) {
    if (MBB.isEHContTarget()) {
      MF.addEHContTarget(MBB.getEHContSymbol());
      EHContGuardTargetsFound++;
      Result = true;
    }
  }

  return Result;
}
