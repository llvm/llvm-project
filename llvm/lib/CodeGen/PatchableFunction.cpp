//===-- PatchableFunction.cpp - Patchable prologues for LLVM -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements edits function bodies in place to support the
// "patchable-function" attribute.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

using namespace llvm;

namespace {
struct PatchableFunction : public MachineFunctionPass {
  static char ID; // Pass identification, replacement for typeid
  PatchableFunction() : MachineFunctionPass(ID) {
    initializePatchableFunctionPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &F) override;
   MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }
};
}

bool PatchableFunction::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getFunction().hasFnAttribute("patchable-function-entry")) {
    MachineBasicBlock &FirstMBB = *MF.begin();
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    // The initial .loc covers PATCHABLE_FUNCTION_ENTER.
    BuildMI(FirstMBB, FirstMBB.begin(), DebugLoc(),
            TII->get(TargetOpcode::PATCHABLE_FUNCTION_ENTER));
    return true;
  }

  if (!MF.getFunction().hasFnAttribute("patchable-function"))
    return false;

#ifndef NDEBUG
  Attribute PatchAttr = MF.getFunction().getFnAttribute("patchable-function");
  StringRef PatchType = PatchAttr.getValueAsString();
  assert(PatchType == "prologue-short-redirect" && "Only possibility today!");
#endif

  auto &FirstMBB = *MF.begin();
  auto *TII = MF.getSubtarget().getInstrInfo();

  MachineBasicBlock::iterator FirstActualI = llvm::find_if(
      FirstMBB, [](const MachineInstr &MI) { return !MI.isMetaInstruction(); });

  if (FirstActualI == FirstMBB.end()) {
    // As of Microsoft documentation on /hotpatch feature, we must ensure that
    // "the first instruction of each function is at least two bytes, and no
    // jump within the function goes to the first instruction"

    // When the first MBB is empty, insert a patchable no-op. This ensures the
    // first instruction is patchable in two special cases:
    // - the function is empty (e.g. unreachable)
    // - the function jumps back to the first instruction, which is in a
    // successor MBB.
    BuildMI(&FirstMBB, DebugLoc(), TII->get(TargetOpcode::PATCHABLE_OP))
        .addImm(2)
        .addImm(TargetOpcode::PATCHABLE_OP);
    MF.ensureAlignment(Align(16));
    return true;
  }

  auto MIB = BuildMI(FirstMBB, FirstActualI, FirstActualI->getDebugLoc(),
                     TII->get(TargetOpcode::PATCHABLE_OP))
                 .addImm(2)
                 .addImm(FirstActualI->getOpcode());

  for (auto &MO : FirstActualI->operands())
    MIB.add(MO);

  FirstActualI->eraseFromParent();
  MF.ensureAlignment(Align(16));
  return true;
}

char PatchableFunction::ID = 0;
char &llvm::PatchableFunctionID = PatchableFunction::ID;
INITIALIZE_PASS(PatchableFunction, "patchable-function",
                "Implement the 'patchable-function' attribute", false, false)
