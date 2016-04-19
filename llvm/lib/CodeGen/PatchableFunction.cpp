//===-- PatchableFunction.cpp - Patchable prologues for LLVM -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements edits function bodies in place to support the
// "patchable-function" attribute.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

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
        MachineFunctionProperties::Property::AllVRegsAllocated);
  }
};
}

bool PatchableFunction::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction()->hasFnAttribute("patchable-function"))
    return false;

#ifndef NDEBUG
  Attribute PatchAttr = MF.getFunction()->getFnAttribute("patchable-function");
  StringRef PatchType = PatchAttr.getValueAsString();
  assert(PatchType == "prologue-short-redirect" && "Only possibility today!");
#endif

  auto &FirstMBB = *MF.begin();
  auto &FirstMI = *FirstMBB.begin();

  auto *TII = MF.getSubtarget().getInstrInfo();
  auto MIB = BuildMI(FirstMBB, FirstMBB.begin(), FirstMI.getDebugLoc(),
                     TII->get(TargetOpcode::PATCHABLE_OP))
                 .addImm(2)
                 .addImm(FirstMI.getOpcode());

  for (auto &MO : FirstMI.operands())
    MIB.addOperand(MO);

  FirstMI.eraseFromParent();
  MF.ensureAlignment(4);
  return true;
}

char PatchableFunction::ID = 0;
char &llvm::PatchableFunctionID = PatchableFunction::ID;
INITIALIZE_PASS(PatchableFunction, "patchable-function",
                "Implement the 'patchable-function' attribute", false, false)
