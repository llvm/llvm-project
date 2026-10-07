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

#include "llvm/CodeGen/PatchableFunction.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {
struct PatchableFunction {
  bool run(MachineFunction &F);
};

struct PatchableFunctionLegacy : public MachineFunctionPass {
  static char ID;
  PatchableFunctionLegacy() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &F) override {
    return PatchableFunction().run(F);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<MachineRegisterClassInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

PreservedAnalyses
PatchableFunctionPass::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);
  if (!PatchableFunction().run(MF))
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses();
}

// AArch64 instructions are atomically patchable, but an entry block that emits
// no code lets a branch to the next block (e.g. a loop backedge) target the
// function's first instruction. Pad it with a NOP, placed after any zero-byte
// instructions (e.g. an empty Windows prologue). Like x86, also pad when the
// entry starts with inline assembly, which may define a branch target.
static bool padEmptyEntryBlock(MachineFunction &MF) {
  MachineBasicBlock &MBB = MF.front();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  auto FirstEmitting = llvm::find_if(MBB, [&](const MachineInstr &MI) {
    return MI.isInlineAsm() || TII->getInstSizeInBytes(MI) != 0;
  });
  if (FirstEmitting != MBB.end() && !FirstEmitting->isInlineAsm())
    return false;
  TII->insertNoop(MBB, FirstEmitting);
  return true;
}

bool PatchableFunction::run(MachineFunction &MF) {
  MachineBasicBlock &FirstMBB = *MF.begin();
  bool IsAArch64 = MF.getTarget().getTargetTriple().isAArch64();
  bool HasPatchableEntry =
      MF.getFunction().hasFnAttribute("patchable-function-entry");
  bool Changed = false;

  if (HasPatchableEntry) {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    // The initial .loc covers PATCHABLE_FUNCTION_ENTER.
    BuildMI(FirstMBB, FirstMBB.begin(), DebugLoc(),
            TII->get(TargetOpcode::PATCHABLE_FUNCTION_ENTER));
    Changed = true;
  }

  if (!MF.getFunction().hasFnAttribute("patchable-function"))
    return Changed;

#ifndef NDEBUG
  Attribute PatchAttr = MF.getFunction().getFnAttribute("patchable-function");
  StringRef PatchType = PatchAttr.getValueAsString();
  assert(PatchType == "prologue-short-redirect" && "Only possibility today!");
#endif

  // On AArch64 this is needed even with patchable-function-entry, which may be
  // "0" (N,N), and the padding must be computed after PATCHABLE_FUNCTION_ENTER
  // has been inserted so that its size is accounted for.
  if (IsAArch64)
    return padEmptyEntryBlock(MF) || Changed;

  if (!HasPatchableEntry) {
    auto *TII = MF.getSubtarget().getInstrInfo();
    BuildMI(FirstMBB, FirstMBB.begin(), DebugLoc(),
            TII->get(TargetOpcode::PATCHABLE_OP))
        .addImm(2);
    MF.ensureAlignment(Align(16));
    Changed = true;
  }
  return Changed;
}

char PatchableFunctionLegacy::ID = 0;
char &llvm::PatchableFunctionID = PatchableFunctionLegacy::ID;
INITIALIZE_PASS(PatchableFunctionLegacy, "patchable-function",
                "Implement the 'patchable-function' attribute", false, false)
