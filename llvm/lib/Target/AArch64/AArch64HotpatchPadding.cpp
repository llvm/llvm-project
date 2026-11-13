//===-- AArch64HotpatchPadding.cpp - Keep function entries patchable ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements the "patchable-function"="prologue-short-redirect"
// attribute used by /hotpatch. AArch64 instructions are atomically patchable,
// but an entry block that emits no code lets a branch to the next block (e.g. a
// loop backedge) target the function's first instruction, which a hotpatch
// overwrites. Such entry blocks are padded with a NOP, placed after any
// zero-byte instructions (e.g. an empty Windows prologue). Like on x86, an
// entry that starts with inline assembly is also padded, since the assembly may
// define a branch target.
//
// This must run after every pass that may delete instructions, and before
// branch relaxation, which must account for the NOP.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-hotpatch-padding"
#define AARCH64_HOTPATCH_PADDING_NAME "AArch64 Hotpatch Padding"

namespace {
class AArch64HotpatchPaddingLegacy : public MachineFunctionPass {
public:
  static char ID;
  AArch64HotpatchPaddingLegacy() : MachineFunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override {
    return AARCH64_HOTPATCH_PADDING_NAME;
  }
};
} // end anonymous namespace

char AArch64HotpatchPaddingLegacy::ID = 0;

INITIALIZE_PASS(AArch64HotpatchPaddingLegacy, DEBUG_TYPE,
                AARCH64_HOTPATCH_PADDING_NAME, false, false)

FunctionPass *llvm::createAArch64HotpatchPaddingPass() {
  return new AArch64HotpatchPaddingLegacy();
}

static bool padEmptyEntryBlock(MachineFunction &MF) {
  if (MF.getFunction()
          .getFnAttribute("patchable-function")
          .getValueAsString() != "prologue-short-redirect")
    return false;

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

bool AArch64HotpatchPaddingLegacy::runOnMachineFunction(MachineFunction &MF) {
  return padEmptyEntryBlock(MF);
}

PreservedAnalyses
AArch64HotpatchPaddingPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  if (!padEmptyEntryBlock(MF))
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
