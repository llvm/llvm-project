//===- X86LFIRewritePass.cpp - Modify code generation for LFI ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86LFIRewritePass, which prepares machine code for
// LFI sandboxing by making sure that every address which may legitimately be
// the destination of an indirect branch is aligned to a bundle boundary.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86MCLFIRewriter.h"
#include "X86.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

static constexpr Align BundleAlign = Align::Constant<X86::LFIBundleSize>();

namespace {
class X86LFIRewriteLegacy : public MachineFunctionPass {
public:
  static char ID;
  X86LFIRewriteLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "X86 LFI rewrites"; }
};
} // namespace

char X86LFIRewriteLegacy::ID = 0;

static void alignToBundle(MachineBasicBlock &MBB) {
  MBB.setAlignment(std::max(MBB.getAlignment(), BundleAlign), /*MaxBytes=*/0);
}

// Returns true if MBB may be reached by an indirect branch (does not include
// jump table targets).
static bool isIndirectlyReachable(MachineFunction &MF,
                                  const MachineBasicBlock &MBB) {
  if (MBB.hasAddressTaken() || MBB.isEHPad())
    return true;

  // With SJLJ exception handling, the dispatch block jumps indirectly to the
  // block holding the call site's landing pad label, which is no longer marked
  // as an EH pad by that point.
  if (MF.getTarget().Options.ExceptionModel == ExceptionHandling::SjLj)
    for (const MachineInstr &MI : MBB)
      if (MI.isEHLabel() &&
          MF.hasCallSiteLandingPad(MI.getOperand(0).getMCSymbol()))
        return true;

  return false;
}

static void alignIndirectBranchTargets(MachineFunction &MF) {
  // Function entry points are reachable through function pointers.
  MF.ensureAlignment(BundleAlign);

  // Blocks that are the target of a jump table are not considered
  // address-taken by LLVM, but they are still reached by an indirect branch.
  if (const MachineJumpTableInfo *JTI = MF.getJumpTableInfo())
    for (const MachineJumpTableEntry &JTE : JTI->getJumpTables())
      for (MachineBasicBlock *MBB : JTE.MBBs)
        alignToBundle(*MBB);

  for (MachineBasicBlock &MBB : MF)
    if (isIndirectlyReachable(MF, MBB))
      alignToBundle(MBB);
}

bool X86LFIRewriteLegacy::runOnMachineFunction(MachineFunction &MF) {
  alignIndirectBranchTargets(MF);
  return true;
}

PreservedAnalyses X86LFIRewritePass::run(MachineFunction &MF,
                                         MachineFunctionAnalysisManager &) {
  alignIndirectBranchTargets(MF);
  return PreservedAnalyses::all();
}

FunctionPass *llvm::createX86LFIRewritePass() {
  return new X86LFIRewriteLegacy();
}
