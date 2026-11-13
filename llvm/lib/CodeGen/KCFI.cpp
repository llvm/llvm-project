//===---- KCFI.cpp - Implements Kernel Control-Flow Integrity (KCFI) ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements Kernel Control-Flow Integrity (KCFI) indirect call
// check lowering. For each call instruction with a cfi-type attribute, it
// emits an arch-specific check before the call and either bundles the check
// with the call (keeping the trap inline) or splits the block to move the trap
// out of line when enabled and supported.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/KCFI.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCDwarf.h"

using namespace llvm;

#define DEBUG_TYPE "kcfi"
#define KCFI_PASS_NAME "Insert KCFI indirect call checks"

STATISTIC(NumKCFIChecksAdded, "Number of indirect call checks added");

static bool hasNonZeroModuleFlag(const Module &M, StringRef Name) {
  auto *Val = mdconst::extract_or_null<ConstantInt>(M.getModuleFlag(Name));
  return Val && !Val->isZero();
}

namespace {
class KCFI {
public:
  bool run(MachineFunction &MF);
  bool didSplitBlocks() const { return SplitBlocks; }

private:
  /// Machine instruction info used throughout the class.
  const TargetInstrInfo *TII = nullptr;

  /// Target lowering for arch-specific parts.
  const TargetLowering *TLI = nullptr;

  /// Whether a failed check branches to a trap in a block of its own, rather
  /// than a passed check branching over a trap that's right before the call.
  bool OutOfLineTraps = false;

  /// Whether run() actually split any block (creating a TrapMBB/SplitBB).
  bool SplitBlocks = false;

  /// Moves \p Check's trap out of line if safe for \p MBBI, returning true if
  /// the block was split.
  bool tryEmitOutOfLineTrap(MachineBasicBlock &MBB,
                            MachineBasicBlock::instr_iterator MBBI,
                            MachineInstr &Check);

  /// Emits a KCFI check before an indirect call.
  void emitCheck(MachineBasicBlock &MBB, MachineBasicBlock::instr_iterator I);
};

class MachineKCFILegacy : public MachineFunctionPass {
public:
  static char ID;

  MachineKCFILegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return KCFI_PASS_NAME; }
  bool runOnMachineFunction(MachineFunction &MF) override {
    return KCFI().run(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<MachineRegisterClassInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char MachineKCFILegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS(MachineKCFILegacy, DEBUG_TYPE, KCFI_PASS_NAME, false, false)

FunctionPass *llvm::createKCFIPass() { return new MachineKCFILegacy(); }

PreservedAnalyses MachineKCFIPass::run(MachineFunction &MF,
                                       MachineFunctionAnalysisManager &MFAM) {
  KCFI Impl;
  if (!Impl.run(MF))
    return PreservedAnalyses::all();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  if (!Impl.didSplitBlocks())
    PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool KCFI::tryEmitOutOfLineTrap(MachineBasicBlock &MBB,
                                MachineBasicBlock::instr_iterator MBBI,
                                MachineInstr &Check) {
  if (!OutOfLineTraps || MBBI->isBundled() || MBB.isEHPad())
    return false;

  MachineFunction &MF = *MBB.getParent();
  SmallVector<MachineInstr *, 2> MoveBeforeCall;
  if (MBB.hasEHPadSuccessor()) {
    // Only split the trailing invoke in an EH block: require no following
    // call/bundle and a single end EH_LABEL after MBBI, plus a single
    // begin EH_LABEL before Check with no intervening call/bundle and no
    // earlier EH_LABEL in MBB.
    unsigned EndLabels = 0;
    for (auto It = std::next(MBBI); It != MBB.instr_end(); ++It) {
      if (It->isCall() || It->isBundled())
        return false;
      if (It->isEHLabel() && ++EndLabels > 1)
        return false;
    }
    if (EndLabels != 1)
      return false;

    MachineInstr *BeginLabel = nullptr;
    for (auto It = Check.getIterator(); It != MBB.instr_begin();) {
      --It;
      if (!BeginLabel) {
        if (It->isCall() || It->isBundled())
          return false;
        if (It->isEHLabel()) {
          BeginLabel = &*It;
          MoveBeforeCall.push_back(&*It);
        } else if (It->isCFIInstruction() &&
                   MF.getFrameInstructions()[It->getOperand(0).getCFIIndex()]
                           .getOperation() == MCCFIInstruction::OpGnuArgsSize) {
          MoveBeforeCall.push_back(&*It);
        }
      } else if (It->isEHLabel()) {
        return false;
      }
    }
    if (!BeginLabel)
      return false;
  }

  MachineBasicBlock *TrapMBB = MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MF.push_back(TrapMBB);
  MachineInstr *Branch = TLI->EmitKCFIOutOfLineTrap(Check, *TrapMBB, TII);

  for (MachineInstr *MI : reverse(MoveBeforeCall))
    MBB.splice(MBBI->getIterator(), &MBB, MI->getIterator());

  MachineBasicBlock *SplitBB = MBB.splitAt(*Branch, /*UpdateLiveIns=*/true);
  assert(SplitBB != &MBB && "Expected a split block before the call");
  (void)SplitBB;
  MBB.setSuccProbability(MBB.succ_begin(), BranchProbability::getOne());
  MBB.addSuccessor(TrapMBB, BranchProbability::getZero());

  SplitBlocks = true;
  return true;
}

void KCFI::emitCheck(MachineBasicBlock &MBB,
                     MachineBasicBlock::instr_iterator MBBI) {
  assert(TII && "Target instruction info was not initialized");
  assert(TLI && "Target lowering was not initialized");

  // If the call instruction is bundled, we can only emit a check safely if
  // it's the first instruction in the bundle.
  if (MBBI->isBundled() && !std::prev(MBBI)->isBundle())
    report_fatal_error("Cannot emit a KCFI check for a bundled call");

  // Emit a KCFI check for the call instruction at MBBI. The implementation
  // must unfold memory operands if applicable.
  MachineInstr *Check = TLI->EmitKCFICheck(MBB, MBBI, TII);

  // Clear the original call's CFI type.
  assert(MBBI->isCall() && "Unexpected instruction type");
  MBBI->setCFIType(*MBB.getParent(), 0);

  // Move the trap out of line if supported and safe; otherwise bundle the
  // check and call to prevent further changes.
  if (!tryEmitOutOfLineTrap(MBB, MBBI, *Check) && !MBBI->isBundled())
    finalizeBundle(MBB, Check->getIterator(), std::next(MBBI->getIterator()));

  ++NumKCFIChecksAdded;
}

bool KCFI::run(MachineFunction &MF) {
  const Module *M = MF.getFunction().getParent();
  if (!M->getModuleFlag("kcfi"))
    return false;

  const auto &SubTarget = MF.getSubtarget();
  TII = SubTarget.getInstrInfo();
  TLI = SubTarget.getTargetLowering();

  const Function &F = MF.getFunction();
  if (hasNonZeroModuleFlag(*M, "kcfi-out-of-line-traps") &&
      !hasNonZeroModuleFlag(*M, "eh-asynch") && !MF.getWinEHFuncInfo() &&
      (!F.hasPersonalityFn() ||
       !isAsynchronousEHPersonality(
           classifyEHPersonality(F.getPersonalityFn()))) &&
      TLI->supportKCFIOutOfLineTraps(MF)) {
    // Moving a trap out of line splits the block of its call, so find the
    // calls first, and go through them last to first so that each split only
    // moves the instructions up to the previous one.
    SmallVector<MachineInstr *, 8> Calls;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB.instrs())
        if (MI.isCall() && MI.getCFIType())
          Calls.push_back(&MI);
    if (Calls.empty())
      return false;

    OutOfLineTraps = true;
    for (MachineInstr *Call : reverse(Calls))
      emitCheck(*Call->getParent(), Call->getIterator());
    return true;
  }

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // Use instr_iterator because we don't want to skip bundles.
    for (MachineBasicBlock::instr_iterator MII = MBB.instr_begin(),
                                           MIE = MBB.instr_end();
         MII != MIE; ++MII) {
      if (MII->isCall() && MII->getCFIType()) {
        emitCheck(MBB, MII);
        Changed = true;
      }
    }
  }

  return Changed;
}
