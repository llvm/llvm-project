//===---- KCFI.cpp - Implements Kernel Control-Flow Integrity (KCFI) ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements two MachineFunction passes for Kernel Control-Flow
// Integrity (KCFI) indirect call check lowering:
//
// 1. KCFI ("kcfi", scheduled in addPreSched2): emits an arch-specific check
//    before each call instruction with a cfi-type attribute and bundles the
//    check and the call to prevent unintentional modifications.
//
// 2. KCFISplit ("kcfi-split", scheduled in addPostBBSections): when
//    out-of-line traps are enabled and supported, unbundles eligible KCFI
//    check/call bundles after scheduling, block placement, outlining, and
//    basic block sections / function splitting, moving each failure trap into a
//    cold block at the tail of the function (and into the cold/basic-block
//    section layout when section splitting is enabled) so passing checks fall
//    through.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/KCFI.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "kcfi"
#define KCFI_PASS_NAME "Insert KCFI indirect call checks"
#define KCFI_SPLIT_PASS_NAME "Split KCFI indirect call check traps out of line"

STATISTIC(NumKCFIChecksAdded, "Number of indirect call checks added");

static bool hasNonZeroModuleFlag(const Module &M, StringRef Name) {
  auto *Val = mdconst::extract_or_null<ConstantInt>(M.getModuleFlag(Name));
  return Val && !Val->isZero();
}

namespace {
class KCFI {
public:
  bool run(MachineFunction &MF);

private:
  /// Machine instruction info used throughout the class.
  const TargetInstrInfo *TII = nullptr;

  /// Target lowering for arch-specific parts.
  const TargetLowering *TLI = nullptr;

  /// Emits a KCFI check before an indirect call.
  /// \returns true if the check was added and false otherwise.
  bool emitCheck(MachineBasicBlock &MBB,
                 MachineBasicBlock::instr_iterator I) const;
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

class KCFISplit {
public:
  bool run(MachineFunction &MF);

private:
  const TargetInstrInfo *TII = nullptr;
  const TargetLowering *TLI = nullptr;

  /// Collects the EH begin label and any OpGnuArgsSize CFI instructions that
  /// must move with \p Bundle when splitting an invoke at the end of \p MBB.
  /// \returns false if \p Bundle is not a splittable trailing invoke.
  bool collectInvokePrefix(MachineBasicBlock &MBB, MachineInstr &Bundle,
                           SmallVectorImpl<MachineInstr *> &Prefix) const;

  /// Unbundles \p Bundle, splits \p MBB across the failure trap branch while
  /// keeping the split continuation in \p MBB's section, and inserts the new
  /// trap block before \p TrapInsertPt, updating \p TrapInsertPt to point to
  /// the newly inserted block.
  bool splitBundle(MachineBasicBlock &MBB, MachineInstr &Bundle,
                   MachineFunction::iterator &TrapInsertPt) const;
};

class MachineKCFISplitLegacy : public MachineFunctionPass {
public:
  static char ID;

  MachineKCFISplitLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return KCFI_SPLIT_PASS_NAME; }
  bool runOnMachineFunction(MachineFunction &MF) override {
    return KCFISplit().run(MF);
  }
};

char MachineKCFISplitLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS(MachineKCFILegacy, DEBUG_TYPE, KCFI_PASS_NAME, false, false)
INITIALIZE_PASS(MachineKCFISplitLegacy, "kcfi-split", KCFI_SPLIT_PASS_NAME,
                false, false)

FunctionPass *llvm::createKCFIPass() { return new MachineKCFILegacy(); }
FunctionPass *llvm::createKCFISplitPass() {
  return new MachineKCFISplitLegacy();
}

PreservedAnalyses MachineKCFIPass::run(MachineFunction &MF,
                                       MachineFunctionAnalysisManager &MFAM) {
  if (!KCFI().run(MF))
    return PreservedAnalyses::all();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

PreservedAnalyses
MachineKCFISplitPass::run(MachineFunction &MF,
                          MachineFunctionAnalysisManager &MFAM) {
  if (!KCFISplit().run(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses();
}

bool KCFI::emitCheck(MachineBasicBlock &MBB,
                     MachineBasicBlock::instr_iterator MBBI) const {
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

  // If not already bundled, bundle the check and the call to prevent
  // further changes.
  if (!MBBI->isBundled())
    finalizeBundle(MBB, Check->getIterator(), std::next(MBBI->getIterator()));

  ++NumKCFIChecksAdded;
  return true;
}

bool KCFI::run(MachineFunction &MF) {
  const Module *M = MF.getFunction().getParent();
  if (!M->getModuleFlag("kcfi"))
    return false;

  const auto &SubTarget = MF.getSubtarget();
  TII = SubTarget.getInstrInfo();
  TLI = SubTarget.getTargetLowering();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // Use instr_iterator because we don't want to skip bundles.
    for (MachineBasicBlock::instr_iterator MII = MBB.instr_begin(),
                                           MIE = MBB.instr_end();
         MII != MIE; ++MII) {
      if (MII->isCall() && MII->getCFIType())
        Changed |= emitCheck(MBB, MII);
    }
  }

  return Changed;
}

bool KCFISplit::collectInvokePrefix(
    MachineBasicBlock &MBB, MachineInstr &Bundle,
    SmallVectorImpl<MachineInstr *> &Prefix) const {
  // Only split the trailing invoke in an EH block: require no following
  // call/bundle and a single end EH_LABEL after Bundle, plus a single
  // begin EH_LABEL before Bundle with no intervening call/bundle and no
  // earlier EH_LABEL in MBB.
  unsigned EndLabels = 0;
  for (auto It = std::next(MachineBasicBlock::iterator(&Bundle));
       It != MBB.end(); ++It) {
    if (It->isCall() || It->isBundle())
      return false;
    if (It->isEHLabel() && ++EndLabels > 1)
      return false;
  }
  if (EndLabels != 1)
    return false;

  const MachineFunction &MF = *MBB.getParent();
  MachineInstr *BeginLabel = nullptr;
  for (auto It = MachineBasicBlock::iterator(&Bundle); It != MBB.begin();) {
    --It;
    if (BeginLabel) {
      if (It->isEHLabel())
        return false;
      continue;
    }
    if (It->isCall() || It->isBundle())
      return false;
    if (It->isEHLabel()) {
      BeginLabel = &*It;
      Prefix.push_back(BeginLabel);
    } else if (It->isCFIInstruction() &&
               MF.getFrameInstructions()[It->getOperand(0).getCFIIndex()]
                       .getOperation() == MCCFIInstruction::OpGnuArgsSize) {
      Prefix.push_back(&*It);
    }
  }
  return BeginLabel != nullptr;
}

bool KCFISplit::splitBundle(MachineBasicBlock &MBB, MachineInstr &Bundle,
                            MachineFunction::iterator &TrapInsertPt) const {
  SmallVector<MachineInstr *, 2> MoveBeforeCall;
  if (MBB.hasEHPadSuccessor() &&
      !collectInvokePrefix(MBB, Bundle, MoveBeforeCall))
    return false;

  MachineBasicBlock::instr_iterator CheckIt = std::next(Bundle.getIterator());
  MachineBasicBlock::instr_iterator CallIt = std::next(CheckIt);
  MachineInstr *Check = &*CheckIt;
  MachineInstr *Call = &*CallIt;
  Check->unbundleFromPred();
  Call->unbundleFromPred();
  for (MachineOperand &MO : Call->operands())
    if (MO.isReg() && MO.isInternalRead())
      MO.setIsInternalRead(false);
  Bundle.eraseFromParent();

  MachineFunction &MF = *MBB.getParent();
  MachineBasicBlock *TrapMBB = MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MF.insert(TrapInsertPt, TrapMBB);
  TrapInsertPt = TrapMBB->getIterator();
  MachineInstr *Branch = TLI->EmitKCFIOutOfLineTrap(*Check, *TrapMBB, TII);

  for (MachineInstr *MI : reverse(MoveBeforeCall))
    MBB.splice(Call->getIterator(), &MBB, MI->getIterator());

  MachineBasicBlock *SplitBB = MBB.splitAt(*Branch, /*UpdateLiveIns=*/true);
  assert(SplitBB != &MBB && "Expected a split block before the call");
  SplitBB->setSectionID(MBB.getSectionID());

  MBB.setSuccProbability(MBB.succ_begin(), BranchProbability::getOne());
  MBB.addSuccessor(TrapMBB, BranchProbability::getZero());
  return true;
}

bool KCFISplit::run(MachineFunction &MF) {
  const Module *M = MF.getFunction().getParent();
  if (!M->getModuleFlag("kcfi") ||
      !hasNonZeroModuleFlag(*M, "kcfi-out-of-line-traps") ||
      hasNonZeroModuleFlag(*M, "eh-asynch") || MF.getWinEHFuncInfo())
    return false;

  const Function &F = MF.getFunction();
  if (F.hasPersonalityFn() &&
      isAsynchronousEHPersonality(classifyEHPersonality(F.getPersonalityFn())))
    return false;

  const auto &SubTarget = MF.getSubtarget();
  TII = SubTarget.getInstrInfo();
  TLI = SubTarget.getTargetLowering();
  if (!TLI->supportKCFIOutOfLineTraps(MF))
    return false;

  SmallVector<MachineBasicBlock *, 16> Blocks;
  for (MachineBasicBlock &MBB : MF)
    Blocks.push_back(&MBB);

  MachineFunction::iterator TrapInsertPt = MF.end();
  if (MF.hasBBSections() &&
      MF.getTarget().getBBSectionsType() == BasicBlockSection::All) {
    while (TrapInsertPt != MF.begin() &&
           std::prev(TrapInsertPt)->getSectionID() ==
               MBBSectionID::ExceptionSectionID)
      --TrapInsertPt;
  }
  MachineFunction::iterator TrapEnd = TrapInsertPt;

  bool Changed = false;
  for (MachineBasicBlock *MBB : reverse(Blocks)) {
    if (MBB->isEHPad())
      continue;

    SmallVector<MachineInstr *, 4> Bundles;
    for (MachineInstr &MI : *MBB) {
      if (!MI.isBundle())
        continue;
      auto CheckIt = std::next(MI.getIterator());
      if (CheckIt == MBB->instr_end() || !CheckIt->isBundledWithPred() ||
          !TLI->isKCFICheck(*CheckIt))
        continue;
      auto CallIt = std::next(CheckIt);
      if (CallIt == MBB->instr_end() || !CallIt->isBundledWithPred() ||
          !CallIt->isCall())
        continue;
      auto NextIt = std::next(CallIt);
      if (NextIt != MBB->instr_end() && NextIt->isBundledWithPred())
        continue;
      Bundles.push_back(&MI);
    }

    for (MachineInstr *Bundle : reverse(Bundles))
      Changed |= splitBundle(*MBB, *Bundle, TrapInsertPt);
  }

  if (!Changed)
    return false;

  if (MF.hasBBSections()) {
    BasicBlockSection BBSectionsType = MF.getTarget().getBBSectionsType();
    if (BBSectionsType == BasicBlockSection::All) {
      unsigned MaxSectionNum = 0;
      for (const MachineBasicBlock &MBB : make_range(MF.begin(), TrapInsertPt))
        if (MBB.getSectionID().Type == MBBSectionID::Default)
          MaxSectionNum = std::max(MaxSectionNum, MBB.getSectionID().Number);
      for (MachineBasicBlock &MBB : make_range(TrapInsertPt, TrapEnd))
        MBB.setSectionID(++MaxSectionNum);
    } else if (BBSectionsType != BasicBlockSection::List ||
               MF.getTarget().Options.FunctionSplitting !=
                   FunctionSplittingMode::None) {
      for (MachineBasicBlock &MBB : make_range(TrapInsertPt, TrapEnd))
        MBB.setSectionID(MBBSectionID::ColdSectionID);
    }
    for (MachineBasicBlock &MBB : MF) {
      MBB.setIsBeginSection(false);
      MBB.setIsEndSection(false);
    }
    MF.assignBeginEndSections();
  }

  if (MF.getTarget().Options.BBAddrMap)
    MF.RenumberBlocks();

  return true;
}
