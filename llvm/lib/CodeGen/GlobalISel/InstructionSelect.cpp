//===- llvm/CodeGen/GlobalISel/InstructionSelect.cpp - InstructionSelect ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the InstructionSelect class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Config/config.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGenCoverage.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "instruction-select"

using namespace llvm;

DEBUG_COUNTER(GlobalISelCounter, "globalisel",
              "Controls whether to select function with GlobalISel");

#ifdef LLVM_GISEL_COV_PREFIX
static cl::opt<std::string>
    CoveragePrefix("gisel-coverage-prefix", cl::init(LLVM_GISEL_COV_PREFIX),
                   cl::desc("Record GlobalISel rule coverage files of this "
                            "prefix if instrumentation was generated"));
#else
static const std::string CoveragePrefix;
#endif

char InstructionSelect::ID = 0;
INITIALIZE_PASS_BEGIN(InstructionSelect, DEBUG_TYPE,
                      "Select target instructions out of generic instructions",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LazyBlockFrequencyInfoPass)
INITIALIZE_PASS_END(InstructionSelect, DEBUG_TYPE,
                    "Select target instructions out of generic instructions",
                    false, false)

InstructionSelect::InstructionSelect(CodeGenOptLevel OL, char &PassID)
    : MachineFunctionPass(PassID), OptLevel(OL) {}

/// This class observes instruction insertions/removals.
/// InstructionSelect stores an iterator of the instruction prior to the one
/// that is currently being selected to determine which instruction to select
/// next. Previously this meant that selecting multiple instructions at once was
/// illegal behavior due to potential invalidation of this iterator. This is
/// a non-obvious limitation for selector implementers. Therefore, to allow
/// deletion of arbitrary instructions, we detect this case and continue
/// selection with the predecessor of the deleted instruction.
class InstructionSelect::MIIteratorMaintainer : public GISelChangeObserver {
#ifndef NDEBUG
  SmallSetVector<const MachineInstr *, 32> CreatedInstrs;
#endif
public:
  MachineBasicBlock::reverse_iterator MII;

  void changingInstr(MachineInstr &MI) override {
    llvm_unreachable("InstructionSelect does not track changed instructions!");
  }
  void changedInstr(MachineInstr &MI) override {
    llvm_unreachable("InstructionSelect does not track changed instructions!");
  }

  void createdInstr(MachineInstr &MI) override {
    LLVM_DEBUG(dbgs() << "Creating:  " << MI; CreatedInstrs.insert(&MI));
  }

  void erasingInstr(MachineInstr &MI) override {
    LLVM_DEBUG(dbgs() << "Erasing:   " << MI; CreatedInstrs.remove(&MI));
    if (MII.getInstrIterator().getNodePtr() == &MI) {
      // If the iterator points to the MI that will be erased (i.e. the MI prior
      // to the MI that is currently being selected), the iterator would be
      // invalidated. Continue selection with its predecessor.
      ++MII;
      LLVM_DEBUG(dbgs() << "Instruction removal updated iterator.\n");
    }
  }

  void reportFullyCreatedInstrs() {
    LLVM_DEBUG({
      if (CreatedInstrs.empty()) {
        dbgs() << "Created no instructions.\n";
      } else {
        dbgs() << "Created:\n";
        for (const auto *MI : CreatedInstrs) {
          dbgs() << "  " << *MI;
        }
        CreatedInstrs.clear();
      }
    });
  }
};

void InstructionSelect::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.addRequired<GISelKnownBitsAnalysis>();
  AU.addPreserved<GISelKnownBitsAnalysis>();

  if (OptLevel != CodeGenOptLevel::None) {
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    LazyBlockFrequencyInfoPass::getLazyBFIAnalysisUsage(AU);
  }
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool InstructionSelect::runOnMachineFunction(MachineFunction &MF) {
  // If the ISel pipeline failed, do not bother running that pass.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;

  ISel = MF.getSubtarget().getInstructionSelector();
  ISel->TPC = &getAnalysis<TargetPassConfig>();

  // FIXME: Properly override OptLevel in TargetMachine. See OptLevelChanger
  CodeGenOptLevel OldOptLevel = OptLevel;
  auto RestoreOptLevel = make_scope_exit([=]() { OptLevel = OldOptLevel; });
  OptLevel = MF.getFunction().hasOptNone() ? CodeGenOptLevel::None
                                           : MF.getTarget().getOptLevel();

  KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);
  if (OptLevel != CodeGenOptLevel::None) {
    PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
    if (PSI && PSI->hasProfileSummary())
      BFI = &getAnalysis<LazyBlockFrequencyInfoPass>().getBFI();
  }

  return selectMachineFunction(MF);
}

bool InstructionSelect::selectMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "Selecting function: " << MF.getName() << '\n');
  assert(ISel && "Cannot work without InstructionSelector");

  const TargetPassConfig &TPC = *ISel->TPC;
  CodeGenCoverage CoverageInfo;
  ISel->setupMF(MF, KB, &CoverageInfo, PSI, BFI);

  // An optimization remark emitter. Used to report failures.
  MachineOptimizationRemarkEmitter MORE(MF, /*MBFI=*/nullptr);
  ISel->MORE = &MORE;

  // FIXME: There are many other MF/MFI fields we need to initialize.

  MachineRegisterInfo &MRI = MF.getRegInfo();
#ifndef NDEBUG
  // Check that our input is fully legal: we require the function to have the
  // Legalized property, so it should be.
  // FIXME: This should be in the MachineVerifier, as the RegBankSelected
  // property check already is.
  if (!DisableGISelLegalityCheck)
    if (const MachineInstr *MI = machineFunctionIsIllegal(MF)) {
      reportGISelFailure(MF, TPC, MORE, "gisel-select",
                         "instruction is not legal", *MI);
      return false;
    }
  // FIXME: We could introduce new blocks and will need to fix the outer loop.
  // Until then, keep track of the number of blocks to assert that we don't.
  const size_t NumBlocks = MF.size();
#endif
  // Keep track of selected blocks, so we can delete unreachable ones later.
  DenseSet<MachineBasicBlock *> SelectedBlocks;

  {
    // Observe IR insertions and removals during selection.
    // We only install a MachineFunction::Delegate instead of a
    // GISelChangeObserver, because we do not want notifications about changed
    // instructions. This prevents significant compile-time regressions from
    // e.g. constrainOperandRegClass().
    GISelObserverWrapper AllObservers;
    MIIteratorMaintainer MIIMaintainer;
    AllObservers.addObserver(&MIIMaintainer);
    RAIIDelegateInstaller DelInstaller(MF, &AllObservers);
    ISel->AllObservers = &AllObservers;

    for (MachineBasicBlock *MBB : post_order(&MF)) {
      ISel->CurMBB = MBB;
      SelectedBlocks.insert(MBB);

      // Select instructions in reverse block order.
      MIIMaintainer.MII = MBB->rbegin();
      for (auto End = MBB->rend(); MIIMaintainer.MII != End;) {
        MachineInstr &MI = *MIIMaintainer.MII;
        // Increment early to skip instructions inserted by select().
        ++MIIMaintainer.MII;

        LLVM_DEBUG(dbgs() << "\nSelect:  " << MI);
        if (!selectInstr(MI)) {
          LLVM_DEBUG(dbgs() << "Selection failed!\n";
                     MIIMaintainer.reportFullyCreatedInstrs());
          reportGISelFailure(MF, TPC, MORE, "gisel-select", "cannot select",
                             MI);
          return false;
        }
        LLVM_DEBUG(MIIMaintainer.reportFullyCreatedInstrs());
      }
    }
  }

  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty())
      continue;

    if (!SelectedBlocks.contains(&MBB)) {
      // This is an unreachable block and therefore hasn't been selected, since
      // the main selection loop above uses a postorder block traversal.
      // We delete all the instructions in this block since it's unreachable.
      MBB.clear();
      // Don't delete the block in case the block has it's address taken or is
      // still being referenced by a phi somewhere.
      continue;
    }
    // Try to find redundant copies b/w vregs of the same register class.
    for (auto MII = MBB.rbegin(), End = MBB.rend(); MII != End;) {
      MachineInstr &MI = *MII;
      ++MII;

      if (MI.getOpcode() != TargetOpcode::COPY)
        continue;
      Register SrcReg = MI.getOperand(1).getReg();
      Register DstReg = MI.getOperand(0).getReg();
      if (SrcReg.isVirtual() && DstReg.isVirtual()) {
        auto SrcRC = MRI.getRegClass(SrcReg);
        auto DstRC = MRI.getRegClass(DstReg);
        if (SrcRC == DstRC) {
          MRI.replaceRegWith(DstReg, SrcReg);
          MI.eraseFromParent();
        }
      }
    }
  }

#ifndef NDEBUG
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  // Now that selection is complete, there are no more generic vregs.  Verify
  // that the size of the now-constrained vreg is unchanged and that it has a
  // register class.
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register VReg = Register::index2VirtReg(I);

    MachineInstr *MI = nullptr;
    if (!MRI.def_empty(VReg))
      MI = &*MRI.def_instr_begin(VReg);
    else if (!MRI.use_empty(VReg)) {
      MI = &*MRI.use_instr_begin(VReg);
      // Debug value instruction is permitted to use undefined vregs.
      if (MI->isDebugValue())
        continue;
    }
    if (!MI)
      continue;

    const TargetRegisterClass *RC = MRI.getRegClassOrNull(VReg);
    if (!RC) {
      reportGISelFailure(MF, TPC, MORE, "gisel-select",
                         "VReg has no regclass after selection", *MI);
      return false;
    }

    const LLT Ty = MRI.getType(VReg);
    if (Ty.isValid() &&
        TypeSize::isKnownGT(Ty.getSizeInBits(), TRI.getRegSizeInBits(*RC))) {
      reportGISelFailure(
          MF, TPC, MORE, "gisel-select",
          "VReg's low-level type and register class have different sizes", *MI);
      return false;
    }
  }

  if (MF.size() != NumBlocks) {
    MachineOptimizationRemarkMissed R("gisel-select", "GISelFailure",
                                      MF.getFunction().getSubprogram(),
                                      /*MBB=*/nullptr);
    R << "inserting blocks is not supported yet";
    reportGISelFailure(MF, TPC, MORE, R);
    return false;
  }
#endif

  if (!DebugCounter::shouldExecute(GlobalISelCounter)) {
    dbgs() << "Falling back for function " << MF.getName() << "\n";
    MF.getProperties().set(MachineFunctionProperties::Property::FailedISel);
    return false;
  }

  // Determine if there are any calls in this machine function. Ported from
  // SelectionDAG.
  MachineFrameInfo &MFI = MF.getFrameInfo();
  for (const auto &MBB : MF) {
    if (MFI.hasCalls() && MF.hasInlineAsm())
      break;

    for (const auto &MI : MBB) {
      if ((MI.isCall() && !MI.isReturn()) || MI.isStackAligningInlineAsm())
        MFI.setHasCalls(true);
      if (MI.isInlineAsm())
        MF.setHasInlineAsm(true);
    }
  }

  // FIXME: FinalizeISel pass calls finalizeLowering, so it's called twice.
  auto &TLI = *MF.getSubtarget().getTargetLowering();
  TLI.finalizeLowering(MF);

  LLVM_DEBUG({
    dbgs() << "Rules covered by selecting function: " << MF.getName() << ":";
    for (auto RuleID : CoverageInfo.covered())
      dbgs() << " id" << RuleID;
    dbgs() << "\n\n";
  });
  CoverageInfo.emit(CoveragePrefix,
                    TLI.getTargetMachine().getTarget().getBackendName());

  // If we successfully selected the function nothing is going to use the vreg
  // types after us (otherwise MIRPrinter would need them). Make sure the types
  // disappear.
  MRI.clearVirtRegTypes();

  // FIXME: Should we accurately track changes?
  return true;
}

bool InstructionSelect::selectInstr(MachineInstr &MI) {
  MachineRegisterInfo &MRI = ISel->MF->getRegInfo();

  // We could have folded this instruction away already, making it dead.
  // If so, erase it.
  if (isTriviallyDead(MI, MRI)) {
    LLVM_DEBUG(dbgs() << "Is dead.\n");
    salvageDebugInfo(MRI, MI);
    MI.eraseFromParent();
    return true;
  }

  // Eliminate hints or G_CONSTANT_FOLD_BARRIER.
  if (isPreISelGenericOptimizationHint(MI.getOpcode()) ||
      MI.getOpcode() == TargetOpcode::G_CONSTANT_FOLD_BARRIER) {
    auto [DstReg, SrcReg] = MI.getFirst2Regs();

    // At this point, the destination register class of the op may have
    // been decided.
    //
    // Propagate that through to the source register.
    const TargetRegisterClass *DstRC = MRI.getRegClassOrNull(DstReg);
    if (DstRC)
      MRI.setRegClass(SrcReg, DstRC);
    assert(canReplaceReg(DstReg, SrcReg, MRI) &&
           "Must be able to replace dst with src!");
    MI.eraseFromParent();
    MRI.replaceRegWith(DstReg, SrcReg);
    return true;
  }

  if (MI.getOpcode() == TargetOpcode::G_INVOKE_REGION_START) {
    MI.eraseFromParent();
    return true;
  }

  return ISel->select(MI);
}
