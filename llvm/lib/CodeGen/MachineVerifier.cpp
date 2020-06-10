//===- MachineVerifier.cpp - Machine Code Verifier ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to verify generated machine code. The following is checked:
//
// Operand counts: All explicit operands must be present.
//
// Register classes: All physical and virtual register operands must be
// compatible with the register class required by the instruction descriptor.
//
// Register live intervals: Registers must be defined only once, and must be
// defined before use.
//
// The machine code verifier is enabled with the command-line option
// -verify-machineinstrs.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/CodeGen/GlobalISel/RegisterBank.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalCalc.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LowLevelTypeImpl.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>

using namespace llvm;

namespace {

  struct MachineVerifier {
    MachineVerifier(Pass *pass, const char *b) : PASS(pass), Banner(b) {}

    unsigned verify(MachineFunction &MF);

    Pass *const PASS;
    const char *Banner;
    const MachineFunction *MF;
    const TargetMachine *TM;
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    const MachineRegisterInfo *MRI;

    unsigned foundErrors;

    // Avoid querying the MachineFunctionProperties for each operand.
    bool isFunctionRegBankSelected;
    bool isFunctionSelected;

    using RegVector = SmallVector<unsigned, 16>;
    using RegMaskVector = SmallVector<const uint32_t *, 4>;
    using RegSet = DenseSet<unsigned>;
    using RegMap = DenseMap<unsigned, const MachineInstr *>;
    using BlockSet = SmallPtrSet<const MachineBasicBlock *, 8>;

    const MachineInstr *FirstNonPHI;
    const MachineInstr *FirstTerminator;
    BlockSet FunctionBlocks;

    BitVector regsReserved;
    RegSet regsLive;
    RegVector regsDefined, regsDead, regsKilled;
    RegMaskVector regMasks;

    SlotIndex lastIndex;

    // Add Reg and any sub-registers to RV
    void addRegWithSubRegs(RegVector &RV, unsigned Reg) {
      RV.push_back(Reg);
      if (Register::isPhysicalRegister(Reg))
        for (const MCPhysReg &SubReg : TRI->subregs(Reg))
          RV.push_back(SubReg);
    }

    struct BBInfo {
      // Is this MBB reachable from the MF entry point?
      bool reachable = false;

      // Vregs that must be live in because they are used without being
      // defined. Map value is the user.
      RegMap vregsLiveIn;

      // Regs killed in MBB. They may be defined again, and will then be in both
      // regsKilled and regsLiveOut.
      RegSet regsKilled;

      // Regs defined in MBB and live out. Note that vregs passing through may
      // be live out without being mentioned here.
      RegSet regsLiveOut;

      // Vregs that pass through MBB untouched. This set is disjoint from
      // regsKilled and regsLiveOut.
      RegSet vregsPassed;

      // Vregs that must pass through MBB because they are needed by a successor
      // block. This set is disjoint from regsLiveOut.
      RegSet vregsRequired;

      // Set versions of block's predecessor and successor lists.
      BlockSet Preds, Succs;

      BBInfo() = default;

      // Add register to vregsRequired if it belongs there. Return true if
      // anything changed.
      bool addRequired(unsigned Reg) {
        if (!Register::isVirtualRegister(Reg))
          return false;
        if (regsLiveOut.count(Reg))
          return false;
        return vregsRequired.insert(Reg).second;
      }

      // Same for a full set.
      bool addRequired(const RegSet &RS) {
        bool Changed = false;
        for (unsigned Reg : RS)
          Changed |= addRequired(Reg);
        return Changed;
      }

      // Same for a full map.
      bool addRequired(const RegMap &RM) {
        bool Changed = false;
        for (const auto &I : RM)
          Changed |= addRequired(I.first);
        return Changed;
      }

      // Live-out registers are either in regsLiveOut or vregsPassed.
      bool isLiveOut(unsigned Reg) const {
        return regsLiveOut.count(Reg) || vregsPassed.count(Reg);
      }
    };

    // Extra register info per MBB.
    DenseMap<const MachineBasicBlock*, BBInfo> MBBInfoMap;

    bool isReserved(unsigned Reg) {
      return Reg < regsReserved.size() && regsReserved.test(Reg);
    }

    bool isAllocatable(unsigned Reg) const {
      return Reg < TRI->getNumRegs() && TRI->isInAllocatableClass(Reg) &&
             !regsReserved.test(Reg);
    }

    // Analysis information if available
    LiveVariables *LiveVars;
    LiveIntervals *LiveInts;
    LiveStacks *LiveStks;
    SlotIndexes *Indexes;

    void visitMachineFunctionBefore();
    void visitMachineBasicBlockBefore(const MachineBasicBlock *MBB);
    void visitMachineBundleBefore(const MachineInstr *MI);

    bool verifyVectorElementMatch(LLT Ty0, LLT Ty1, const MachineInstr *MI);
    void verifyPreISelGenericInstruction(const MachineInstr *MI);
    void visitMachineInstrBefore(const MachineInstr *MI);
    void visitMachineOperand(const MachineOperand *MO, unsigned MONum);
    void visitMachineBundleAfter(const MachineInstr *MI);
    void visitMachineBasicBlockAfter(const MachineBasicBlock *MBB);
    void visitMachineFunctionAfter();

    void report(const char *msg, const MachineFunction *MF);
    void report(const char *msg, const MachineBasicBlock *MBB);
    void report(const char *msg, const MachineInstr *MI);
    void report(const char *msg, const MachineOperand *MO, unsigned MONum,
                LLT MOVRegType = LLT{});

    void report_context(const LiveInterval &LI) const;
    void report_context(const LiveRange &LR, unsigned VRegUnit,
                        LaneBitmask LaneMask) const;
    void report_context(const LiveRange::Segment &S) const;
    void report_context(const VNInfo &VNI) const;
    void report_context(SlotIndex Pos) const;
    void report_context(MCPhysReg PhysReg) const;
    void report_context_liverange(const LiveRange &LR) const;
    void report_context_lanemask(LaneBitmask LaneMask) const;
    void report_context_vreg(unsigned VReg) const;
    void report_context_vreg_regunit(unsigned VRegOrUnit) const;

    void verifyInlineAsm(const MachineInstr *MI);

    void checkLiveness(const MachineOperand *MO, unsigned MONum);
    void checkLivenessAtUse(const MachineOperand *MO, unsigned MONum,
                            SlotIndex UseIdx, const LiveRange &LR, unsigned VRegOrUnit,
                            LaneBitmask LaneMask = LaneBitmask::getNone());
    void checkLivenessAtDef(const MachineOperand *MO, unsigned MONum,
                            SlotIndex DefIdx, const LiveRange &LR, unsigned VRegOrUnit,
                            bool SubRangeCheck = false,
                            LaneBitmask LaneMask = LaneBitmask::getNone());

    void markReachable(const MachineBasicBlock *MBB);
    void calcRegsPassed();
    void checkPHIOps(const MachineBasicBlock &MBB);

    void calcRegsRequired();
    void verifyLiveVariables();
    void verifyLiveIntervals();
    void verifyLiveInterval(const LiveInterval&);
    void verifyLiveRangeValue(const LiveRange&, const VNInfo*, unsigned,
                              LaneBitmask);
    void verifyLiveRangeSegment(const LiveRange&,
                                const LiveRange::const_iterator I, unsigned,
                                LaneBitmask);
    void verifyLiveRange(const LiveRange&, unsigned,
                         LaneBitmask LaneMask = LaneBitmask::getNone());

    void verifyStackFrame();

    void verifySlotIndexes() const;
    void verifyProperties(const MachineFunction &MF);
  };

  struct MachineVerifierPass : public MachineFunctionPass {
    static char ID; // Pass ID, replacement for typeid

    const std::string Banner;

    MachineVerifierPass(std::string banner = std::string())
      : MachineFunctionPass(ID), Banner(std::move(banner)) {
        initializeMachineVerifierPassPass(*PassRegistry::getPassRegistry());
      }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
      unsigned FoundErrors = MachineVerifier(this, Banner.c_str()).verify(MF);
      if (FoundErrors)
        report_fatal_error("Found "+Twine(FoundErrors)+" machine code errors.");
      return false;
    }
  };

} // end anonymous namespace

char MachineVerifierPass::ID = 0;

INITIALIZE_PASS(MachineVerifierPass, "machineverifier",
                "Verify generated machine code", false, false)

FunctionPass *llvm::createMachineVerifierPass(const std::string &Banner) {
  return new MachineVerifierPass(Banner);
}

bool MachineFunction::verify(Pass *p, const char *Banner, bool AbortOnErrors)
    const {
  MachineFunction &MF = const_cast<MachineFunction&>(*this);
  unsigned FoundErrors = MachineVerifier(p, Banner).verify(MF);
  if (AbortOnErrors && FoundErrors)
    report_fatal_error("Found "+Twine(FoundErrors)+" machine code errors.");
  return FoundErrors == 0;
}

void MachineVerifier::verifySlotIndexes() const {
  if (Indexes == nullptr)
    return;

  // Ensure the IdxMBB list is sorted by slot indexes.
  SlotIndex Last;
  for (SlotIndexes::MBBIndexIterator I = Indexes->MBBIndexBegin(),
       E = Indexes->MBBIndexEnd(); I != E; ++I) {
    assert(!Last.isValid() || I->first > Last);
    Last = I->first;
  }
}

void MachineVerifier::verifyProperties(const MachineFunction &MF) {
  // If a pass has introduced virtual registers without clearing the
  // NoVRegs property (or set it without allocating the vregs)
  // then report an error.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::NoVRegs) &&
      MRI->getNumVirtRegs())
    report("Function has NoVRegs property but there are VReg operands", &MF);
}

unsigned MachineVerifier::verify(MachineFunction &MF) {
  foundErrors = 0;

  this->MF = &MF;
  TM = &MF.getTarget();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  const bool isFunctionFailedISel = MF.getProperties().hasProperty(
      MachineFunctionProperties::Property::FailedISel);

  // If we're mid-GlobalISel and we already triggered the fallback path then
  // it's expected that the MIR is somewhat broken but that's ok since we'll
  // reset it and clear the FailedISel attribute in ResetMachineFunctions.
  if (isFunctionFailedISel)
    return foundErrors;

  isFunctionRegBankSelected = MF.getProperties().hasProperty(
      MachineFunctionProperties::Property::RegBankSelected);
  isFunctionSelected = MF.getProperties().hasProperty(
      MachineFunctionProperties::Property::Selected);

  LiveVars = nullptr;
  LiveInts = nullptr;
  LiveStks = nullptr;
  Indexes = nullptr;
  if (PASS) {
    LiveInts = PASS->getAnalysisIfAvailable<LiveIntervals>();
    // We don't want to verify LiveVariables if LiveIntervals is available.
    if (!LiveInts)
      LiveVars = PASS->getAnalysisIfAvailable<LiveVariables>();
    LiveStks = PASS->getAnalysisIfAvailable<LiveStacks>();
    Indexes = PASS->getAnalysisIfAvailable<SlotIndexes>();
  }

  verifySlotIndexes();

  verifyProperties(MF);

  visitMachineFunctionBefore();
  for (const MachineBasicBlock &MBB : MF) {
    visitMachineBasicBlockBefore(&MBB);
    // Keep track of the current bundle header.
    const MachineInstr *CurBundle = nullptr;
    // Do we expect the next instruction to be part of the same bundle?
    bool InBundle = false;

    for (const MachineInstr &MI : MBB.instrs()) {
      if (MI.getParent() != &MBB) {
        report("Bad instruction parent pointer", &MBB);
        errs() << "Instruction: " << MI;
        continue;
      }

      // Check for consistent bundle flags.
      if (InBundle && !MI.isBundledWithPred())
        report("Missing BundledPred flag, "
               "BundledSucc was set on predecessor",
               &MI);
      if (!InBundle && MI.isBundledWithPred())
        report("BundledPred flag is set, "
               "but BundledSucc not set on predecessor",
               &MI);

      // Is this a bundle header?
      if (!MI.isInsideBundle()) {
        if (CurBundle)
          visitMachineBundleAfter(CurBundle);
        CurBundle = &MI;
        visitMachineBundleBefore(CurBundle);
      } else if (!CurBundle)
        report("No bundle header", &MI);
      visitMachineInstrBefore(&MI);
      for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
        const MachineOperand &Op = MI.getOperand(I);
        if (Op.getParent() != &MI) {
          // Make sure to use correct addOperand / RemoveOperand / ChangeTo
          // functions when replacing operands of a MachineInstr.
          report("Instruction has operand with wrong parent set", &MI);
        }

        visitMachineOperand(&Op, I);
      }

      // Was this the last bundled instruction?
      InBundle = MI.isBundledWithSucc();
    }
    if (CurBundle)
      visitMachineBundleAfter(CurBundle);
    if (InBundle)
      report("BundledSucc flag set on last instruction in block", &MBB.back());
    visitMachineBasicBlockAfter(&MBB);
  }
  visitMachineFunctionAfter();

  // Clean up.
  regsLive.clear();
  regsDefined.clear();
  regsDead.clear();
  regsKilled.clear();
  regMasks.clear();
  MBBInfoMap.clear();

  return foundErrors;
}

void MachineVerifier::report(const char *msg, const MachineFunction *MF) {
  assert(MF);
  errs() << '\n';
  if (!foundErrors++) {
    if (Banner)
      errs() << "# " << Banner << '\n';
    if (LiveInts != nullptr)
      LiveInts->print(errs());
    else
      MF->print(errs(), Indexes);
  }
  errs() << "*** Bad machine code: " << msg << " ***\n"
      << "- function:    " << MF->getName() << "\n";
}

void MachineVerifier::report(const char *msg, const MachineBasicBlock *MBB) {
  assert(MBB);
  report(msg, MBB->getParent());
  errs() << "- basic block: " << printMBBReference(*MBB) << ' '
         << MBB->getName() << " (" << (const void *)MBB << ')';
  if (Indexes)
    errs() << " [" << Indexes->getMBBStartIdx(MBB)
        << ';' <<  Indexes->getMBBEndIdx(MBB) << ')';
  errs() << '\n';
}

void MachineVerifier::report(const char *msg, const MachineInstr *MI) {
  assert(MI);
  report(msg, MI->getParent());
  errs() << "- instruction: ";
  if (Indexes && Indexes->hasIndex(*MI))
    errs() << Indexes->getInstructionIndex(*MI) << '\t';
  MI->print(errs(), /*SkipOpers=*/true);
}

void MachineVerifier::report(const char *msg, const MachineOperand *MO,
                             unsigned MONum, LLT MOVRegType) {
  assert(MO);
  report(msg, MO->getParent());
  errs() << "- operand " << MONum << ":   ";
  MO->print(errs(), MOVRegType, TRI);
  errs() << "\n";
}

void MachineVerifier::report_context(SlotIndex Pos) const {
  errs() << "- at:          " << Pos << '\n';
}

void MachineVerifier::report_context(const LiveInterval &LI) const {
  errs() << "- interval:    " << LI << '\n';
}

void MachineVerifier::report_context(const LiveRange &LR, unsigned VRegUnit,
                                     LaneBitmask LaneMask) const {
  report_context_liverange(LR);
  report_context_vreg_regunit(VRegUnit);
  if (LaneMask.any())
    report_context_lanemask(LaneMask);
}

void MachineVerifier::report_context(const LiveRange::Segment &S) const {
  errs() << "- segment:     " << S << '\n';
}

void MachineVerifier::report_context(const VNInfo &VNI) const {
  errs() << "- ValNo:       " << VNI.id << " (def " << VNI.def << ")\n";
}

void MachineVerifier::report_context_liverange(const LiveRange &LR) const {
  errs() << "- liverange:   " << LR << '\n';
}

void MachineVerifier::report_context(MCPhysReg PReg) const {
  errs() << "- p. register: " << printReg(PReg, TRI) << '\n';
}

void MachineVerifier::report_context_vreg(unsigned VReg) const {
  errs() << "- v. register: " << printReg(VReg, TRI) << '\n';
}

void MachineVerifier::report_context_vreg_regunit(unsigned VRegOrUnit) const {
  if (Register::isVirtualRegister(VRegOrUnit)) {
    report_context_vreg(VRegOrUnit);
  } else {
    errs() << "- regunit:     " << printRegUnit(VRegOrUnit, TRI) << '\n';
  }
}

void MachineVerifier::report_context_lanemask(LaneBitmask LaneMask) const {
  errs() << "- lanemask:    " << PrintLaneMask(LaneMask) << '\n';
}

void MachineVerifier::markReachable(const MachineBasicBlock *MBB) {
  BBInfo &MInfo = MBBInfoMap[MBB];
  if (!MInfo.reachable) {
    MInfo.reachable = true;
    for (const MachineBasicBlock *Succ : MBB->successors())
      markReachable(Succ);
  }
}

void MachineVerifier::visitMachineFunctionBefore() {
  lastIndex = SlotIndex();
  regsReserved = MRI->reservedRegsFrozen() ? MRI->getReservedRegs()
                                           : TRI->getReservedRegs(*MF);

  if (!MF->empty())
    markReachable(&MF->front());

  // Build a set of the basic blocks in the function.
  FunctionBlocks.clear();
  for (const auto &MBB : *MF) {
    FunctionBlocks.insert(&MBB);
    BBInfo &MInfo = MBBInfoMap[&MBB];

    MInfo.Preds.insert(MBB.pred_begin(), MBB.pred_end());
    if (MInfo.Preds.size() != MBB.pred_size())
      report("MBB has duplicate entries in its predecessor list.", &MBB);

    MInfo.Succs.insert(MBB.succ_begin(), MBB.succ_end());
    if (MInfo.Succs.size() != MBB.succ_size())
      report("MBB has duplicate entries in its successor list.", &MBB);
  }

  // Check that the register use lists are sane.
  MRI->verifyUseLists();

  if (!MF->empty())
    verifyStackFrame();
}

void
MachineVerifier::visitMachineBasicBlockBefore(const MachineBasicBlock *MBB) {
  FirstTerminator = nullptr;
  FirstNonPHI = nullptr;

  if (!MF->getProperties().hasProperty(
      MachineFunctionProperties::Property::NoPHIs) && MRI->tracksLiveness()) {
    // If this block has allocatable physical registers live-in, check that
    // it is an entry block or landing pad.
    for (const auto &LI : MBB->liveins()) {
      if (isAllocatable(LI.PhysReg) && !MBB->isEHPad() &&
          !MBB->isInlineAsmBrDefaultTarget() &&
          MBB->getIterator() != MBB->getParent()->begin()) {
        report("MBB has allocatable live-in, but isn't entry or landing-pad.", MBB);
        report_context(LI.PhysReg);
      }
    }
  }

  // Count the number of landing pad successors.
  SmallPtrSet<const MachineBasicBlock*, 4> LandingPadSuccs;
  for (const auto *succ : MBB->successors()) {
    if (succ->isEHPad())
      LandingPadSuccs.insert(succ);
    if (!FunctionBlocks.count(succ))
      report("MBB has successor that isn't part of the function.", MBB);
    if (!MBBInfoMap[succ].Preds.count(MBB)) {
      report("Inconsistent CFG", MBB);
      errs() << "MBB is not in the predecessor list of the successor "
             << printMBBReference(*succ) << ".\n";
    }
  }

  // Check the predecessor list.
  for (const MachineBasicBlock *Pred : MBB->predecessors()) {
    if (!FunctionBlocks.count(Pred))
      report("MBB has predecessor that isn't part of the function.", MBB);
    if (!MBBInfoMap[Pred].Succs.count(MBB)) {
      report("Inconsistent CFG", MBB);
      errs() << "MBB is not in the successor list of the predecessor "
             << printMBBReference(*Pred) << ".\n";
    }
  }

  const MCAsmInfo *AsmInfo = TM->getMCAsmInfo();
  const BasicBlock *BB = MBB->getBasicBlock();
  const Function &F = MF->getFunction();
  if (LandingPadSuccs.size() > 1 &&
      !(AsmInfo &&
        AsmInfo->getExceptionHandlingType() == ExceptionHandling::SjLj &&
        BB && isa<SwitchInst>(BB->getTerminator())) &&
      !isScopedEHPersonality(classifyEHPersonality(F.getPersonalityFn())))
    report("MBB has more than one landing pad successor", MBB);

  // Call analyzeBranch. If it succeeds, there several more conditions to check.
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (!TII->analyzeBranch(*const_cast<MachineBasicBlock *>(MBB), TBB, FBB,
                          Cond)) {
    // Ok, analyzeBranch thinks it knows what's going on with this block. Let's
    // check whether its answers match up with reality.
    if (!TBB && !FBB) {
      // Block falls through to its successor.
      if (!MBB->empty() && MBB->back().isBarrier() &&
          !TII->isPredicated(MBB->back())) {
        report("MBB exits via unconditional fall-through but ends with a "
               "barrier instruction!", MBB);
      }
      if (!Cond.empty()) {
        report("MBB exits via unconditional fall-through but has a condition!",
               MBB);
      }
    } else if (TBB && !FBB && Cond.empty()) {
      // Block unconditionally branches somewhere.
      if (MBB->empty()) {
        report("MBB exits via unconditional branch but doesn't contain "
               "any instructions!", MBB);
      } else if (!MBB->back().isBarrier()) {
        report("MBB exits via unconditional branch but doesn't end with a "
               "barrier instruction!", MBB);
      } else if (!MBB->back().isTerminator()) {
        report("MBB exits via unconditional branch but the branch isn't a "
               "terminator instruction!", MBB);
      }
    } else if (TBB && !FBB && !Cond.empty()) {
      // Block conditionally branches somewhere, otherwise falls through.
      if (MBB->empty()) {
        report("MBB exits via conditional branch/fall-through but doesn't "
               "contain any instructions!", MBB);
      } else if (MBB->back().isBarrier()) {
        report("MBB exits via conditional branch/fall-through but ends with a "
               "barrier instruction!", MBB);
      } else if (!MBB->back().isTerminator()) {
        report("MBB exits via conditional branch/fall-through but the branch "
               "isn't a terminator instruction!", MBB);
      }
    } else if (TBB && FBB) {
      // Block conditionally branches somewhere, otherwise branches
      // somewhere else.
      if (MBB->empty()) {
        report("MBB exits via conditional branch/branch but doesn't "
               "contain any instructions!", MBB);
      } else if (!MBB->back().isBarrier()) {
        report("MBB exits via conditional branch/branch but doesn't end with a "
               "barrier instruction!", MBB);
      } else if (!MBB->back().isTerminator()) {
        report("MBB exits via conditional branch/branch but the branch "
               "isn't a terminator instruction!", MBB);
      }
      if (Cond.empty()) {
        report("MBB exits via conditional branch/branch but there's no "
               "condition!", MBB);
      }
    } else {
      report("analyzeBranch returned invalid data!", MBB);
    }

    // Now check that the successors match up with the answers reported by
    // analyzeBranch.
    if (TBB && !MBB->isSuccessor(TBB))
      report("MBB exits via jump or conditional branch, but its target isn't a "
             "CFG successor!",
             MBB);
    if (FBB && !MBB->isSuccessor(FBB))
      report("MBB exits via conditional branch, but its target isn't a CFG "
             "successor!",
             MBB);

    // There might be a fallthrough to the next block if there's either no
    // unconditional true branch, or if there's a condition, and one of the
    // branches is missing.
    bool Fallthrough = !TBB || (!Cond.empty() && !FBB);

    // A conditional fallthrough must be an actual CFG successor, not
    // unreachable. (Conversely, an unconditional fallthrough might not really
    // be a successor, because the block might end in unreachable.)
    if (!Cond.empty() && !FBB) {
      MachineFunction::const_iterator MBBI = std::next(MBB->getIterator());
      if (MBBI == MF->end()) {
        report("MBB conditionally falls through out of function!", MBB);
      } else if (!MBB->isSuccessor(&*MBBI))
        report("MBB exits via conditional branch/fall-through but the CFG "
               "successors don't match the actual successors!",
               MBB);
    }

    // Verify that there aren't any extra un-accounted-for successors.
    for (const MachineBasicBlock *SuccMBB : MBB->successors()) {
      // If this successor is one of the branch targets, it's okay.
      if (SuccMBB == TBB || SuccMBB == FBB)
        continue;
      // If we might have a fallthrough, and the successor is the fallthrough
      // block, that's also ok.
      if (Fallthrough && SuccMBB == MBB->getNextNode())
        continue;
      // Also accept successors which are for exception-handling or might be
      // inlineasm_br targets.
      if (SuccMBB->isEHPad() || MBB->isInlineAsmBrIndirectTarget(SuccMBB))
        continue;
      report("MBB has unexpected successors which are not branch targets, "
             "fallthrough, EHPads, or inlineasm_br targets.",
             MBB);
    }
  }

  regsLive.clear();
  if (MRI->tracksLiveness()) {
    for (const auto &LI : MBB->liveins()) {
      if (!Register::isPhysicalRegister(LI.PhysReg)) {
        report("MBB live-in list contains non-physical register", MBB);
        continue;
      }
      for (const MCPhysReg &SubReg : TRI->subregs_inclusive(LI.PhysReg))
        regsLive.insert(SubReg);
    }
  }

  const MachineFrameInfo &MFI = MF->getFrameInfo();
  BitVector PR = MFI.getPristineRegs(*MF);
  for (unsigned I : PR.set_bits()) {
    for (const MCPhysReg &SubReg : TRI->subregs_inclusive(I))
      regsLive.insert(SubReg);
  }

  regsKilled.clear();
  regsDefined.clear();

  if (Indexes)
    lastIndex = Indexes->getMBBStartIdx(MBB);
}

// This function gets called for all bundle headers, including normal
// stand-alone unbundled instructions.
void MachineVerifier::visitMachineBundleBefore(const MachineInstr *MI) {
  if (Indexes && Indexes->hasIndex(*MI)) {
    SlotIndex idx = Indexes->getInstructionIndex(*MI);
    if (!(idx > lastIndex)) {
      report("Instruction index out of order", MI);
      errs() << "Last instruction was at " << lastIndex << '\n';
    }
    lastIndex = idx;
  }

  // Ensure non-terminators don't follow terminators.
  // Ignore predicated terminators formed by if conversion.
  // FIXME: If conversion shouldn't need to violate this rule.
  if (MI->isTerminator() && !TII->isPredicated(*MI)) {
    if (!FirstTerminator)
      FirstTerminator = MI;
  } else if (FirstTerminator) {
    report("Non-terminator instruction after the first terminator", MI);
    errs() << "First terminator was:\t" << *FirstTerminator;
  }
}

// The operands on an INLINEASM instruction must follow a template.
// Verify that the flag operands make sense.
void MachineVerifier::verifyInlineAsm(const MachineInstr *MI) {
  // The first two operands on INLINEASM are the asm string and global flags.
  if (MI->getNumOperands() < 2) {
    report("Too few operands on inline asm", MI);
    return;
  }
  if (!MI->getOperand(0).isSymbol())
    report("Asm string must be an external symbol", MI);
  if (!MI->getOperand(1).isImm())
    report("Asm flags must be an immediate", MI);
  // Allowed flags are Extra_HasSideEffects = 1, Extra_IsAlignStack = 2,
  // Extra_AsmDialect = 4, Extra_MayLoad = 8, and Extra_MayStore = 16,
  // and Extra_IsConvergent = 32.
  if (!isUInt<6>(MI->getOperand(1).getImm()))
    report("Unknown asm flags", &MI->getOperand(1), 1);

  static_assert(InlineAsm::MIOp_FirstOperand == 2, "Asm format changed");

  unsigned OpNo = InlineAsm::MIOp_FirstOperand;
  unsigned NumOps;
  for (unsigned e = MI->getNumOperands(); OpNo < e; OpNo += NumOps) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    // There may be implicit ops after the fixed operands.
    if (!MO.isImm())
      break;
    NumOps = 1 + InlineAsm::getNumOperandRegisters(MO.getImm());
  }

  if (OpNo > MI->getNumOperands())
    report("Missing operands in last group", MI);

  // An optional MDNode follows the groups.
  if (OpNo < MI->getNumOperands() && MI->getOperand(OpNo).isMetadata())
    ++OpNo;

  // All trailing operands must be implicit registers.
  for (unsigned e = MI->getNumOperands(); OpNo < e; ++OpNo) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    if (!MO.isReg() || !MO.isImplicit())
      report("Expected implicit register after groups", &MO, OpNo);
  }
}

/// Check that types are consistent when two operands need to have the same
/// number of vector elements.
/// \return true if the types are valid.
bool MachineVerifier::verifyVectorElementMatch(LLT Ty0, LLT Ty1,
                                               const MachineInstr *MI) {
  if (Ty0.isVector() != Ty1.isVector()) {
    report("operand types must be all-vector or all-scalar", MI);
    // Generally we try to report as many issues as possible at once, but in
    // this case it's not clear what should we be comparing the size of the
    // scalar with: the size of the whole vector or its lane. Instead of
    // making an arbitrary choice and emitting not so helpful message, let's
    // avoid the extra noise and stop here.
    return false;
  }

  if (Ty0.isVector() && Ty0.getNumElements() != Ty1.getNumElements()) {
    report("operand types must preserve number of vector elements", MI);
    return false;
  }

  return true;
}

void MachineVerifier::verifyPreISelGenericInstruction(const MachineInstr *MI) {
  if (isFunctionSelected)
    report("Unexpected generic instruction in a Selected function", MI);

  const MCInstrDesc &MCID = MI->getDesc();
  unsigned NumOps = MI->getNumOperands();

  // Branches must reference a basic block if they are not indirect
  if (MI->isBranch() && !MI->isIndirectBranch()) {
    bool HasMBB = false;
    for (const MachineOperand &Op : MI->operands()) {
      if (Op.isMBB()) {
        HasMBB = true;
        break;
      }
    }

    if (!HasMBB)
      report("Branch instruction is missing a basic block operand or "
             "isIndirectBranch property",
             MI);
  }

  // Check types.
  SmallVector<LLT, 4> Types;
  for (unsigned I = 0, E = std::min(MCID.getNumOperands(), NumOps);
       I != E; ++I) {
    if (!MCID.OpInfo[I].isGenericType())
      continue;
    // Generic instructions specify type equality constraints between some of
    // their operands. Make sure these are consistent.
    size_t TypeIdx = MCID.OpInfo[I].getGenericTypeIndex();
    Types.resize(std::max(TypeIdx + 1, Types.size()));

    const MachineOperand *MO = &MI->getOperand(I);
    if (!MO->isReg()) {
      report("generic instruction must use register operands", MI);
      continue;
    }

    LLT OpTy = MRI->getType(MO->getReg());
    // Don't report a type mismatch if there is no actual mismatch, only a
    // type missing, to reduce noise:
    if (OpTy.isValid()) {
      // Only the first valid type for a type index will be printed: don't
      // overwrite it later so it's always clear which type was expected:
      if (!Types[TypeIdx].isValid())
        Types[TypeIdx] = OpTy;
      else if (Types[TypeIdx] != OpTy)
        report("Type mismatch in generic instruction", MO, I, OpTy);
    } else {
      // Generic instructions must have types attached to their operands.
      report("Generic instruction is missing a virtual register type", MO, I);
    }
  }

  // Generic opcodes must not have physical register operands.
  for (unsigned I = 0; I < MI->getNumOperands(); ++I) {
    const MachineOperand *MO = &MI->getOperand(I);
    if (MO->isReg() && Register::isPhysicalRegister(MO->getReg()))
      report("Generic instruction cannot have physical register", MO, I);
  }

  // Avoid out of bounds in checks below. This was already reported earlier.
  if (MI->getNumOperands() < MCID.getNumOperands())
    return;

  StringRef ErrorInfo;
  if (!TII->verifyInstruction(*MI, ErrorInfo))
    report(ErrorInfo.data(), MI);

  // Verify properties of various specific instruction types
  switch (MI->getOpcode()) {
  case TargetOpcode::G_CONSTANT:
  case TargetOpcode::G_FCONSTANT: {
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    if (DstTy.isVector())
      report("Instruction cannot use a vector result type", MI);

    if (MI->getOpcode() == TargetOpcode::G_CONSTANT) {
      if (!MI->getOperand(1).isCImm()) {
        report("G_CONSTANT operand must be cimm", MI);
        break;
      }

      const ConstantInt *CI = MI->getOperand(1).getCImm();
      if (CI->getBitWidth() != DstTy.getSizeInBits())
        report("inconsistent constant size", MI);
    } else {
      if (!MI->getOperand(1).isFPImm()) {
        report("G_FCONSTANT operand must be fpimm", MI);
        break;
      }
      const ConstantFP *CF = MI->getOperand(1).getFPImm();

      if (APFloat::getSizeInBits(CF->getValueAPF().getSemantics()) !=
          DstTy.getSizeInBits()) {
        report("inconsistent constant size", MI);
      }
    }

    break;
  }
  case TargetOpcode::G_LOAD:
  case TargetOpcode::G_STORE:
  case TargetOpcode::G_ZEXTLOAD:
  case TargetOpcode::G_SEXTLOAD: {
    LLT ValTy = MRI->getType(MI->getOperand(0).getReg());
    LLT PtrTy = MRI->getType(MI->getOperand(1).getReg());
    if (!PtrTy.isPointer())
      report("Generic memory instruction must access a pointer", MI);

    // Generic loads and stores must have a single MachineMemOperand
    // describing that access.
    if (!MI->hasOneMemOperand()) {
      report("Generic instruction accessing memory must have one mem operand",
             MI);
    } else {
      const MachineMemOperand &MMO = **MI->memoperands_begin();
      if (MI->getOpcode() == TargetOpcode::G_ZEXTLOAD ||
          MI->getOpcode() == TargetOpcode::G_SEXTLOAD) {
        if (MMO.getSizeInBits() >= ValTy.getSizeInBits())
          report("Generic extload must have a narrower memory type", MI);
      } else if (MI->getOpcode() == TargetOpcode::G_LOAD) {
        if (MMO.getSize() > ValTy.getSizeInBytes())
          report("load memory size cannot exceed result size", MI);
      } else if (MI->getOpcode() == TargetOpcode::G_STORE) {
        if (ValTy.getSizeInBytes() < MMO.getSize())
          report("store memory size cannot exceed value size", MI);
      }
    }

    break;
  }
  case TargetOpcode::G_PHI: {
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    if (!DstTy.isValid() ||
        !std::all_of(MI->operands_begin() + 1, MI->operands_end(),
                     [this, &DstTy](const MachineOperand &MO) {
                       if (!MO.isReg())
                         return true;
                       LLT Ty = MRI->getType(MO.getReg());
                       if (!Ty.isValid() || (Ty != DstTy))
                         return false;
                       return true;
                     }))
      report("Generic Instruction G_PHI has operands with incompatible/missing "
             "types",
             MI);
    break;
  }
  case TargetOpcode::G_BITCAST: {
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT SrcTy = MRI->getType(MI->getOperand(1).getReg());
    if (!DstTy.isValid() || !SrcTy.isValid())
      break;

    if (SrcTy.isPointer() != DstTy.isPointer())
      report("bitcast cannot convert between pointers and other types", MI);

    if (SrcTy.getSizeInBits() != DstTy.getSizeInBits())
      report("bitcast sizes must match", MI);
    break;
  }
  case TargetOpcode::G_INTTOPTR:
  case TargetOpcode::G_PTRTOINT:
  case TargetOpcode::G_ADDRSPACE_CAST: {
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT SrcTy = MRI->getType(MI->getOperand(1).getReg());
    if (!DstTy.isValid() || !SrcTy.isValid())
      break;

    verifyVectorElementMatch(DstTy, SrcTy, MI);

    DstTy = DstTy.getScalarType();
    SrcTy = SrcTy.getScalarType();

    if (MI->getOpcode() == TargetOpcode::G_INTTOPTR) {
      if (!DstTy.isPointer())
        report("inttoptr result type must be a pointer", MI);
      if (SrcTy.isPointer())
        report("inttoptr source type must not be a pointer", MI);
    } else if (MI->getOpcode() == TargetOpcode::G_PTRTOINT) {
      if (!SrcTy.isPointer())
        report("ptrtoint source type must be a pointer", MI);
      if (DstTy.isPointer())
        report("ptrtoint result type must not be a pointer", MI);
    } else {
      assert(MI->getOpcode() == TargetOpcode::G_ADDRSPACE_CAST);
      if (!SrcTy.isPointer() || !DstTy.isPointer())
        report("addrspacecast types must be pointers", MI);
      else {
        if (SrcTy.getAddressSpace() == DstTy.getAddressSpace())
          report("addrspacecast must convert different address spaces", MI);
      }
    }

    break;
  }
  case TargetOpcode::G_PTR_ADD: {
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT PtrTy = MRI->getType(MI->getOperand(1).getReg());
    LLT OffsetTy = MRI->getType(MI->getOperand(2).getReg());
    if (!DstTy.isValid() || !PtrTy.isValid() || !OffsetTy.isValid())
      break;

    if (!PtrTy.getScalarType().isPointer())
      report("gep first operand must be a pointer", MI);

    if (OffsetTy.getScalarType().isPointer())
      report("gep offset operand must not be a pointer", MI);

    // TODO: Is the offset allowed to be a scalar with a vector?
    break;
  }
  case TargetOpcode::G_PTRMASK: {
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT SrcTy = MRI->getType(MI->getOperand(1).getReg());
    LLT MaskTy = MRI->getType(MI->getOperand(2).getReg());
    if (!DstTy.isValid() || !SrcTy.isValid() || !MaskTy.isValid())
      break;

    if (!DstTy.getScalarType().isPointer())
      report("ptrmask result type must be a pointer", MI);

    if (!MaskTy.getScalarType().isScalar())
      report("ptrmask mask type must be an integer", MI);

    verifyVectorElementMatch(DstTy, MaskTy, MI);
    break;
  }
  case TargetOpcode::G_SEXT:
  case TargetOpcode::G_ZEXT:
  case TargetOpcode::G_ANYEXT:
  case TargetOpcode::G_TRUNC:
  case TargetOpcode::G_FPEXT:
  case TargetOpcode::G_FPTRUNC: {
    // Number of operands and presense of types is already checked (and
    // reported in case of any issues), so no need to report them again. As
    // we're trying to report as many issues as possible at once, however, the
    // instructions aren't guaranteed to have the right number of operands or
    // types attached to them at this point
    assert(MCID.getNumOperands() == 2 && "Expected 2 operands G_*{EXT,TRUNC}");
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT SrcTy = MRI->getType(MI->getOperand(1).getReg());
    if (!DstTy.isValid() || !SrcTy.isValid())
      break;

    LLT DstElTy = DstTy.getScalarType();
    LLT SrcElTy = SrcTy.getScalarType();
    if (DstElTy.isPointer() || SrcElTy.isPointer())
      report("Generic extend/truncate can not operate on pointers", MI);

    verifyVectorElementMatch(DstTy, SrcTy, MI);

    unsigned DstSize = DstElTy.getSizeInBits();
    unsigned SrcSize = SrcElTy.getSizeInBits();
    switch (MI->getOpcode()) {
    default:
      if (DstSize <= SrcSize)
        report("Generic extend has destination type no larger than source", MI);
      break;
    case TargetOpcode::G_TRUNC:
    case TargetOpcode::G_FPTRUNC:
      if (DstSize >= SrcSize)
        report("Generic truncate has destination type no smaller than source",
               MI);
      break;
    }
    break;
  }
  case TargetOpcode::G_SELECT: {
    LLT SelTy = MRI->getType(MI->getOperand(0).getReg());
    LLT CondTy = MRI->getType(MI->getOperand(1).getReg());
    if (!SelTy.isValid() || !CondTy.isValid())
      break;

    // Scalar condition select on a vector is valid.
    if (CondTy.isVector())
      verifyVectorElementMatch(SelTy, CondTy, MI);
    break;
  }
  case TargetOpcode::G_MERGE_VALUES: {
    // G_MERGE_VALUES should only be used to merge scalars into a larger scalar,
    // e.g. s2N = MERGE sN, sN
    // Merging multiple scalars into a vector is not allowed, should use
    // G_BUILD_VECTOR for that.
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT SrcTy = MRI->getType(MI->getOperand(1).getReg());
    if (DstTy.isVector() || SrcTy.isVector())
      report("G_MERGE_VALUES cannot operate on vectors", MI);

    const unsigned NumOps = MI->getNumOperands();
    if (DstTy.getSizeInBits() != SrcTy.getSizeInBits() * (NumOps - 1))
      report("G_MERGE_VALUES result size is inconsistent", MI);

    for (unsigned I = 2; I != NumOps; ++I) {
      if (MRI->getType(MI->getOperand(I).getReg()) != SrcTy)
        report("G_MERGE_VALUES source types do not match", MI);
    }

    break;
  }
  case TargetOpcode::G_UNMERGE_VALUES: {
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT SrcTy = MRI->getType(MI->getOperand(MI->getNumOperands()-1).getReg());
    // For now G_UNMERGE can split vectors.
    for (unsigned i = 0; i < MI->getNumOperands()-1; ++i) {
      if (MRI->getType(MI->getOperand(i).getReg()) != DstTy)
        report("G_UNMERGE_VALUES destination types do not match", MI);
    }
    if (SrcTy.getSizeInBits() !=
        (DstTy.getSizeInBits() * (MI->getNumOperands() - 1))) {
      report("G_UNMERGE_VALUES source operand does not cover dest operands",
             MI);
    }
    break;
  }
  case TargetOpcode::G_BUILD_VECTOR: {
    // Source types must be scalars, dest type a vector. Total size of scalars
    // must match the dest vector size.
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT SrcEltTy = MRI->getType(MI->getOperand(1).getReg());
    if (!DstTy.isVector() || SrcEltTy.isVector()) {
      report("G_BUILD_VECTOR must produce a vector from scalar operands", MI);
      break;
    }

    if (DstTy.getElementType() != SrcEltTy)
      report("G_BUILD_VECTOR result element type must match source type", MI);

    if (DstTy.getNumElements() != MI->getNumOperands() - 1)
      report("G_BUILD_VECTOR must have an operand for each elemement", MI);

    for (unsigned i = 2; i < MI->getNumOperands(); ++i) {
      if (MRI->getType(MI->getOperand(1).getReg()) !=
          MRI->getType(MI->getOperand(i).getReg()))
        report("G_BUILD_VECTOR source operand types are not homogeneous", MI);
    }

    break;
  }
  case TargetOpcode::G_BUILD_VECTOR_TRUNC: {
    // Source types must be scalars, dest type a vector. Scalar types must be
    // larger than the dest vector elt type, as this is a truncating operation.
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT SrcEltTy = MRI->getType(MI->getOperand(1).getReg());
    if (!DstTy.isVector() || SrcEltTy.isVector())
      report("G_BUILD_VECTOR_TRUNC must produce a vector from scalar operands",
             MI);
    for (unsigned i = 2; i < MI->getNumOperands(); ++i) {
      if (MRI->getType(MI->getOperand(1).getReg()) !=
          MRI->getType(MI->getOperand(i).getReg()))
        report("G_BUILD_VECTOR_TRUNC source operand types are not homogeneous",
               MI);
    }
    if (SrcEltTy.getSizeInBits() <= DstTy.getElementType().getSizeInBits())
      report("G_BUILD_VECTOR_TRUNC source operand types are not larger than "
             "dest elt type",
             MI);
    break;
  }
  case TargetOpcode::G_CONCAT_VECTORS: {
    // Source types should be vectors, and total size should match the dest
    // vector size.
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT SrcTy = MRI->getType(MI->getOperand(1).getReg());
    if (!DstTy.isVector() || !SrcTy.isVector())
      report("G_CONCAT_VECTOR requires vector source and destination operands",
             MI);
    for (unsigned i = 2; i < MI->getNumOperands(); ++i) {
      if (MRI->getType(MI->getOperand(1).getReg()) !=
          MRI->getType(MI->getOperand(i).getReg()))
        report("G_CONCAT_VECTOR source operand types are not homogeneous", MI);
    }
    if (DstTy.getNumElements() !=
        SrcTy.getNumElements() * (MI->getNumOperands() - 1))
      report("G_CONCAT_VECTOR num dest and source elements should match", MI);
    break;
  }
  case TargetOpcode::G_ICMP:
  case TargetOpcode::G_FCMP: {
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT SrcTy = MRI->getType(MI->getOperand(2).getReg());

    if ((DstTy.isVector() != SrcTy.isVector()) ||
        (DstTy.isVector() && DstTy.getNumElements() != SrcTy.getNumElements()))
      report("Generic vector icmp/fcmp must preserve number of lanes", MI);

    break;
  }
  case TargetOpcode::G_EXTRACT: {
    const MachineOperand &SrcOp = MI->getOperand(1);
    if (!SrcOp.isReg()) {
      report("extract source must be a register", MI);
      break;
    }

    const MachineOperand &OffsetOp = MI->getOperand(2);
    if (!OffsetOp.isImm()) {
      report("extract offset must be a constant", MI);
      break;
    }

    unsigned DstSize = MRI->getType(MI->getOperand(0).getReg()).getSizeInBits();
    unsigned SrcSize = MRI->getType(SrcOp.getReg()).getSizeInBits();
    if (SrcSize == DstSize)
      report("extract source must be larger than result", MI);

    if (DstSize + OffsetOp.getImm() > SrcSize)
      report("extract reads past end of register", MI);
    break;
  }
  case TargetOpcode::G_INSERT: {
    const MachineOperand &SrcOp = MI->getOperand(2);
    if (!SrcOp.isReg()) {
      report("insert source must be a register", MI);
      break;
    }

    const MachineOperand &OffsetOp = MI->getOperand(3);
    if (!OffsetOp.isImm()) {
      report("insert offset must be a constant", MI);
      break;
    }

    unsigned DstSize = MRI->getType(MI->getOperand(0).getReg()).getSizeInBits();
    unsigned SrcSize = MRI->getType(SrcOp.getReg()).getSizeInBits();

    if (DstSize <= SrcSize)
      report("inserted size must be smaller than total register", MI);

    if (SrcSize + OffsetOp.getImm() > DstSize)
      report("insert writes past end of register", MI);

    break;
  }
  case TargetOpcode::G_JUMP_TABLE: {
    if (!MI->getOperand(1).isJTI())
      report("G_JUMP_TABLE source operand must be a jump table index", MI);
    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    if (!DstTy.isPointer())
      report("G_JUMP_TABLE dest operand must have a pointer type", MI);
    break;
  }
  case TargetOpcode::G_BRJT: {
    if (!MRI->getType(MI->getOperand(0).getReg()).isPointer())
      report("G_BRJT src operand 0 must be a pointer type", MI);

    if (!MI->getOperand(1).isJTI())
      report("G_BRJT src operand 1 must be a jump table index", MI);

    const auto &IdxOp = MI->getOperand(2);
    if (!IdxOp.isReg() || MRI->getType(IdxOp.getReg()).isPointer())
      report("G_BRJT src operand 2 must be a scalar reg type", MI);
    break;
  }
  case TargetOpcode::G_INTRINSIC:
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS: {
    // TODO: Should verify number of def and use operands, but the current
    // interface requires passing in IR types for mangling.
    const MachineOperand &IntrIDOp = MI->getOperand(MI->getNumExplicitDefs());
    if (!IntrIDOp.isIntrinsicID()) {
      report("G_INTRINSIC first src operand must be an intrinsic ID", MI);
      break;
    }

    bool NoSideEffects = MI->getOpcode() == TargetOpcode::G_INTRINSIC;
    unsigned IntrID = IntrIDOp.getIntrinsicID();
    if (IntrID != 0 && IntrID < Intrinsic::num_intrinsics) {
      AttributeList Attrs
        = Intrinsic::getAttributes(MF->getFunction().getContext(),
                                   static_cast<Intrinsic::ID>(IntrID));
      bool DeclHasSideEffects = !Attrs.hasFnAttribute(Attribute::ReadNone);
      if (NoSideEffects && DeclHasSideEffects) {
        report("G_INTRINSIC used with intrinsic that accesses memory", MI);
        break;
      }
      if (!NoSideEffects && !DeclHasSideEffects) {
        report("G_INTRINSIC_W_SIDE_EFFECTS used with readnone intrinsic", MI);
        break;
      }
    }
    switch (IntrID) {
    case Intrinsic::memcpy:
      if (MI->getNumOperands() != 5)
        report("Expected memcpy intrinsic to have 5 operands", MI);
      break;
    case Intrinsic::memmove:
      if (MI->getNumOperands() != 5)
        report("Expected memmove intrinsic to have 5 operands", MI);
      break;
    case Intrinsic::memset:
      if (MI->getNumOperands() != 5)
        report("Expected memset intrinsic to have 5 operands", MI);
      break;
    }
    break;
  }
  case TargetOpcode::G_SEXT_INREG: {
    if (!MI->getOperand(2).isImm()) {
      report("G_SEXT_INREG expects an immediate operand #2", MI);
      break;
    }

    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT SrcTy = MRI->getType(MI->getOperand(1).getReg());
    verifyVectorElementMatch(DstTy, SrcTy, MI);

    int64_t Imm = MI->getOperand(2).getImm();
    if (Imm <= 0)
      report("G_SEXT_INREG size must be >= 1", MI);
    if (Imm >= SrcTy.getScalarSizeInBits())
      report("G_SEXT_INREG size must be less than source bit width", MI);
    break;
  }
  case TargetOpcode::G_SHUFFLE_VECTOR: {
    const MachineOperand &MaskOp = MI->getOperand(3);
    if (!MaskOp.isShuffleMask()) {
      report("Incorrect mask operand type for G_SHUFFLE_VECTOR", MI);
      break;
    }

    LLT DstTy = MRI->getType(MI->getOperand(0).getReg());
    LLT Src0Ty = MRI->getType(MI->getOperand(1).getReg());
    LLT Src1Ty = MRI->getType(MI->getOperand(2).getReg());

    if (Src0Ty != Src1Ty)
      report("Source operands must be the same type", MI);

    if (Src0Ty.getScalarType() != DstTy.getScalarType())
      report("G_SHUFFLE_VECTOR cannot change element type", MI);

    // Don't check that all operands are vector because scalars are used in
    // place of 1 element vectors.
    int SrcNumElts = Src0Ty.isVector() ? Src0Ty.getNumElements() : 1;
    int DstNumElts = DstTy.isVector() ? DstTy.getNumElements() : 1;

    ArrayRef<int> MaskIdxes = MaskOp.getShuffleMask();

    if (static_cast<int>(MaskIdxes.size()) != DstNumElts)
      report("Wrong result type for shufflemask", MI);

    for (int Idx : MaskIdxes) {
      if (Idx < 0)
        continue;

      if (Idx >= 2 * SrcNumElts)
        report("Out of bounds shuffle index", MI);
    }

    break;
  }
  case TargetOpcode::G_DYN_STACKALLOC: {
    const MachineOperand &DstOp = MI->getOperand(0);
    const MachineOperand &AllocOp = MI->getOperand(1);
    const MachineOperand &AlignOp = MI->getOperand(2);

    if (!DstOp.isReg() || !MRI->getType(DstOp.getReg()).isPointer()) {
      report("dst operand 0 must be a pointer type", MI);
      break;
    }

    if (!AllocOp.isReg() || !MRI->getType(AllocOp.getReg()).isScalar()) {
      report("src operand 1 must be a scalar reg type", MI);
      break;
    }

    if (!AlignOp.isImm()) {
      report("src operand 2 must be an immediate type", MI);
      break;
    }
    break;
  }
  default:
    break;
  }
}

void MachineVerifier::visitMachineInstrBefore(const MachineInstr *MI) {
  const MCInstrDesc &MCID = MI->getDesc();
  if (MI->getNumOperands() < MCID.getNumOperands()) {
    report("Too few operands", MI);
    errs() << MCID.getNumOperands() << " operands expected, but "
           << MI->getNumOperands() << " given.\n";
  }

  if (MI->isPHI()) {
    if (MF->getProperties().hasProperty(
            MachineFunctionProperties::Property::NoPHIs))
      report("Found PHI instruction with NoPHIs property set", MI);

    if (FirstNonPHI)
      report("Found PHI instruction after non-PHI", MI);
  } else if (FirstNonPHI == nullptr)
    FirstNonPHI = MI;

  // Check the tied operands.
  if (MI->isInlineAsm())
    verifyInlineAsm(MI);

  // A fully-formed DBG_VALUE must have a location. Ignore partially formed
  // DBG_VALUEs: these are convenient to use in tests, but should never get
  // generated.
  if (MI->isDebugValue() && MI->getNumOperands() == 4)
    if (!MI->getDebugLoc())
      report("Missing DebugLoc for debug instruction", MI);

  // Check the MachineMemOperands for basic consistency.
  for (MachineMemOperand *Op : MI->memoperands()) {
    if (Op->isLoad() && !MI->mayLoad())
      report("Missing mayLoad flag", MI);
    if (Op->isStore() && !MI->mayStore())
      report("Missing mayStore flag", MI);
  }

  // Debug values must not have a slot index.
  // Other instructions must have one, unless they are inside a bundle.
  if (LiveInts) {
    bool mapped = !LiveInts->isNotInMIMap(*MI);
    if (MI->isDebugInstr()) {
      if (mapped)
        report("Debug instruction has a slot index", MI);
    } else if (MI->isInsideBundle()) {
      if (mapped)
        report("Instruction inside bundle has a slot index", MI);
    } else {
      if (!mapped)
        report("Missing slot index", MI);
    }
  }

  if (isPreISelGenericOpcode(MCID.getOpcode())) {
    verifyPreISelGenericInstruction(MI);
    return;
  }

  StringRef ErrorInfo;
  if (!TII->verifyInstruction(*MI, ErrorInfo))
    report(ErrorInfo.data(), MI);

  // Verify properties of various specific instruction types
  switch (MI->getOpcode()) {
  case TargetOpcode::COPY: {
    if (foundErrors)
      break;
    const MachineOperand &DstOp = MI->getOperand(0);
    const MachineOperand &SrcOp = MI->getOperand(1);
    LLT DstTy = MRI->getType(DstOp.getReg());
    LLT SrcTy = MRI->getType(SrcOp.getReg());
    if (SrcTy.isValid() && DstTy.isValid()) {
      // If both types are valid, check that the types are the same.
      if (SrcTy != DstTy) {
        report("Copy Instruction is illegal with mismatching types", MI);
        errs() << "Def = " << DstTy << ", Src = " << SrcTy << "\n";
      }
    }
    if (SrcTy.isValid() || DstTy.isValid()) {
      // If one of them have valid types, let's just check they have the same
      // size.
      unsigned SrcSize = TRI->getRegSizeInBits(SrcOp.getReg(), *MRI);
      unsigned DstSize = TRI->getRegSizeInBits(DstOp.getReg(), *MRI);
      assert(SrcSize && "Expecting size here");
      assert(DstSize && "Expecting size here");
      if (SrcSize != DstSize)
        if (!DstOp.getSubReg() && !SrcOp.getSubReg()) {
          report("Copy Instruction is illegal with mismatching sizes", MI);
          errs() << "Def Size = " << DstSize << ", Src Size = " << SrcSize
                 << "\n";
        }
    }
    break;
  }
  case TargetOpcode::STATEPOINT: {
    StatepointOpers SO(MI);
    if (!MI->getOperand(SO.getIDPos()).isImm() ||
        !MI->getOperand(SO.getNBytesPos()).isImm() ||
        !MI->getOperand(SO.getNCallArgsPos()).isImm()) {
      report("meta operands to STATEPOINT not constant!", MI);
      break;
    }

    auto VerifyStackMapConstant = [&](unsigned Offset) {
      if (!MI->getOperand(Offset - 1).isImm() ||
          MI->getOperand(Offset - 1).getImm() != StackMaps::ConstantOp ||
          !MI->getOperand(Offset).isImm())
        report("stack map constant to STATEPOINT not well formed!", MI);
    };
    VerifyStackMapConstant(SO.getCCIdx());
    VerifyStackMapConstant(SO.getFlagsIdx());
    VerifyStackMapConstant(SO.getNumDeoptArgsIdx());

    // TODO: verify we have properly encoded deopt arguments
  } break;
  }
}

void
MachineVerifier::visitMachineOperand(const MachineOperand *MO, unsigned MONum) {
  const MachineInstr *MI = MO->getParent();
  const MCInstrDesc &MCID = MI->getDesc();
  unsigned NumDefs = MCID.getNumDefs();
  if (MCID.getOpcode() == TargetOpcode::PATCHPOINT)
    NumDefs = (MONum == 0 && MO->isReg()) ? NumDefs : 0;

  // The first MCID.NumDefs operands must be explicit register defines
  if (MONum < NumDefs) {
    const MCOperandInfo &MCOI = MCID.OpInfo[MONum];
    if (!MO->isReg())
      report("Explicit definition must be a register", MO, MONum);
    else if (!MO->isDef() && !MCOI.isOptionalDef())
      report("Explicit definition marked as use", MO, MONum);
    else if (MO->isImplicit())
      report("Explicit definition marked as implicit", MO, MONum);
  } else if (MONum < MCID.getNumOperands()) {
    const MCOperandInfo &MCOI = MCID.OpInfo[MONum];
    // Don't check if it's the last operand in a variadic instruction. See,
    // e.g., LDM_RET in the arm back end. Check non-variadic operands only.
    bool IsOptional = MI->isVariadic() && MONum == MCID.getNumOperands() - 1;
    if (!IsOptional) {
      if (MO->isReg()) {
        if (MO->isDef() && !MCOI.isOptionalDef() && !MCID.variadicOpsAreDefs())
          report("Explicit operand marked as def", MO, MONum);
        if (MO->isImplicit())
          report("Explicit operand marked as implicit", MO, MONum);
      }

      // Check that an instruction has register operands only as expected.
      if (MCOI.OperandType == MCOI::OPERAND_REGISTER &&
          !MO->isReg() && !MO->isFI())
        report("Expected a register operand.", MO, MONum);
      if ((MCOI.OperandType == MCOI::OPERAND_IMMEDIATE ||
           MCOI.OperandType == MCOI::OPERAND_PCREL) && MO->isReg())
        report("Expected a non-register operand.", MO, MONum);
    }

    int TiedTo = MCID.getOperandConstraint(MONum, MCOI::TIED_TO);
    if (TiedTo != -1) {
      if (!MO->isReg())
        report("Tied use must be a register", MO, MONum);
      else if (!MO->isTied())
        report("Operand should be tied", MO, MONum);
      else if (unsigned(TiedTo) != MI->findTiedOperandIdx(MONum))
        report("Tied def doesn't match MCInstrDesc", MO, MONum);
      else if (Register::isPhysicalRegister(MO->getReg())) {
        const MachineOperand &MOTied = MI->getOperand(TiedTo);
        if (!MOTied.isReg())
          report("Tied counterpart must be a register", &MOTied, TiedTo);
        else if (Register::isPhysicalRegister(MOTied.getReg()) &&
                 MO->getReg() != MOTied.getReg())
          report("Tied physical registers must match.", &MOTied, TiedTo);
      }
    } else if (MO->isReg() && MO->isTied())
      report("Explicit operand should not be tied", MO, MONum);
  } else {
    // ARM adds %reg0 operands to indicate predicates. We'll allow that.
    if (MO->isReg() && !MO->isImplicit() && !MI->isVariadic() && MO->getReg())
      report("Extra explicit operand on non-variadic instruction", MO, MONum);
  }

  switch (MO->getType()) {
  case MachineOperand::MO_Register: {
    const Register Reg = MO->getReg();
    if (!Reg)
      return;
    if (MRI->tracksLiveness() && !MI->isDebugValue())
      checkLiveness(MO, MONum);

    // Verify the consistency of tied operands.
    if (MO->isTied()) {
      unsigned OtherIdx = MI->findTiedOperandIdx(MONum);
      const MachineOperand &OtherMO = MI->getOperand(OtherIdx);
      if (!OtherMO.isReg())
        report("Must be tied to a register", MO, MONum);
      if (!OtherMO.isTied())
        report("Missing tie flags on tied operand", MO, MONum);
      if (MI->findTiedOperandIdx(OtherIdx) != MONum)
        report("Inconsistent tie links", MO, MONum);
      if (MONum < MCID.getNumDefs()) {
        if (OtherIdx < MCID.getNumOperands()) {
          if (-1 == MCID.getOperandConstraint(OtherIdx, MCOI::TIED_TO))
            report("Explicit def tied to explicit use without tie constraint",
                   MO, MONum);
        } else {
          if (!OtherMO.isImplicit())
            report("Explicit def should be tied to implicit use", MO, MONum);
        }
      }
    }

    // Verify two-address constraints after the twoaddressinstruction pass.
    // Both twoaddressinstruction pass and phi-node-elimination pass call
    // MRI->leaveSSA() to set MF as NoSSA, we should do the verification after
    // twoaddressinstruction pass not after phi-node-elimination pass. So we
    // shouldn't use the NoSSA as the condition, we should based on
    // TiedOpsRewritten property to verify two-address constraints, this
    // property will be set in twoaddressinstruction pass.
    unsigned DefIdx;
    if (MF->getProperties().hasProperty(
            MachineFunctionProperties::Property::TiedOpsRewritten) &&
        MO->isUse() && MI->isRegTiedToDefOperand(MONum, &DefIdx) &&
        Reg != MI->getOperand(DefIdx).getReg())
      report("Two-address instruction operands must be identical", MO, MONum);

    // Check register classes.
    unsigned SubIdx = MO->getSubReg();

    if (Register::isPhysicalRegister(Reg)) {
      if (SubIdx) {
        report("Illegal subregister index for physical register", MO, MONum);
        return;
      }
      if (MONum < MCID.getNumOperands()) {
        if (const TargetRegisterClass *DRC =
              TII->getRegClass(MCID, MONum, TRI, *MF)) {
          if (!DRC->contains(Reg)) {
            report("Illegal physical register for instruction", MO, MONum);
            errs() << printReg(Reg, TRI) << " is not a "
                   << TRI->getRegClassName(DRC) << " register.\n";
          }
        }
      }
      if (MO->isRenamable()) {
        if (MRI->isReserved(Reg)) {
          report("isRenamable set on reserved register", MO, MONum);
          return;
        }
      }
      if (MI->isDebugValue() && MO->isUse() && !MO->isDebug()) {
        report("Use-reg is not IsDebug in a DBG_VALUE", MO, MONum);
        return;
      }
    } else {
      // Virtual register.
      const TargetRegisterClass *RC = MRI->getRegClassOrNull(Reg);
      if (!RC) {
        // This is a generic virtual register.

        // If we're post-Select, we can't have gvregs anymore.
        if (isFunctionSelected) {
          report("Generic virtual register invalid in a Selected function",
                 MO, MONum);
          return;
        }

        // The gvreg must have a type and it must not have a SubIdx.
        LLT Ty = MRI->getType(Reg);
        if (!Ty.isValid()) {
          report("Generic virtual register must have a valid type", MO,
                 MONum);
          return;
        }

        const RegisterBank *RegBank = MRI->getRegBankOrNull(Reg);

        // If we're post-RegBankSelect, the gvreg must have a bank.
        if (!RegBank && isFunctionRegBankSelected) {
          report("Generic virtual register must have a bank in a "
                 "RegBankSelected function",
                 MO, MONum);
          return;
        }

        // Make sure the register fits into its register bank if any.
        if (RegBank && Ty.isValid() &&
            RegBank->getSize() < Ty.getSizeInBits()) {
          report("Register bank is too small for virtual register", MO,
                 MONum);
          errs() << "Register bank " << RegBank->getName() << " too small("
                 << RegBank->getSize() << ") to fit " << Ty.getSizeInBits()
                 << "-bits\n";
          return;
        }
        if (SubIdx)  {
          report("Generic virtual register does not allow subregister index", MO,
                 MONum);
          return;
        }

        // If this is a target specific instruction and this operand
        // has register class constraint, the virtual register must
        // comply to it.
        if (!isPreISelGenericOpcode(MCID.getOpcode()) &&
            MONum < MCID.getNumOperands() &&
            TII->getRegClass(MCID, MONum, TRI, *MF)) {
          report("Virtual register does not match instruction constraint", MO,
                 MONum);
          errs() << "Expect register class "
                 << TRI->getRegClassName(
                        TII->getRegClass(MCID, MONum, TRI, *MF))
                 << " but got nothing\n";
          return;
        }

        break;
      }
      if (SubIdx) {
        const TargetRegisterClass *SRC =
          TRI->getSubClassWithSubReg(RC, SubIdx);
        if (!SRC) {
          report("Invalid subregister index for virtual register", MO, MONum);
          errs() << "Register class " << TRI->getRegClassName(RC)
              << " does not support subreg index " << SubIdx << "\n";
          return;
        }
        if (RC != SRC) {
          report("Invalid register class for subregister index", MO, MONum);
          errs() << "Register class " << TRI->getRegClassName(RC)
              << " does not fully support subreg index " << SubIdx << "\n";
          return;
        }
      }
      if (MONum < MCID.getNumOperands()) {
        if (const TargetRegisterClass *DRC =
              TII->getRegClass(MCID, MONum, TRI, *MF)) {
          if (SubIdx) {
            const TargetRegisterClass *SuperRC =
                TRI->getLargestLegalSuperClass(RC, *MF);
            if (!SuperRC) {
              report("No largest legal super class exists.", MO, MONum);
              return;
            }
            DRC = TRI->getMatchingSuperRegClass(SuperRC, DRC, SubIdx);
            if (!DRC) {
              report("No matching super-reg register class.", MO, MONum);
              return;
            }
          }
          if (!RC->hasSuperClassEq(DRC)) {
            report("Illegal virtual register for instruction", MO, MONum);
            errs() << "Expected a " << TRI->getRegClassName(DRC)
                << " register, but got a " << TRI->getRegClassName(RC)
                << " register\n";
          }
        }
      }
    }
    break;
  }

  case MachineOperand::MO_RegisterMask:
    regMasks.push_back(MO->getRegMask());
    break;

  case MachineOperand::MO_MachineBasicBlock:
    if (MI->isPHI() && !MO->getMBB()->isSuccessor(MI->getParent()))
      report("PHI operand is not in the CFG", MO, MONum);
    break;

  case MachineOperand::MO_FrameIndex:
    if (LiveStks && LiveStks->hasInterval(MO->getIndex()) &&
        LiveInts && !LiveInts->isNotInMIMap(*MI)) {
      int FI = MO->getIndex();
      LiveInterval &LI = LiveStks->getInterval(FI);
      SlotIndex Idx = LiveInts->getInstructionIndex(*MI);

      bool stores = MI->mayStore();
      bool loads = MI->mayLoad();
      // For a memory-to-memory move, we need to check if the frame
      // index is used for storing or loading, by inspecting the
      // memory operands.
      if (stores && loads) {
        for (auto *MMO : MI->memoperands()) {
          const PseudoSourceValue *PSV = MMO->getPseudoValue();
          if (PSV == nullptr) continue;
          const FixedStackPseudoSourceValue *Value =
            dyn_cast<FixedStackPseudoSourceValue>(PSV);
          if (Value == nullptr) continue;
          if (Value->getFrameIndex() != FI) continue;

          if (MMO->isStore())
            loads = false;
          else
            stores = false;
          break;
        }
        if (loads == stores)
          report("Missing fixed stack memoperand.", MI);
      }
      if (loads && !LI.liveAt(Idx.getRegSlot(true))) {
        report("Instruction loads from dead spill slot", MO, MONum);
        errs() << "Live stack: " << LI << '\n';
      }
      if (stores && !LI.liveAt(Idx.getRegSlot())) {
        report("Instruction stores to dead spill slot", MO, MONum);
        errs() << "Live stack: " << LI << '\n';
      }
    }
    break;

  default:
    break;
  }
}

void MachineVerifier::checkLivenessAtUse(const MachineOperand *MO,
    unsigned MONum, SlotIndex UseIdx, const LiveRange &LR, unsigned VRegOrUnit,
    LaneBitmask LaneMask) {
  LiveQueryResult LRQ = LR.Query(UseIdx);
  // Check if we have a segment at the use, note however that we only need one
  // live subregister range, the others may be dead.
  if (!LRQ.valueIn() && LaneMask.none()) {
    report("No live segment at use", MO, MONum);
    report_context_liverange(LR);
    report_context_vreg_regunit(VRegOrUnit);
    report_context(UseIdx);
  }
  if (MO->isKill() && !LRQ.isKill()) {
    report("Live range continues after kill flag", MO, MONum);
    report_context_liverange(LR);
    report_context_vreg_regunit(VRegOrUnit);
    if (LaneMask.any())
      report_context_lanemask(LaneMask);
    report_context(UseIdx);
  }
}

void MachineVerifier::checkLivenessAtDef(const MachineOperand *MO,
    unsigned MONum, SlotIndex DefIdx, const LiveRange &LR, unsigned VRegOrUnit,
    bool SubRangeCheck, LaneBitmask LaneMask) {
  if (const VNInfo *VNI = LR.getVNInfoAt(DefIdx)) {
    assert(VNI && "NULL valno is not allowed");
    if (VNI->def != DefIdx) {
      report("Inconsistent valno->def", MO, MONum);
      report_context_liverange(LR);
      report_context_vreg_regunit(VRegOrUnit);
      if (LaneMask.any())
        report_context_lanemask(LaneMask);
      report_context(*VNI);
      report_context(DefIdx);
    }
  } else {
    report("No live segment at def", MO, MONum);
    report_context_liverange(LR);
    report_context_vreg_regunit(VRegOrUnit);
    if (LaneMask.any())
      report_context_lanemask(LaneMask);
    report_context(DefIdx);
  }
  // Check that, if the dead def flag is present, LiveInts agree.
  if (MO->isDead()) {
    LiveQueryResult LRQ = LR.Query(DefIdx);
    if (!LRQ.isDeadDef()) {
      assert(Register::isVirtualRegister(VRegOrUnit) &&
             "Expecting a virtual register.");
      // A dead subreg def only tells us that the specific subreg is dead. There
      // could be other non-dead defs of other subregs, or we could have other
      // parts of the register being live through the instruction. So unless we
      // are checking liveness for a subrange it is ok for the live range to
      // continue, given that we have a dead def of a subregister.
      if (SubRangeCheck || MO->getSubReg() == 0) {
        report("Live range continues after dead def flag", MO, MONum);
        report_context_liverange(LR);
        report_context_vreg_regunit(VRegOrUnit);
        if (LaneMask.any())
          report_context_lanemask(LaneMask);
      }
    }
  }
}

void MachineVerifier::checkLiveness(const MachineOperand *MO, unsigned MONum) {
  const MachineInstr *MI = MO->getParent();
  const unsigned Reg = MO->getReg();

  // Both use and def operands can read a register.
  if (MO->readsReg()) {
    if (MO->isKill())
      addRegWithSubRegs(regsKilled, Reg);

    // Check that LiveVars knows this kill.
    if (LiveVars && Register::isVirtualRegister(Reg) && MO->isKill()) {
      LiveVariables::VarInfo &VI = LiveVars->getVarInfo(Reg);
      if (!is_contained(VI.Kills, MI))
        report("Kill missing from LiveVariables", MO, MONum);
    }

    // Check LiveInts liveness and kill.
    if (LiveInts && !LiveInts->isNotInMIMap(*MI)) {
      SlotIndex UseIdx = LiveInts->getInstructionIndex(*MI);
      // Check the cached regunit intervals.
      if (Register::isPhysicalRegister(Reg) && !isReserved(Reg)) {
        for (MCRegUnitIterator Units(Reg, TRI); Units.isValid(); ++Units) {
          if (MRI->isReservedRegUnit(*Units))
            continue;
          if (const LiveRange *LR = LiveInts->getCachedRegUnit(*Units))
            checkLivenessAtUse(MO, MONum, UseIdx, *LR, *Units);
        }
      }

      if (Register::isVirtualRegister(Reg)) {
        if (LiveInts->hasInterval(Reg)) {
          // This is a virtual register interval.
          const LiveInterval &LI = LiveInts->getInterval(Reg);
          checkLivenessAtUse(MO, MONum, UseIdx, LI, Reg);

          if (LI.hasSubRanges() && !MO->isDef()) {
            unsigned SubRegIdx = MO->getSubReg();
            LaneBitmask MOMask = SubRegIdx != 0
                               ? TRI->getSubRegIndexLaneMask(SubRegIdx)
                               : MRI->getMaxLaneMaskForVReg(Reg);
            LaneBitmask LiveInMask;
            for (const LiveInterval::SubRange &SR : LI.subranges()) {
              if ((MOMask & SR.LaneMask).none())
                continue;
              checkLivenessAtUse(MO, MONum, UseIdx, SR, Reg, SR.LaneMask);
              LiveQueryResult LRQ = SR.Query(UseIdx);
              if (LRQ.valueIn())
                LiveInMask |= SR.LaneMask;
            }
            // At least parts of the register has to be live at the use.
            if ((LiveInMask & MOMask).none()) {
              report("No live subrange at use", MO, MONum);
              report_context(LI);
              report_context(UseIdx);
            }
          }
        } else {
          report("Virtual register has no live interval", MO, MONum);
        }
      }
    }

    // Use of a dead register.
    if (!regsLive.count(Reg)) {
      if (Register::isPhysicalRegister(Reg)) {
        // Reserved registers may be used even when 'dead'.
        bool Bad = !isReserved(Reg);
        // We are fine if just any subregister has a defined value.
        if (Bad) {

          for (const MCPhysReg &SubReg : TRI->subregs(Reg)) {
            if (regsLive.count(SubReg)) {
              Bad = false;
              break;
            }
          }
        }
        // If there is an additional implicit-use of a super register we stop
        // here. By definition we are fine if the super register is not
        // (completely) dead, if the complete super register is dead we will
        // get a report for its operand.
        if (Bad) {
          for (const MachineOperand &MOP : MI->uses()) {
            if (!MOP.isReg() || !MOP.isImplicit())
              continue;

            if (!Register::isPhysicalRegister(MOP.getReg()))
              continue;

            for (const MCPhysReg &SubReg : TRI->subregs(MOP.getReg())) {
              if (SubReg == Reg) {
                Bad = false;
                break;
              }
            }
          }
        }
        if (Bad)
          report("Using an undefined physical register", MO, MONum);
      } else if (MRI->def_empty(Reg)) {
        report("Reading virtual register without a def", MO, MONum);
      } else {
        BBInfo &MInfo = MBBInfoMap[MI->getParent()];
        // We don't know which virtual registers are live in, so only complain
        // if vreg was killed in this MBB. Otherwise keep track of vregs that
        // must be live in. PHI instructions are handled separately.
        if (MInfo.regsKilled.count(Reg))
          report("Using a killed virtual register", MO, MONum);
        else if (!MI->isPHI())
          MInfo.vregsLiveIn.insert(std::make_pair(Reg, MI));
      }
    }
  }

  if (MO->isDef()) {
    // Register defined.
    // TODO: verify that earlyclobber ops are not used.
    if (MO->isDead())
      addRegWithSubRegs(regsDead, Reg);
    else
      addRegWithSubRegs(regsDefined, Reg);

    // Verify SSA form.
    if (MRI->isSSA() && Register::isVirtualRegister(Reg) &&
        std::next(MRI->def_begin(Reg)) != MRI->def_end())
      report("Multiple virtual register defs in SSA form", MO, MONum);

    // Check LiveInts for a live segment, but only for virtual registers.
    if (LiveInts && !LiveInts->isNotInMIMap(*MI)) {
      SlotIndex DefIdx = LiveInts->getInstructionIndex(*MI);
      DefIdx = DefIdx.getRegSlot(MO->isEarlyClobber());

      if (Register::isVirtualRegister(Reg)) {
        if (LiveInts->hasInterval(Reg)) {
          const LiveInterval &LI = LiveInts->getInterval(Reg);
          checkLivenessAtDef(MO, MONum, DefIdx, LI, Reg);

          if (LI.hasSubRanges()) {
            unsigned SubRegIdx = MO->getSubReg();
            LaneBitmask MOMask = SubRegIdx != 0
              ? TRI->getSubRegIndexLaneMask(SubRegIdx)
              : MRI->getMaxLaneMaskForVReg(Reg);
            for (const LiveInterval::SubRange &SR : LI.subranges()) {
              if ((SR.LaneMask & MOMask).none())
                continue;
              checkLivenessAtDef(MO, MONum, DefIdx, SR, Reg, true, SR.LaneMask);
            }
          }
        } else {
          report("Virtual register has no Live interval", MO, MONum);
        }
      }
    }
  }
}

// This function gets called after visiting all instructions in a bundle. The
// argument points to the bundle header.
// Normal stand-alone instructions are also considered 'bundles', and this
// function is called for all of them.
void MachineVerifier::visitMachineBundleAfter(const MachineInstr *MI) {
  BBInfo &MInfo = MBBInfoMap[MI->getParent()];
  set_union(MInfo.regsKilled, regsKilled);
  set_subtract(regsLive, regsKilled); regsKilled.clear();
  // Kill any masked registers.
  while (!regMasks.empty()) {
    const uint32_t *Mask = regMasks.pop_back_val();
    for (unsigned Reg : regsLive)
      if (Register::isPhysicalRegister(Reg) &&
          MachineOperand::clobbersPhysReg(Mask, Reg))
        regsDead.push_back(Reg);
  }
  set_subtract(regsLive, regsDead);   regsDead.clear();
  set_union(regsLive, regsDefined);   regsDefined.clear();
}

void
MachineVerifier::visitMachineBasicBlockAfter(const MachineBasicBlock *MBB) {
  MBBInfoMap[MBB].regsLiveOut = regsLive;
  regsLive.clear();

  if (Indexes) {
    SlotIndex stop = Indexes->getMBBEndIdx(MBB);
    if (!(stop > lastIndex)) {
      report("Block ends before last instruction index", MBB);
      errs() << "Block ends at " << stop
          << " last instruction was at " << lastIndex << '\n';
    }
    lastIndex = stop;
  }
}

namespace {
// This implements a set of registers that serves as a filter: can filter other
// sets by passing through elements not in the filter and blocking those that
// are. Any filter implicitly includes the full set of physical registers upon
// creation, thus filtering them all out. The filter itself as a set only grows,
// and needs to be as efficient as possible.
struct VRegFilter {
  // Add elements to the filter itself. \pre Input set \p FromRegSet must have
  // no duplicates. Both virtual and physical registers are fine.
  template <typename RegSetT> void add(const RegSetT &FromRegSet) {
    SmallVector<unsigned, 0> VRegsBuffer;
    filterAndAdd(FromRegSet, VRegsBuffer);
  }
  // Filter \p FromRegSet through the filter and append passed elements into \p
  // ToVRegs. All elements appended are then added to the filter itself.
  // \returns true if anything changed.
  template <typename RegSetT>
  bool filterAndAdd(const RegSetT &FromRegSet,
                    SmallVectorImpl<unsigned> &ToVRegs) {
    unsigned SparseUniverse = Sparse.size();
    unsigned NewSparseUniverse = SparseUniverse;
    unsigned NewDenseSize = Dense.size();
    size_t Begin = ToVRegs.size();
    for (unsigned Reg : FromRegSet) {
      if (!Register::isVirtualRegister(Reg))
        continue;
      unsigned Index = Register::virtReg2Index(Reg);
      if (Index < SparseUniverseMax) {
        if (Index < SparseUniverse && Sparse.test(Index))
          continue;
        NewSparseUniverse = std::max(NewSparseUniverse, Index + 1);
      } else {
        if (Dense.count(Reg))
          continue;
        ++NewDenseSize;
      }
      ToVRegs.push_back(Reg);
    }
    size_t End = ToVRegs.size();
    if (Begin == End)
      return false;
    // Reserving space in sets once performs better than doing so continuously
    // and pays easily for double look-ups (even in Dense with SparseUniverseMax
    // tuned all the way down) and double iteration (the second one is over a
    // SmallVector, which is a lot cheaper compared to DenseSet or BitVector).
    Sparse.resize(NewSparseUniverse);
    Dense.reserve(NewDenseSize);
    for (unsigned I = Begin; I < End; ++I) {
      unsigned Reg = ToVRegs[I];
      unsigned Index = Register::virtReg2Index(Reg);
      if (Index < SparseUniverseMax)
        Sparse.set(Index);
      else
        Dense.insert(Reg);
    }
    return true;
  }

private:
  static constexpr unsigned SparseUniverseMax = 10 * 1024 * 8;
  // VRegs indexed within SparseUniverseMax are tracked by Sparse, those beyound
  // are tracked by Dense. The only purpose of the threashold and the Dense set
  // is to have a reasonably growing memory usage in pathological cases (large
  // number of very sparse VRegFilter instances live at the same time). In
  // practice even in the worst-by-execution time cases having all elements
  // tracked by Sparse (very large SparseUniverseMax scenario) tends to be more
  // space efficient than if tracked by Dense. The threashold is set to keep the
  // worst-case memory usage within 2x of figures determined empirically for
  // "all Dense" scenario in such worst-by-execution-time cases.
  BitVector Sparse;
  DenseSet<unsigned> Dense;
};

// Implements both a transfer function and a (binary, in-place) join operator
// for a dataflow over register sets with set union join and filtering transfer
// (out_b = in_b \ filter_b). filter_b is expected to be set-up ahead of time.
// Maintains out_b as its state, allowing for O(n) iteration over it at any
// time, where n is the size of the set (as opposed to O(U) where U is the
// universe). filter_b implicitly contains all physical registers at all times.
class FilteringVRegSet {
  VRegFilter Filter;
  SmallVector<unsigned, 0> VRegs;

public:
  // Set-up the filter_b. \pre Input register set \p RS must have no duplicates.
  // Both virtual and physical registers are fine.
  template <typename RegSetT> void addToFilter(const RegSetT &RS) {
    Filter.add(RS);
  }
  // Passes \p RS through the filter_b (transfer function) and adds what's left
  // to itself (out_b).
  template <typename RegSetT> bool add(const RegSetT &RS) {
    // Double-duty the Filter: to maintain VRegs a set (and the join operation
    // a set union) just add everything being added here to the Filter as well.
    return Filter.filterAndAdd(RS, VRegs);
  }
  using const_iterator = decltype(VRegs)::const_iterator;
  const_iterator begin() const { return VRegs.begin(); }
  const_iterator end() const { return VRegs.end(); }
  size_t size() const { return VRegs.size(); }
};
} // namespace

// Calculate the largest possible vregsPassed sets. These are the registers that
// can pass through an MBB live, but may not be live every time. It is assumed
// that all vregsPassed sets are empty before the call.
void MachineVerifier::calcRegsPassed() {
  // This is a forward dataflow, doing it in RPO. A standard map serves as a
  // priority (sorting by RPO number) queue, deduplicating worklist, and an RPO
  // number to MBB mapping all at once.
  std::map<unsigned, const MachineBasicBlock *> RPOWorklist;
  DenseMap<const MachineBasicBlock *, unsigned> RPONumbers;
  if (MF->empty()) {
    // ReversePostOrderTraversal doesn't handle empty functions.
    return;
  }
  std::vector<FilteringVRegSet> VRegsPassedSets(MF->size());
  for (const MachineBasicBlock *MBB :
       ReversePostOrderTraversal<const MachineFunction *>(MF)) {
    // Careful with the evaluation order, fetch next number before allocating.
    unsigned Number = RPONumbers.size();
    RPONumbers[MBB] = Number;
    // Set-up the transfer functions for all blocks.
    const BBInfo &MInfo = MBBInfoMap[MBB];
    VRegsPassedSets[Number].addToFilter(MInfo.regsKilled);
    VRegsPassedSets[Number].addToFilter(MInfo.regsLiveOut);
  }
  // First push live-out regs to successors' vregsPassed. Remember the MBBs that
  // have any vregsPassed.
  for (const MachineBasicBlock &MBB : *MF) {
    const BBInfo &MInfo = MBBInfoMap[&MBB];
    if (!MInfo.reachable)
      continue;
    for (const MachineBasicBlock *Succ : MBB.successors()) {
      unsigned SuccNumber = RPONumbers[Succ];
      FilteringVRegSet &SuccSet = VRegsPassedSets[SuccNumber];
      if (SuccSet.add(MInfo.regsLiveOut))
        RPOWorklist.emplace(SuccNumber, Succ);
    }
  }

  // Iteratively push vregsPassed to successors.
  while (!RPOWorklist.empty()) {
    auto Next = RPOWorklist.begin();
    const MachineBasicBlock *MBB = Next->second;
    RPOWorklist.erase(Next);
    FilteringVRegSet &MSet = VRegsPassedSets[RPONumbers[MBB]];
    for (const MachineBasicBlock *Succ : MBB->successors()) {
      if (Succ == MBB)
        continue;
      unsigned SuccNumber = RPONumbers[Succ];
      FilteringVRegSet &SuccSet = VRegsPassedSets[SuccNumber];
      if (SuccSet.add(MSet))
        RPOWorklist.emplace(SuccNumber, Succ);
    }
  }
  // Copy the results back to BBInfos.
  for (const MachineBasicBlock &MBB : *MF) {
    BBInfo &MInfo = MBBInfoMap[&MBB];
    if (!MInfo.reachable)
      continue;
    const FilteringVRegSet &MSet = VRegsPassedSets[RPONumbers[&MBB]];
    MInfo.vregsPassed.reserve(MSet.size());
    MInfo.vregsPassed.insert(MSet.begin(), MSet.end());
  }
}

// Calculate the set of virtual registers that must be passed through each basic
// block in order to satisfy the requirements of successor blocks. This is very
// similar to calcRegsPassed, only backwards.
void MachineVerifier::calcRegsRequired() {
  // First push live-in regs to predecessors' vregsRequired.
  SmallPtrSet<const MachineBasicBlock*, 8> todo;
  for (const auto &MBB : *MF) {
    BBInfo &MInfo = MBBInfoMap[&MBB];
    for (const MachineBasicBlock *Pred : MBB.predecessors()) {
      BBInfo &PInfo = MBBInfoMap[Pred];
      if (PInfo.addRequired(MInfo.vregsLiveIn))
        todo.insert(Pred);
    }
  }

  // Iteratively push vregsRequired to predecessors. This will converge to the
  // same final state regardless of DenseSet iteration order.
  while (!todo.empty()) {
    const MachineBasicBlock *MBB = *todo.begin();
    todo.erase(MBB);
    BBInfo &MInfo = MBBInfoMap[MBB];
    for (const MachineBasicBlock *Pred : MBB->predecessors()) {
      if (Pred == MBB)
        continue;
      BBInfo &SInfo = MBBInfoMap[Pred];
      if (SInfo.addRequired(MInfo.vregsRequired))
        todo.insert(Pred);
    }
  }
}

// Check PHI instructions at the beginning of MBB. It is assumed that
// calcRegsPassed has been run so BBInfo::isLiveOut is valid.
void MachineVerifier::checkPHIOps(const MachineBasicBlock &MBB) {
  BBInfo &MInfo = MBBInfoMap[&MBB];

  SmallPtrSet<const MachineBasicBlock*, 8> seen;
  for (const MachineInstr &Phi : MBB) {
    if (!Phi.isPHI())
      break;
    seen.clear();

    const MachineOperand &MODef = Phi.getOperand(0);
    if (!MODef.isReg() || !MODef.isDef()) {
      report("Expected first PHI operand to be a register def", &MODef, 0);
      continue;
    }
    if (MODef.isTied() || MODef.isImplicit() || MODef.isInternalRead() ||
        MODef.isEarlyClobber() || MODef.isDebug())
      report("Unexpected flag on PHI operand", &MODef, 0);
    Register DefReg = MODef.getReg();
    if (!Register::isVirtualRegister(DefReg))
      report("Expected first PHI operand to be a virtual register", &MODef, 0);

    for (unsigned I = 1, E = Phi.getNumOperands(); I != E; I += 2) {
      const MachineOperand &MO0 = Phi.getOperand(I);
      if (!MO0.isReg()) {
        report("Expected PHI operand to be a register", &MO0, I);
        continue;
      }
      if (MO0.isImplicit() || MO0.isInternalRead() || MO0.isEarlyClobber() ||
          MO0.isDebug() || MO0.isTied())
        report("Unexpected flag on PHI operand", &MO0, I);

      const MachineOperand &MO1 = Phi.getOperand(I + 1);
      if (!MO1.isMBB()) {
        report("Expected PHI operand to be a basic block", &MO1, I + 1);
        continue;
      }

      const MachineBasicBlock &Pre = *MO1.getMBB();
      if (!Pre.isSuccessor(&MBB)) {
        report("PHI input is not a predecessor block", &MO1, I + 1);
        continue;
      }

      if (MInfo.reachable) {
        seen.insert(&Pre);
        BBInfo &PrInfo = MBBInfoMap[&Pre];
        if (!MO0.isUndef() && PrInfo.reachable &&
            !PrInfo.isLiveOut(MO0.getReg()))
          report("PHI operand is not live-out from predecessor", &MO0, I);
      }
    }

    // Did we see all predecessors?
    if (MInfo.reachable) {
      for (MachineBasicBlock *Pred : MBB.predecessors()) {
        if (!seen.count(Pred)) {
          report("Missing PHI operand", &Phi);
          errs() << printMBBReference(*Pred)
                 << " is a predecessor according to the CFG.\n";
        }
      }
    }
  }
}

void MachineVerifier::visitMachineFunctionAfter() {
  calcRegsPassed();

  for (const MachineBasicBlock &MBB : *MF)
    checkPHIOps(MBB);

  // Now check liveness info if available
  calcRegsRequired();

  // Check for killed virtual registers that should be live out.
  for (const auto &MBB : *MF) {
    BBInfo &MInfo = MBBInfoMap[&MBB];
    for (unsigned VReg : MInfo.vregsRequired)
      if (MInfo.regsKilled.count(VReg)) {
        report("Virtual register killed in block, but needed live out.", &MBB);
        errs() << "Virtual register " << printReg(VReg)
               << " is used after the block.\n";
      }
  }

  if (!MF->empty()) {
    BBInfo &MInfo = MBBInfoMap[&MF->front()];
    for (unsigned VReg : MInfo.vregsRequired) {
      report("Virtual register defs don't dominate all uses.", MF);
      report_context_vreg(VReg);
    }
  }

  if (LiveVars)
    verifyLiveVariables();
  if (LiveInts)
    verifyLiveIntervals();

  // Check live-in list of each MBB. If a register is live into MBB, check
  // that the register is in regsLiveOut of each predecessor block. Since
  // this must come from a definition in the predecesssor or its live-in
  // list, this will catch a live-through case where the predecessor does not
  // have the register in its live-in list.  This currently only checks
  // registers that have no aliases, are not allocatable and are not
  // reserved, which could mean a condition code register for instance.
  if (MRI->tracksLiveness())
    for (const auto &MBB : *MF)
      for (MachineBasicBlock::RegisterMaskPair P : MBB.liveins()) {
        MCPhysReg LiveInReg = P.PhysReg;
        bool hasAliases = MCRegAliasIterator(LiveInReg, TRI, false).isValid();
        if (hasAliases || isAllocatable(LiveInReg) || isReserved(LiveInReg))
          continue;
        for (const MachineBasicBlock *Pred : MBB.predecessors()) {
          BBInfo &PInfo = MBBInfoMap[Pred];
          if (!PInfo.regsLiveOut.count(LiveInReg)) {
            report("Live in register not found to be live out from predecessor.",
                   &MBB);
            errs() << TRI->getName(LiveInReg)
                   << " not found to be live out from "
                   << printMBBReference(*Pred) << "\n";
          }
        }
      }

  for (auto CSInfo : MF->getCallSitesInfo())
    if (!CSInfo.first->isCall())
      report("Call site info referencing instruction that is not call", MF);
}

void MachineVerifier::verifyLiveVariables() {
  assert(LiveVars && "Don't call verifyLiveVariables without LiveVars");
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = Register::index2VirtReg(i);
    LiveVariables::VarInfo &VI = LiveVars->getVarInfo(Reg);
    for (const auto &MBB : *MF) {
      BBInfo &MInfo = MBBInfoMap[&MBB];

      // Our vregsRequired should be identical to LiveVariables' AliveBlocks
      if (MInfo.vregsRequired.count(Reg)) {
        if (!VI.AliveBlocks.test(MBB.getNumber())) {
          report("LiveVariables: Block missing from AliveBlocks", &MBB);
          errs() << "Virtual register " << printReg(Reg)
                 << " must be live through the block.\n";
        }
      } else {
        if (VI.AliveBlocks.test(MBB.getNumber())) {
          report("LiveVariables: Block should not be in AliveBlocks", &MBB);
          errs() << "Virtual register " << printReg(Reg)
                 << " is not needed live through the block.\n";
        }
      }
    }
  }
}

void MachineVerifier::verifyLiveIntervals() {
  assert(LiveInts && "Don't call verifyLiveIntervals without LiveInts");
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = Register::index2VirtReg(i);

    // Spilling and splitting may leave unused registers around. Skip them.
    if (MRI->reg_nodbg_empty(Reg))
      continue;

    if (!LiveInts->hasInterval(Reg)) {
      report("Missing live interval for virtual register", MF);
      errs() << printReg(Reg, TRI) << " still has defs or uses\n";
      continue;
    }

    const LiveInterval &LI = LiveInts->getInterval(Reg);
    assert(Reg == LI.reg && "Invalid reg to interval mapping");
    verifyLiveInterval(LI);
  }

  // Verify all the cached regunit intervals.
  for (unsigned i = 0, e = TRI->getNumRegUnits(); i != e; ++i)
    if (const LiveRange *LR = LiveInts->getCachedRegUnit(i))
      verifyLiveRange(*LR, i);
}

void MachineVerifier::verifyLiveRangeValue(const LiveRange &LR,
                                           const VNInfo *VNI, unsigned Reg,
                                           LaneBitmask LaneMask) {
  if (VNI->isUnused())
    return;

  const VNInfo *DefVNI = LR.getVNInfoAt(VNI->def);

  if (!DefVNI) {
    report("Value not live at VNInfo def and not marked unused", MF);
    report_context(LR, Reg, LaneMask);
    report_context(*VNI);
    return;
  }

  if (DefVNI != VNI) {
    report("Live segment at def has different VNInfo", MF);
    report_context(LR, Reg, LaneMask);
    report_context(*VNI);
    return;
  }

  const MachineBasicBlock *MBB = LiveInts->getMBBFromIndex(VNI->def);
  if (!MBB) {
    report("Invalid VNInfo definition index", MF);
    report_context(LR, Reg, LaneMask);
    report_context(*VNI);
    return;
  }

  if (VNI->isPHIDef()) {
    if (VNI->def != LiveInts->getMBBStartIdx(MBB)) {
      report("PHIDef VNInfo is not defined at MBB start", MBB);
      report_context(LR, Reg, LaneMask);
      report_context(*VNI);
    }
    return;
  }

  // Non-PHI def.
  const MachineInstr *MI = LiveInts->getInstructionFromIndex(VNI->def);
  if (!MI) {
    report("No instruction at VNInfo def index", MBB);
    report_context(LR, Reg, LaneMask);
    report_context(*VNI);
    return;
  }

  if (Reg != 0) {
    bool hasDef = false;
    bool isEarlyClobber = false;
    for (ConstMIBundleOperands MOI(*MI); MOI.isValid(); ++MOI) {
      if (!MOI->isReg() || !MOI->isDef())
        continue;
      if (Register::isVirtualRegister(Reg)) {
        if (MOI->getReg() != Reg)
          continue;
      } else {
        if (!Register::isPhysicalRegister(MOI->getReg()) ||
            !TRI->hasRegUnit(MOI->getReg(), Reg))
          continue;
      }
      if (LaneMask.any() &&
          (TRI->getSubRegIndexLaneMask(MOI->getSubReg()) & LaneMask).none())
        continue;
      hasDef = true;
      if (MOI->isEarlyClobber())
        isEarlyClobber = true;
    }

    if (!hasDef) {
      report("Defining instruction does not modify register", MI);
      report_context(LR, Reg, LaneMask);
      report_context(*VNI);
    }

    // Early clobber defs begin at USE slots, but other defs must begin at
    // DEF slots.
    if (isEarlyClobber) {
      if (!VNI->def.isEarlyClobber()) {
        report("Early clobber def must be at an early-clobber slot", MBB);
        report_context(LR, Reg, LaneMask);
        report_context(*VNI);
      }
    } else if (!VNI->def.isRegister()) {
      report("Non-PHI, non-early clobber def must be at a register slot", MBB);
      report_context(LR, Reg, LaneMask);
      report_context(*VNI);
    }
  }
}

void MachineVerifier::verifyLiveRangeSegment(const LiveRange &LR,
                                             const LiveRange::const_iterator I,
                                             unsigned Reg, LaneBitmask LaneMask)
{
  const LiveRange::Segment &S = *I;
  const VNInfo *VNI = S.valno;
  assert(VNI && "Live segment has no valno");

  if (VNI->id >= LR.getNumValNums() || VNI != LR.getValNumInfo(VNI->id)) {
    report("Foreign valno in live segment", MF);
    report_context(LR, Reg, LaneMask);
    report_context(S);
    report_context(*VNI);
  }

  if (VNI->isUnused()) {
    report("Live segment valno is marked unused", MF);
    report_context(LR, Reg, LaneMask);
    report_context(S);
  }

  const MachineBasicBlock *MBB = LiveInts->getMBBFromIndex(S.start);
  if (!MBB) {
    report("Bad start of live segment, no basic block", MF);
    report_context(LR, Reg, LaneMask);
    report_context(S);
    return;
  }
  SlotIndex MBBStartIdx = LiveInts->getMBBStartIdx(MBB);
  if (S.start != MBBStartIdx && S.start != VNI->def) {
    report("Live segment must begin at MBB entry or valno def", MBB);
    report_context(LR, Reg, LaneMask);
    report_context(S);
  }

  const MachineBasicBlock *EndMBB =
    LiveInts->getMBBFromIndex(S.end.getPrevSlot());
  if (!EndMBB) {
    report("Bad end of live segment, no basic block", MF);
    report_context(LR, Reg, LaneMask);
    report_context(S);
    return;
  }

  // No more checks for live-out segments.
  if (S.end == LiveInts->getMBBEndIdx(EndMBB))
    return;

  // RegUnit intervals are allowed dead phis.
  if (!Register::isVirtualRegister(Reg) && VNI->isPHIDef() &&
      S.start == VNI->def && S.end == VNI->def.getDeadSlot())
    return;

  // The live segment is ending inside EndMBB
  const MachineInstr *MI =
    LiveInts->getInstructionFromIndex(S.end.getPrevSlot());
  if (!MI) {
    report("Live segment doesn't end at a valid instruction", EndMBB);
    report_context(LR, Reg, LaneMask);
    report_context(S);
    return;
  }

  // The block slot must refer to a basic block boundary.
  if (S.end.isBlock()) {
    report("Live segment ends at B slot of an instruction", EndMBB);
    report_context(LR, Reg, LaneMask);
    report_context(S);
  }

  if (S.end.isDead()) {
    // Segment ends on the dead slot.
    // That means there must be a dead def.
    if (!SlotIndex::isSameInstr(S.start, S.end)) {
      report("Live segment ending at dead slot spans instructions", EndMBB);
      report_context(LR, Reg, LaneMask);
      report_context(S);
    }
  }

  // A live segment can only end at an early-clobber slot if it is being
  // redefined by an early-clobber def.
  if (S.end.isEarlyClobber()) {
    if (I+1 == LR.end() || (I+1)->start != S.end) {
      report("Live segment ending at early clobber slot must be "
             "redefined by an EC def in the same instruction", EndMBB);
      report_context(LR, Reg, LaneMask);
      report_context(S);
    }
  }

  // The following checks only apply to virtual registers. Physreg liveness
  // is too weird to check.
  if (Register::isVirtualRegister(Reg)) {
    // A live segment can end with either a redefinition, a kill flag on a
    // use, or a dead flag on a def.
    bool hasRead = false;
    bool hasSubRegDef = false;
    bool hasDeadDef = false;
    for (ConstMIBundleOperands MOI(*MI); MOI.isValid(); ++MOI) {
      if (!MOI->isReg() || MOI->getReg() != Reg)
        continue;
      unsigned Sub = MOI->getSubReg();
      LaneBitmask SLM = Sub != 0 ? TRI->getSubRegIndexLaneMask(Sub)
                                 : LaneBitmask::getAll();
      if (MOI->isDef()) {
        if (Sub != 0) {
          hasSubRegDef = true;
          // An operand %0:sub0 reads %0:sub1..n. Invert the lane
          // mask for subregister defs. Read-undef defs will be handled by
          // readsReg below.
          SLM = ~SLM;
        }
        if (MOI->isDead())
          hasDeadDef = true;
      }
      if (LaneMask.any() && (LaneMask & SLM).none())
        continue;
      if (MOI->readsReg())
        hasRead = true;
    }
    if (S.end.isDead()) {
      // Make sure that the corresponding machine operand for a "dead" live
      // range has the dead flag. We cannot perform this check for subregister
      // liveranges as partially dead values are allowed.
      if (LaneMask.none() && !hasDeadDef) {
        report("Instruction ending live segment on dead slot has no dead flag",
               MI);
        report_context(LR, Reg, LaneMask);
        report_context(S);
      }
    } else {
      if (!hasRead) {
        // When tracking subregister liveness, the main range must start new
        // values on partial register writes, even if there is no read.
        if (!MRI->shouldTrackSubRegLiveness(Reg) || LaneMask.any() ||
            !hasSubRegDef) {
          report("Instruction ending live segment doesn't read the register",
                 MI);
          report_context(LR, Reg, LaneMask);
          report_context(S);
        }
      }
    }
  }

  // Now check all the basic blocks in this live segment.
  MachineFunction::const_iterator MFI = MBB->getIterator();
  // Is this live segment the beginning of a non-PHIDef VN?
  if (S.start == VNI->def && !VNI->isPHIDef()) {
    // Not live-in to any blocks.
    if (MBB == EndMBB)
      return;
    // Skip this block.
    ++MFI;
  }

  SmallVector<SlotIndex, 4> Undefs;
  if (LaneMask.any()) {
    LiveInterval &OwnerLI = LiveInts->getInterval(Reg);
    OwnerLI.computeSubRangeUndefs(Undefs, LaneMask, *MRI, *Indexes);
  }

  while (true) {
    assert(LiveInts->isLiveInToMBB(LR, &*MFI));
    // We don't know how to track physregs into a landing pad.
    if (!Register::isVirtualRegister(Reg) && MFI->isEHPad()) {
      if (&*MFI == EndMBB)
        break;
      ++MFI;
      continue;
    }

    // Is VNI a PHI-def in the current block?
    bool IsPHI = VNI->isPHIDef() &&
      VNI->def == LiveInts->getMBBStartIdx(&*MFI);

    // Check that VNI is live-out of all predecessors.
    for (const MachineBasicBlock *Pred : MFI->predecessors()) {
      SlotIndex PEnd = LiveInts->getMBBEndIdx(Pred);
      const VNInfo *PVNI = LR.getVNInfoBefore(PEnd);

      // All predecessors must have a live-out value. However for a phi
      // instruction with subregister intervals
      // only one of the subregisters (not necessarily the current one) needs to
      // be defined.
      if (!PVNI && (LaneMask.none() || !IsPHI)) {
        if (LiveRangeCalc::isJointlyDominated(Pred, Undefs, *Indexes))
          continue;
        report("Register not marked live out of predecessor", Pred);
        report_context(LR, Reg, LaneMask);
        report_context(*VNI);
        errs() << " live into " << printMBBReference(*MFI) << '@'
               << LiveInts->getMBBStartIdx(&*MFI) << ", not live before "
               << PEnd << '\n';
        continue;
      }

      // Only PHI-defs can take different predecessor values.
      if (!IsPHI && PVNI != VNI) {
        report("Different value live out of predecessor", Pred);
        report_context(LR, Reg, LaneMask);
        errs() << "Valno #" << PVNI->id << " live out of "
               << printMBBReference(*Pred) << '@' << PEnd << "\nValno #"
               << VNI->id << " live into " << printMBBReference(*MFI) << '@'
               << LiveInts->getMBBStartIdx(&*MFI) << '\n';
      }
    }
    if (&*MFI == EndMBB)
      break;
    ++MFI;
  }
}

void MachineVerifier::verifyLiveRange(const LiveRange &LR, unsigned Reg,
                                      LaneBitmask LaneMask) {
  for (const VNInfo *VNI : LR.valnos)
    verifyLiveRangeValue(LR, VNI, Reg, LaneMask);

  for (LiveRange::const_iterator I = LR.begin(), E = LR.end(); I != E; ++I)
    verifyLiveRangeSegment(LR, I, Reg, LaneMask);
}

void MachineVerifier::verifyLiveInterval(const LiveInterval &LI) {
  unsigned Reg = LI.reg;
  assert(Register::isVirtualRegister(Reg));
  verifyLiveRange(LI, Reg);

  LaneBitmask Mask;
  LaneBitmask MaxMask = MRI->getMaxLaneMaskForVReg(Reg);
  for (const LiveInterval::SubRange &SR : LI.subranges()) {
    if ((Mask & SR.LaneMask).any()) {
      report("Lane masks of sub ranges overlap in live interval", MF);
      report_context(LI);
    }
    if ((SR.LaneMask & ~MaxMask).any()) {
      report("Subrange lanemask is invalid", MF);
      report_context(LI);
    }
    if (SR.empty()) {
      report("Subrange must not be empty", MF);
      report_context(SR, LI.reg, SR.LaneMask);
    }
    Mask |= SR.LaneMask;
    verifyLiveRange(SR, LI.reg, SR.LaneMask);
    if (!LI.covers(SR)) {
      report("A Subrange is not covered by the main range", MF);
      report_context(LI);
    }
  }

  // Check the LI only has one connected component.
  ConnectedVNInfoEqClasses ConEQ(*LiveInts);
  unsigned NumComp = ConEQ.Classify(LI);
  if (NumComp > 1) {
    report("Multiple connected components in live interval", MF);
    report_context(LI);
    for (unsigned comp = 0; comp != NumComp; ++comp) {
      errs() << comp << ": valnos";
      for (const VNInfo *I : LI.valnos)
        if (comp == ConEQ.getEqClass(I))
          errs() << ' ' << I->id;
      errs() << '\n';
    }
  }
}

namespace {

  // FrameSetup and FrameDestroy can have zero adjustment, so using a single
  // integer, we can't tell whether it is a FrameSetup or FrameDestroy if the
  // value is zero.
  // We use a bool plus an integer to capture the stack state.
  struct StackStateOfBB {
    StackStateOfBB() = default;
    StackStateOfBB(int EntryVal, int ExitVal, bool EntrySetup, bool ExitSetup) :
      EntryValue(EntryVal), ExitValue(ExitVal), EntryIsSetup(EntrySetup),
      ExitIsSetup(ExitSetup) {}

    // Can be negative, which means we are setting up a frame.
    int EntryValue = 0;
    int ExitValue = 0;
    bool EntryIsSetup = false;
    bool ExitIsSetup = false;
  };

} // end anonymous namespace

/// Make sure on every path through the CFG, a FrameSetup <n> is always followed
/// by a FrameDestroy <n>, stack adjustments are identical on all
/// CFG edges to a merge point, and frame is destroyed at end of a return block.
void MachineVerifier::verifyStackFrame() {
  unsigned FrameSetupOpcode   = TII->getCallFrameSetupOpcode();
  unsigned FrameDestroyOpcode = TII->getCallFrameDestroyOpcode();
  if (FrameSetupOpcode == ~0u && FrameDestroyOpcode == ~0u)
    return;

  SmallVector<StackStateOfBB, 8> SPState;
  SPState.resize(MF->getNumBlockIDs());
  df_iterator_default_set<const MachineBasicBlock*> Reachable;

  // Visit the MBBs in DFS order.
  for (df_ext_iterator<const MachineFunction *,
                       df_iterator_default_set<const MachineBasicBlock *>>
       DFI = df_ext_begin(MF, Reachable), DFE = df_ext_end(MF, Reachable);
       DFI != DFE; ++DFI) {
    const MachineBasicBlock *MBB = *DFI;

    StackStateOfBB BBState;
    // Check the exit state of the DFS stack predecessor.
    if (DFI.getPathLength() >= 2) {
      const MachineBasicBlock *StackPred = DFI.getPath(DFI.getPathLength() - 2);
      assert(Reachable.count(StackPred) &&
             "DFS stack predecessor is already visited.\n");
      BBState.EntryValue = SPState[StackPred->getNumber()].ExitValue;
      BBState.EntryIsSetup = SPState[StackPred->getNumber()].ExitIsSetup;
      BBState.ExitValue = BBState.EntryValue;
      BBState.ExitIsSetup = BBState.EntryIsSetup;
    }

    // Update stack state by checking contents of MBB.
    for (const auto &I : *MBB) {
      if (I.getOpcode() == FrameSetupOpcode) {
        if (BBState.ExitIsSetup)
          report("FrameSetup is after another FrameSetup", &I);
        BBState.ExitValue -= TII->getFrameTotalSize(I);
        BBState.ExitIsSetup = true;
      }

      if (I.getOpcode() == FrameDestroyOpcode) {
        int Size = TII->getFrameTotalSize(I);
        if (!BBState.ExitIsSetup)
          report("FrameDestroy is not after a FrameSetup", &I);
        int AbsSPAdj = BBState.ExitValue < 0 ? -BBState.ExitValue :
                                               BBState.ExitValue;
        if (BBState.ExitIsSetup && AbsSPAdj != Size) {
          report("FrameDestroy <n> is after FrameSetup <m>", &I);
          errs() << "FrameDestroy <" << Size << "> is after FrameSetup <"
              << AbsSPAdj << ">.\n";
        }
        BBState.ExitValue += Size;
        BBState.ExitIsSetup = false;
      }
    }
    SPState[MBB->getNumber()] = BBState;

    // Make sure the exit state of any predecessor is consistent with the entry
    // state.
    for (const MachineBasicBlock *Pred : MBB->predecessors()) {
      if (Reachable.count(Pred) &&
          (SPState[Pred->getNumber()].ExitValue != BBState.EntryValue ||
           SPState[Pred->getNumber()].ExitIsSetup != BBState.EntryIsSetup)) {
        report("The exit stack state of a predecessor is inconsistent.", MBB);
        errs() << "Predecessor " << printMBBReference(*Pred)
               << " has exit state (" << SPState[Pred->getNumber()].ExitValue
               << ", " << SPState[Pred->getNumber()].ExitIsSetup << "), while "
               << printMBBReference(*MBB) << " has entry state ("
               << BBState.EntryValue << ", " << BBState.EntryIsSetup << ").\n";
      }
    }

    // Make sure the entry state of any successor is consistent with the exit
    // state.
    for (const MachineBasicBlock *Succ : MBB->successors()) {
      if (Reachable.count(Succ) &&
          (SPState[Succ->getNumber()].EntryValue != BBState.ExitValue ||
           SPState[Succ->getNumber()].EntryIsSetup != BBState.ExitIsSetup)) {
        report("The entry stack state of a successor is inconsistent.", MBB);
        errs() << "Successor " << printMBBReference(*Succ)
               << " has entry state (" << SPState[Succ->getNumber()].EntryValue
               << ", " << SPState[Succ->getNumber()].EntryIsSetup << "), while "
               << printMBBReference(*MBB) << " has exit state ("
               << BBState.ExitValue << ", " << BBState.ExitIsSetup << ").\n";
      }
    }

    // Make sure a basic block with return ends with zero stack adjustment.
    if (!MBB->empty() && MBB->back().isReturn()) {
      if (BBState.ExitIsSetup)
        report("A return block ends with a FrameSetup.", MBB);
      if (BBState.ExitValue)
        report("A return block ends with a nonzero stack adjustment.", MBB);
    }
  }
}
