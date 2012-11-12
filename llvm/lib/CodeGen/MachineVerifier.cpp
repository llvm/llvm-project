//===-- MachineVerifier.cpp - Machine Code Verifier -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
// The machine code verifier is enabled from LLVMTargetMachine.cpp with the
// command-line option -verify-machineinstrs, or by defining the environment
// variable LLVM_VERIFY_MACHINEINSTRS to the name of a file that will receive
// the verifier errors.
//===----------------------------------------------------------------------===//

#include "llvm/BasicBlock.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  struct MachineVerifier {

    MachineVerifier(Pass *pass, const char *b) :
      PASS(pass),
      Banner(b),
      OutFileName(getenv("LLVM_VERIFY_MACHINEINSTRS"))
      {}

    bool runOnMachineFunction(MachineFunction &MF);

    Pass *const PASS;
    const char *Banner;
    const char *const OutFileName;
    raw_ostream *OS;
    const MachineFunction *MF;
    const TargetMachine *TM;
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    const MachineRegisterInfo *MRI;

    unsigned foundErrors;

    typedef SmallVector<unsigned, 16> RegVector;
    typedef SmallVector<const uint32_t*, 4> RegMaskVector;
    typedef DenseSet<unsigned> RegSet;
    typedef DenseMap<unsigned, const MachineInstr*> RegMap;
    typedef SmallPtrSet<const MachineBasicBlock*, 8> BlockSet;

    const MachineInstr *FirstTerminator;
    BlockSet FunctionBlocks;

    BitVector regsReserved;
    RegSet regsLive;
    RegVector regsDefined, regsDead, regsKilled;
    RegMaskVector regMasks;
    RegSet regsLiveInButUnused;

    SlotIndex lastIndex;

    // Add Reg and any sub-registers to RV
    void addRegWithSubRegs(RegVector &RV, unsigned Reg) {
      RV.push_back(Reg);
      if (TargetRegisterInfo::isPhysicalRegister(Reg))
        for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
          RV.push_back(*SubRegs);
    }

    struct BBInfo {
      // Is this MBB reachable from the MF entry point?
      bool reachable;

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

      BBInfo() : reachable(false) {}

      // Add register to vregsPassed if it belongs there. Return true if
      // anything changed.
      bool addPassed(unsigned Reg) {
        if (!TargetRegisterInfo::isVirtualRegister(Reg))
          return false;
        if (regsKilled.count(Reg) || regsLiveOut.count(Reg))
          return false;
        return vregsPassed.insert(Reg).second;
      }

      // Same for a full set.
      bool addPassed(const RegSet &RS) {
        bool changed = false;
        for (RegSet::const_iterator I = RS.begin(), E = RS.end(); I != E; ++I)
          if (addPassed(*I))
            changed = true;
        return changed;
      }

      // Add register to vregsRequired if it belongs there. Return true if
      // anything changed.
      bool addRequired(unsigned Reg) {
        if (!TargetRegisterInfo::isVirtualRegister(Reg))
          return false;
        if (regsLiveOut.count(Reg))
          return false;
        return vregsRequired.insert(Reg).second;
      }

      // Same for a full set.
      bool addRequired(const RegSet &RS) {
        bool changed = false;
        for (RegSet::const_iterator I = RS.begin(), E = RS.end(); I != E; ++I)
          if (addRequired(*I))
            changed = true;
        return changed;
      }

      // Same for a full map.
      bool addRequired(const RegMap &RM) {
        bool changed = false;
        for (RegMap::const_iterator I = RM.begin(), E = RM.end(); I != E; ++I)
          if (addRequired(I->first))
            changed = true;
        return changed;
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

    bool isAllocatable(unsigned Reg) {
      return Reg < TRI->getNumRegs() && MRI->isAllocatable(Reg);
    }

    // Analysis information if available
    LiveVariables *LiveVars;
    LiveIntervals *LiveInts;
    LiveStacks *LiveStks;
    SlotIndexes *Indexes;

    void visitMachineFunctionBefore();
    void visitMachineBasicBlockBefore(const MachineBasicBlock *MBB);
    void visitMachineBundleBefore(const MachineInstr *MI);
    void visitMachineInstrBefore(const MachineInstr *MI);
    void visitMachineOperand(const MachineOperand *MO, unsigned MONum);
    void visitMachineInstrAfter(const MachineInstr *MI);
    void visitMachineBundleAfter(const MachineInstr *MI);
    void visitMachineBasicBlockAfter(const MachineBasicBlock *MBB);
    void visitMachineFunctionAfter();

    void report(const char *msg, const MachineFunction *MF);
    void report(const char *msg, const MachineBasicBlock *MBB);
    void report(const char *msg, const MachineInstr *MI);
    void report(const char *msg, const MachineOperand *MO, unsigned MONum);
    void report(const char *msg, const MachineFunction *MF,
                const LiveInterval &LI);
    void report(const char *msg, const MachineBasicBlock *MBB,
                const LiveInterval &LI);

    void verifyInlineAsm(const MachineInstr *MI);

    void checkLiveness(const MachineOperand *MO, unsigned MONum);
    void markReachable(const MachineBasicBlock *MBB);
    void calcRegsPassed();
    void checkPHIOps(const MachineBasicBlock *MBB);

    void calcRegsRequired();
    void verifyLiveVariables();
    void verifyLiveIntervals();
    void verifyLiveInterval(const LiveInterval&);
    void verifyLiveIntervalValue(const LiveInterval&, VNInfo*);
    void verifyLiveIntervalSegment(const LiveInterval&,
                                   LiveInterval::const_iterator);
  };

  struct MachineVerifierPass : public MachineFunctionPass {
    static char ID; // Pass ID, replacement for typeid
    const char *const Banner;

    MachineVerifierPass(const char *b = 0)
      : MachineFunctionPass(ID), Banner(b) {
        initializeMachineVerifierPassPass(*PassRegistry::getPassRegistry());
      }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) {
      MF.verify(this, Banner);
      return false;
    }
  };

}

char MachineVerifierPass::ID = 0;
INITIALIZE_PASS(MachineVerifierPass, "machineverifier",
                "Verify generated machine code", false, false)

FunctionPass *llvm::createMachineVerifierPass(const char *Banner) {
  return new MachineVerifierPass(Banner);
}

void MachineFunction::verify(Pass *p, const char *Banner) const {
  MachineVerifier(p, Banner)
    .runOnMachineFunction(const_cast<MachineFunction&>(*this));
}

bool MachineVerifier::runOnMachineFunction(MachineFunction &MF) {
  raw_ostream *OutFile = 0;
  if (OutFileName) {
    std::string ErrorInfo;
    OutFile = new raw_fd_ostream(OutFileName, ErrorInfo,
                                 raw_fd_ostream::F_Append);
    if (!ErrorInfo.empty()) {
      errs() << "Error opening '" << OutFileName << "': " << ErrorInfo << '\n';
      exit(1);
    }

    OS = OutFile;
  } else {
    OS = &errs();
  }

  foundErrors = 0;

  this->MF = &MF;
  TM = &MF.getTarget();
  TII = TM->getInstrInfo();
  TRI = TM->getRegisterInfo();
  MRI = &MF.getRegInfo();

  LiveVars = NULL;
  LiveInts = NULL;
  LiveStks = NULL;
  Indexes = NULL;
  if (PASS) {
    LiveInts = PASS->getAnalysisIfAvailable<LiveIntervals>();
    // We don't want to verify LiveVariables if LiveIntervals is available.
    if (!LiveInts)
      LiveVars = PASS->getAnalysisIfAvailable<LiveVariables>();
    LiveStks = PASS->getAnalysisIfAvailable<LiveStacks>();
    Indexes = PASS->getAnalysisIfAvailable<SlotIndexes>();
  }

  visitMachineFunctionBefore();
  for (MachineFunction::const_iterator MFI = MF.begin(), MFE = MF.end();
       MFI!=MFE; ++MFI) {
    visitMachineBasicBlockBefore(MFI);
    // Keep track of the current bundle header.
    const MachineInstr *CurBundle = 0;
    for (MachineBasicBlock::const_instr_iterator MBBI = MFI->instr_begin(),
           MBBE = MFI->instr_end(); MBBI != MBBE; ++MBBI) {
      if (MBBI->getParent() != MFI) {
        report("Bad instruction parent pointer", MFI);
        *OS << "Instruction: " << *MBBI;
        continue;
      }
      // Is this a bundle header?
      if (!MBBI->isInsideBundle()) {
        if (CurBundle)
          visitMachineBundleAfter(CurBundle);
        CurBundle = MBBI;
        visitMachineBundleBefore(CurBundle);
      } else if (!CurBundle)
        report("No bundle header", MBBI);
      visitMachineInstrBefore(MBBI);
      for (unsigned I = 0, E = MBBI->getNumOperands(); I != E; ++I)
        visitMachineOperand(&MBBI->getOperand(I), I);
      visitMachineInstrAfter(MBBI);
    }
    if (CurBundle)
      visitMachineBundleAfter(CurBundle);
    visitMachineBasicBlockAfter(MFI);
  }
  visitMachineFunctionAfter();

  if (OutFile)
    delete OutFile;
  else if (foundErrors)
    report_fatal_error("Found "+Twine(foundErrors)+" machine code errors.");

  // Clean up.
  regsLive.clear();
  regsDefined.clear();
  regsDead.clear();
  regsKilled.clear();
  regMasks.clear();
  regsLiveInButUnused.clear();
  MBBInfoMap.clear();

  return false;                 // no changes
}

void MachineVerifier::report(const char *msg, const MachineFunction *MF) {
  assert(MF);
  *OS << '\n';
  if (!foundErrors++) {
    if (Banner)
      *OS << "# " << Banner << '\n';
    MF->print(*OS, Indexes);
  }
  *OS << "*** Bad machine code: " << msg << " ***\n"
      << "- function:    " << MF->getName() << "\n";
}

void MachineVerifier::report(const char *msg, const MachineBasicBlock *MBB) {
  assert(MBB);
  report(msg, MBB->getParent());
  *OS << "- basic block: BB#" << MBB->getNumber()
      << ' ' << MBB->getName()
      << " (" << (const void*)MBB << ')';
  if (Indexes)
    *OS << " [" << Indexes->getMBBStartIdx(MBB)
        << ';' <<  Indexes->getMBBEndIdx(MBB) << ')';
  *OS << '\n';
}

void MachineVerifier::report(const char *msg, const MachineInstr *MI) {
  assert(MI);
  report(msg, MI->getParent());
  *OS << "- instruction: ";
  if (Indexes && Indexes->hasIndex(MI))
    *OS << Indexes->getInstructionIndex(MI) << '\t';
  MI->print(*OS, TM);
}

void MachineVerifier::report(const char *msg,
                             const MachineOperand *MO, unsigned MONum) {
  assert(MO);
  report(msg, MO->getParent());
  *OS << "- operand " << MONum << ":   ";
  MO->print(*OS, TM);
  *OS << "\n";
}

void MachineVerifier::report(const char *msg, const MachineFunction *MF,
                             const LiveInterval &LI) {
  report(msg, MF);
  *OS << "- interval:    ";
  if (TargetRegisterInfo::isVirtualRegister(LI.reg))
    *OS << PrintReg(LI.reg, TRI);
  else
    *OS << PrintRegUnit(LI.reg, TRI);
  *OS << ' ' << LI << '\n';
}

void MachineVerifier::report(const char *msg, const MachineBasicBlock *MBB,
                             const LiveInterval &LI) {
  report(msg, MBB);
  *OS << "- interval:    ";
  if (TargetRegisterInfo::isVirtualRegister(LI.reg))
    *OS << PrintReg(LI.reg, TRI);
  else
    *OS << PrintRegUnit(LI.reg, TRI);
  *OS << ' ' << LI << '\n';
}

void MachineVerifier::markReachable(const MachineBasicBlock *MBB) {
  BBInfo &MInfo = MBBInfoMap[MBB];
  if (!MInfo.reachable) {
    MInfo.reachable = true;
    for (MachineBasicBlock::const_succ_iterator SuI = MBB->succ_begin(),
           SuE = MBB->succ_end(); SuI != SuE; ++SuI)
      markReachable(*SuI);
  }
}

void MachineVerifier::visitMachineFunctionBefore() {
  lastIndex = SlotIndex();
  regsReserved = MRI->getReservedRegs();

  // A sub-register of a reserved register is also reserved
  for (int Reg = regsReserved.find_first(); Reg>=0;
       Reg = regsReserved.find_next(Reg)) {
    for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs) {
      // FIXME: This should probably be:
      // assert(regsReserved.test(*SubRegs) && "Non-reserved sub-register");
      regsReserved.set(*SubRegs);
    }
  }

  markReachable(&MF->front());

  // Build a set of the basic blocks in the function.
  FunctionBlocks.clear();
  for (MachineFunction::const_iterator
       I = MF->begin(), E = MF->end(); I != E; ++I) {
    FunctionBlocks.insert(I);
    BBInfo &MInfo = MBBInfoMap[I];

    MInfo.Preds.insert(I->pred_begin(), I->pred_end());
    if (MInfo.Preds.size() != I->pred_size())
      report("MBB has duplicate entries in its predecessor list.", I);

    MInfo.Succs.insert(I->succ_begin(), I->succ_end());
    if (MInfo.Succs.size() != I->succ_size())
      report("MBB has duplicate entries in its successor list.", I);
  }
}

// Does iterator point to a and b as the first two elements?
static bool matchPair(MachineBasicBlock::const_succ_iterator i,
                      const MachineBasicBlock *a, const MachineBasicBlock *b) {
  if (*i == a)
    return *++i == b;
  if (*i == b)
    return *++i == a;
  return false;
}

void
MachineVerifier::visitMachineBasicBlockBefore(const MachineBasicBlock *MBB) {
  FirstTerminator = 0;

  if (MRI->isSSA()) {
    // If this block has allocatable physical registers live-in, check that
    // it is an entry block or landing pad.
    for (MachineBasicBlock::livein_iterator LI = MBB->livein_begin(),
           LE = MBB->livein_end();
         LI != LE; ++LI) {
      unsigned reg = *LI;
      if (isAllocatable(reg) && !MBB->isLandingPad() &&
          MBB != MBB->getParent()->begin()) {
        report("MBB has allocable live-in, but isn't entry or landing-pad.", MBB);
      }
    }
  }

  // Count the number of landing pad successors.
  SmallPtrSet<MachineBasicBlock*, 4> LandingPadSuccs;
  for (MachineBasicBlock::const_succ_iterator I = MBB->succ_begin(),
       E = MBB->succ_end(); I != E; ++I) {
    if ((*I)->isLandingPad())
      LandingPadSuccs.insert(*I);
    if (!FunctionBlocks.count(*I))
      report("MBB has successor that isn't part of the function.", MBB);
    if (!MBBInfoMap[*I].Preds.count(MBB)) {
      report("Inconsistent CFG", MBB);
      *OS << "MBB is not in the predecessor list of the successor BB#"
          << (*I)->getNumber() << ".\n";
    }
  }

  // Check the predecessor list.
  for (MachineBasicBlock::const_pred_iterator I = MBB->pred_begin(),
       E = MBB->pred_end(); I != E; ++I) {
    if (!FunctionBlocks.count(*I))
      report("MBB has predecessor that isn't part of the function.", MBB);
    if (!MBBInfoMap[*I].Succs.count(MBB)) {
      report("Inconsistent CFG", MBB);
      *OS << "MBB is not in the successor list of the predecessor BB#"
          << (*I)->getNumber() << ".\n";
    }
  }

  const MCAsmInfo *AsmInfo = TM->getMCAsmInfo();
  const BasicBlock *BB = MBB->getBasicBlock();
  if (LandingPadSuccs.size() > 1 &&
      !(AsmInfo &&
        AsmInfo->getExceptionHandlingType() == ExceptionHandling::SjLj &&
        BB && isa<SwitchInst>(BB->getTerminator())))
    report("MBB has more than one landing pad successor", MBB);

  // Call AnalyzeBranch. If it succeeds, there several more conditions to check.
  MachineBasicBlock *TBB = 0, *FBB = 0;
  SmallVector<MachineOperand, 4> Cond;
  if (!TII->AnalyzeBranch(*const_cast<MachineBasicBlock *>(MBB),
                          TBB, FBB, Cond)) {
    // Ok, AnalyzeBranch thinks it knows what's going on with this block. Let's
    // check whether its answers match up with reality.
    if (!TBB && !FBB) {
      // Block falls through to its successor.
      MachineFunction::const_iterator MBBI = MBB;
      ++MBBI;
      if (MBBI == MF->end()) {
        // It's possible that the block legitimately ends with a noreturn
        // call or an unreachable, in which case it won't actually fall
        // out the bottom of the function.
      } else if (MBB->succ_size() == LandingPadSuccs.size()) {
        // It's possible that the block legitimately ends with a noreturn
        // call or an unreachable, in which case it won't actuall fall
        // out of the block.
      } else if (MBB->succ_size() != 1+LandingPadSuccs.size()) {
        report("MBB exits via unconditional fall-through but doesn't have "
               "exactly one CFG successor!", MBB);
      } else if (!MBB->isSuccessor(MBBI)) {
        report("MBB exits via unconditional fall-through but its successor "
               "differs from its CFG successor!", MBB);
      }
      if (!MBB->empty() && getBundleStart(&MBB->back())->isBarrier() &&
          !TII->isPredicated(getBundleStart(&MBB->back()))) {
        report("MBB exits via unconditional fall-through but ends with a "
               "barrier instruction!", MBB);
      }
      if (!Cond.empty()) {
        report("MBB exits via unconditional fall-through but has a condition!",
               MBB);
      }
    } else if (TBB && !FBB && Cond.empty()) {
      // Block unconditionally branches somewhere.
      if (MBB->succ_size() != 1+LandingPadSuccs.size()) {
        report("MBB exits via unconditional branch but doesn't have "
               "exactly one CFG successor!", MBB);
      } else if (!MBB->isSuccessor(TBB)) {
        report("MBB exits via unconditional branch but the CFG "
               "successor doesn't match the actual successor!", MBB);
      }
      if (MBB->empty()) {
        report("MBB exits via unconditional branch but doesn't contain "
               "any instructions!", MBB);
      } else if (!getBundleStart(&MBB->back())->isBarrier()) {
        report("MBB exits via unconditional branch but doesn't end with a "
               "barrier instruction!", MBB);
      } else if (!getBundleStart(&MBB->back())->isTerminator()) {
        report("MBB exits via unconditional branch but the branch isn't a "
               "terminator instruction!", MBB);
      }
    } else if (TBB && !FBB && !Cond.empty()) {
      // Block conditionally branches somewhere, otherwise falls through.
      MachineFunction::const_iterator MBBI = MBB;
      ++MBBI;
      if (MBBI == MF->end()) {
        report("MBB conditionally falls through out of function!", MBB);
      } if (MBB->succ_size() == 1) {
        // A conditional branch with only one successor is weird, but allowed.
        if (&*MBBI != TBB)
          report("MBB exits via conditional branch/fall-through but only has "
                 "one CFG successor!", MBB);
        else if (TBB != *MBB->succ_begin())
          report("MBB exits via conditional branch/fall-through but the CFG "
                 "successor don't match the actual successor!", MBB);
      } else if (MBB->succ_size() != 2) {
        report("MBB exits via conditional branch/fall-through but doesn't have "
               "exactly two CFG successors!", MBB);
      } else if (!matchPair(MBB->succ_begin(), TBB, MBBI)) {
        report("MBB exits via conditional branch/fall-through but the CFG "
               "successors don't match the actual successors!", MBB);
      }
      if (MBB->empty()) {
        report("MBB exits via conditional branch/fall-through but doesn't "
               "contain any instructions!", MBB);
      } else if (getBundleStart(&MBB->back())->isBarrier()) {
        report("MBB exits via conditional branch/fall-through but ends with a "
               "barrier instruction!", MBB);
      } else if (!getBundleStart(&MBB->back())->isTerminator()) {
        report("MBB exits via conditional branch/fall-through but the branch "
               "isn't a terminator instruction!", MBB);
      }
    } else if (TBB && FBB) {
      // Block conditionally branches somewhere, otherwise branches
      // somewhere else.
      if (MBB->succ_size() == 1) {
        // A conditional branch with only one successor is weird, but allowed.
        if (FBB != TBB)
          report("MBB exits via conditional branch/branch through but only has "
                 "one CFG successor!", MBB);
        else if (TBB != *MBB->succ_begin())
          report("MBB exits via conditional branch/branch through but the CFG "
                 "successor don't match the actual successor!", MBB);
      } else if (MBB->succ_size() != 2) {
        report("MBB exits via conditional branch/branch but doesn't have "
               "exactly two CFG successors!", MBB);
      } else if (!matchPair(MBB->succ_begin(), TBB, FBB)) {
        report("MBB exits via conditional branch/branch but the CFG "
               "successors don't match the actual successors!", MBB);
      }
      if (MBB->empty()) {
        report("MBB exits via conditional branch/branch but doesn't "
               "contain any instructions!", MBB);
      } else if (!getBundleStart(&MBB->back())->isBarrier()) {
        report("MBB exits via conditional branch/branch but doesn't end with a "
               "barrier instruction!", MBB);
      } else if (!getBundleStart(&MBB->back())->isTerminator()) {
        report("MBB exits via conditional branch/branch but the branch "
               "isn't a terminator instruction!", MBB);
      }
      if (Cond.empty()) {
        report("MBB exits via conditinal branch/branch but there's no "
               "condition!", MBB);
      }
    } else {
      report("AnalyzeBranch returned invalid data!", MBB);
    }
  }

  regsLive.clear();
  for (MachineBasicBlock::livein_iterator I = MBB->livein_begin(),
         E = MBB->livein_end(); I != E; ++I) {
    if (!TargetRegisterInfo::isPhysicalRegister(*I)) {
      report("MBB live-in list contains non-physical register", MBB);
      continue;
    }
    regsLive.insert(*I);
    for (MCSubRegIterator SubRegs(*I, TRI); SubRegs.isValid(); ++SubRegs)
      regsLive.insert(*SubRegs);
  }
  regsLiveInButUnused = regsLive;

  const MachineFrameInfo *MFI = MF->getFrameInfo();
  assert(MFI && "Function has no frame info");
  BitVector PR = MFI->getPristineRegs(MBB);
  for (int I = PR.find_first(); I>0; I = PR.find_next(I)) {
    regsLive.insert(I);
    for (MCSubRegIterator SubRegs(I, TRI); SubRegs.isValid(); ++SubRegs)
      regsLive.insert(*SubRegs);
  }

  regsKilled.clear();
  regsDefined.clear();

  if (Indexes)
    lastIndex = Indexes->getMBBStartIdx(MBB);
}

// This function gets called for all bundle headers, including normal
// stand-alone unbundled instructions.
void MachineVerifier::visitMachineBundleBefore(const MachineInstr *MI) {
  if (Indexes && Indexes->hasIndex(MI)) {
    SlotIndex idx = Indexes->getInstructionIndex(MI);
    if (!(idx > lastIndex)) {
      report("Instruction index out of order", MI);
      *OS << "Last instruction was at " << lastIndex << '\n';
    }
    lastIndex = idx;
  }

  // Ensure non-terminators don't follow terminators.
  // Ignore predicated terminators formed by if conversion.
  // FIXME: If conversion shouldn't need to violate this rule.
  if (MI->isTerminator() && !TII->isPredicated(MI)) {
    if (!FirstTerminator)
      FirstTerminator = MI;
  } else if (FirstTerminator) {
    report("Non-terminator instruction after the first terminator", MI);
    *OS << "First terminator was:\t" << *FirstTerminator;
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
  // Extra_AsmDialect = 4, Extra_MayLoad = 8, and Extra_MayStore = 16.
  if (!isUInt<5>(MI->getOperand(1).getImm()))
    report("Unknown asm flags", &MI->getOperand(1), 1);

  assert(InlineAsm::MIOp_FirstOperand == 2 && "Asm format changed");

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

void MachineVerifier::visitMachineInstrBefore(const MachineInstr *MI) {
  const MCInstrDesc &MCID = MI->getDesc();
  if (MI->getNumOperands() < MCID.getNumOperands()) {
    report("Too few operands", MI);
    *OS << MCID.getNumOperands() << " operands expected, but "
        << MI->getNumExplicitOperands() << " given.\n";
  }

  // Check the tied operands.
  if (MI->isInlineAsm())
    verifyInlineAsm(MI);

  // Check the MachineMemOperands for basic consistency.
  for (MachineInstr::mmo_iterator I = MI->memoperands_begin(),
       E = MI->memoperands_end(); I != E; ++I) {
    if ((*I)->isLoad() && !MI->mayLoad())
      report("Missing mayLoad flag", MI);
    if ((*I)->isStore() && !MI->mayStore())
      report("Missing mayStore flag", MI);
  }

  // Debug values must not have a slot index.
  // Other instructions must have one, unless they are inside a bundle.
  if (LiveInts) {
    bool mapped = !LiveInts->isNotInMIMap(MI);
    if (MI->isDebugValue()) {
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

  StringRef ErrorInfo;
  if (!TII->verifyInstruction(MI, ErrorInfo))
    report(ErrorInfo.data(), MI);
}

void
MachineVerifier::visitMachineOperand(const MachineOperand *MO, unsigned MONum) {
  const MachineInstr *MI = MO->getParent();
  const MCInstrDesc &MCID = MI->getDesc();

  // The first MCID.NumDefs operands must be explicit register defines
  if (MONum < MCID.getNumDefs()) {
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
    // e.g., LDM_RET in the arm back end.
    if (MO->isReg() &&
        !(MI->isVariadic() && MONum == MCID.getNumOperands()-1)) {
      if (MO->isDef() && !MCOI.isOptionalDef())
          report("Explicit operand marked as def", MO, MONum);
      if (MO->isImplicit())
        report("Explicit operand marked as implicit", MO, MONum);
    }

    int TiedTo = MCID.getOperandConstraint(MONum, MCOI::TIED_TO);
    if (TiedTo != -1) {
      if (!MO->isReg())
        report("Tied use must be a register", MO, MONum);
      else if (!MO->isTied())
        report("Operand should be tied", MO, MONum);
      else if (unsigned(TiedTo) != MI->findTiedOperandIdx(MONum))
        report("Tied def doesn't match MCInstrDesc", MO, MONum);
    } else if (MO->isReg() && MO->isTied())
      report("Explicit operand should not be tied", MO, MONum);
  } else {
    // ARM adds %reg0 operands to indicate predicates. We'll allow that.
    if (MO->isReg() && !MO->isImplicit() && !MI->isVariadic() && MO->getReg())
      report("Extra explicit operand on non-variadic instruction", MO, MONum);
  }

  switch (MO->getType()) {
  case MachineOperand::MO_Register: {
    const unsigned Reg = MO->getReg();
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

    // Verify two-address constraints after leaving SSA form.
    unsigned DefIdx;
    if (!MRI->isSSA() && MO->isUse() &&
        MI->isRegTiedToDefOperand(MONum, &DefIdx) &&
        Reg != MI->getOperand(DefIdx).getReg())
      report("Two-address instruction operands must be identical", MO, MONum);

    // Check register classes.
    if (MONum < MCID.getNumOperands() && !MO->isImplicit()) {
      unsigned SubIdx = MO->getSubReg();

      if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
        if (SubIdx) {
          report("Illegal subregister index for physical register", MO, MONum);
          return;
        }
        if (const TargetRegisterClass *DRC =
              TII->getRegClass(MCID, MONum, TRI, *MF)) {
          if (!DRC->contains(Reg)) {
            report("Illegal physical register for instruction", MO, MONum);
            *OS << TRI->getName(Reg) << " is not a "
                << DRC->getName() << " register.\n";
          }
        }
      } else {
        // Virtual register.
        const TargetRegisterClass *RC = MRI->getRegClass(Reg);
        if (SubIdx) {
          const TargetRegisterClass *SRC =
            TRI->getSubClassWithSubReg(RC, SubIdx);
          if (!SRC) {
            report("Invalid subregister index for virtual register", MO, MONum);
            *OS << "Register class " << RC->getName()
                << " does not support subreg index " << SubIdx << "\n";
            return;
          }
          if (RC != SRC) {
            report("Invalid register class for subregister index", MO, MONum);
            *OS << "Register class " << RC->getName()
                << " does not fully support subreg index " << SubIdx << "\n";
            return;
          }
        }
        if (const TargetRegisterClass *DRC =
              TII->getRegClass(MCID, MONum, TRI, *MF)) {
          if (SubIdx) {
            const TargetRegisterClass *SuperRC =
              TRI->getLargestLegalSuperClass(RC);
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
            *OS << "Expected a " << DRC->getName() << " register, but got a "
                << RC->getName() << " register\n";
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
        LiveInts && !LiveInts->isNotInMIMap(MI)) {
      LiveInterval &LI = LiveStks->getInterval(MO->getIndex());
      SlotIndex Idx = LiveInts->getInstructionIndex(MI);
      if (MI->mayLoad() && !LI.liveAt(Idx.getRegSlot(true))) {
        report("Instruction loads from dead spill slot", MO, MONum);
        *OS << "Live stack: " << LI << '\n';
      }
      if (MI->mayStore() && !LI.liveAt(Idx.getRegSlot())) {
        report("Instruction stores to dead spill slot", MO, MONum);
        *OS << "Live stack: " << LI << '\n';
      }
    }
    break;

  default:
    break;
  }
}

void MachineVerifier::checkLiveness(const MachineOperand *MO, unsigned MONum) {
  const MachineInstr *MI = MO->getParent();
  const unsigned Reg = MO->getReg();

  // Both use and def operands can read a register.
  if (MO->readsReg()) {
    regsLiveInButUnused.erase(Reg);

    if (MO->isKill())
      addRegWithSubRegs(regsKilled, Reg);

    // Check that LiveVars knows this kill.
    if (LiveVars && TargetRegisterInfo::isVirtualRegister(Reg) &&
        MO->isKill()) {
      LiveVariables::VarInfo &VI = LiveVars->getVarInfo(Reg);
      if (std::find(VI.Kills.begin(), VI.Kills.end(), MI) == VI.Kills.end())
        report("Kill missing from LiveVariables", MO, MONum);
    }

    // Check LiveInts liveness and kill.
    if (LiveInts && !LiveInts->isNotInMIMap(MI)) {
      SlotIndex UseIdx = LiveInts->getInstructionIndex(MI);
      // Check the cached regunit intervals.
      if (TargetRegisterInfo::isPhysicalRegister(Reg) && !isReserved(Reg)) {
        for (MCRegUnitIterator Units(Reg, TRI); Units.isValid(); ++Units) {
          if (const LiveInterval *LI = LiveInts->getCachedRegUnit(*Units)) {
            LiveRangeQuery LRQ(*LI, UseIdx);
            if (!LRQ.valueIn()) {
              report("No live range at use", MO, MONum);
              *OS << UseIdx << " is not live in " << PrintRegUnit(*Units, TRI)
                  << ' ' << *LI << '\n';
            }
            if (MO->isKill() && !LRQ.isKill()) {
              report("Live range continues after kill flag", MO, MONum);
              *OS << PrintRegUnit(*Units, TRI) << ' ' << *LI << '\n';
            }
          }
        }
      }

      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        if (LiveInts->hasInterval(Reg)) {
          // This is a virtual register interval.
          const LiveInterval &LI = LiveInts->getInterval(Reg);
          LiveRangeQuery LRQ(LI, UseIdx);
          if (!LRQ.valueIn()) {
            report("No live range at use", MO, MONum);
            *OS << UseIdx << " is not live in " << LI << '\n';
          }
          // Check for extra kill flags.
          // Note that we allow missing kill flags for now.
          if (MO->isKill() && !LRQ.isKill()) {
            report("Live range continues after kill flag", MO, MONum);
            *OS << "Live range: " << LI << '\n';
          }
        } else {
          report("Virtual register has no live interval", MO, MONum);
        }
      }
    }

    // Use of a dead register.
    if (!regsLive.count(Reg)) {
      if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
        // Reserved registers may be used even when 'dead'.
        if (!isReserved(Reg))
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
    if (MRI->isSSA() && TargetRegisterInfo::isVirtualRegister(Reg) &&
        llvm::next(MRI->def_begin(Reg)) != MRI->def_end())
      report("Multiple virtual register defs in SSA form", MO, MONum);

    // Check LiveInts for a live range, but only for virtual registers.
    if (LiveInts && TargetRegisterInfo::isVirtualRegister(Reg) &&
        !LiveInts->isNotInMIMap(MI)) {
      SlotIndex DefIdx = LiveInts->getInstructionIndex(MI);
      DefIdx = DefIdx.getRegSlot(MO->isEarlyClobber());
      if (LiveInts->hasInterval(Reg)) {
        const LiveInterval &LI = LiveInts->getInterval(Reg);
        if (const VNInfo *VNI = LI.getVNInfoAt(DefIdx)) {
          assert(VNI && "NULL valno is not allowed");
          if (VNI->def != DefIdx) {
            report("Inconsistent valno->def", MO, MONum);
            *OS << "Valno " << VNI->id << " is not defined at "
              << DefIdx << " in " << LI << '\n';
          }
        } else {
          report("No live range at def", MO, MONum);
          *OS << DefIdx << " is not live in " << LI << '\n';
        }
      } else {
        report("Virtual register has no Live interval", MO, MONum);
      }
    }
  }
}

void MachineVerifier::visitMachineInstrAfter(const MachineInstr *MI) {
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
    for (RegSet::iterator I = regsLive.begin(), E = regsLive.end(); I != E; ++I)
      if (TargetRegisterInfo::isPhysicalRegister(*I) &&
          MachineOperand::clobbersPhysReg(Mask, *I))
        regsDead.push_back(*I);
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
      *OS << "Block ends at " << stop
          << " last instruction was at " << lastIndex << '\n';
    }
    lastIndex = stop;
  }
}

// Calculate the largest possible vregsPassed sets. These are the registers that
// can pass through an MBB live, but may not be live every time. It is assumed
// that all vregsPassed sets are empty before the call.
void MachineVerifier::calcRegsPassed() {
  // First push live-out regs to successors' vregsPassed. Remember the MBBs that
  // have any vregsPassed.
  SmallPtrSet<const MachineBasicBlock*, 8> todo;
  for (MachineFunction::const_iterator MFI = MF->begin(), MFE = MF->end();
       MFI != MFE; ++MFI) {
    const MachineBasicBlock &MBB(*MFI);
    BBInfo &MInfo = MBBInfoMap[&MBB];
    if (!MInfo.reachable)
      continue;
    for (MachineBasicBlock::const_succ_iterator SuI = MBB.succ_begin(),
           SuE = MBB.succ_end(); SuI != SuE; ++SuI) {
      BBInfo &SInfo = MBBInfoMap[*SuI];
      if (SInfo.addPassed(MInfo.regsLiveOut))
        todo.insert(*SuI);
    }
  }

  // Iteratively push vregsPassed to successors. This will converge to the same
  // final state regardless of DenseSet iteration order.
  while (!todo.empty()) {
    const MachineBasicBlock *MBB = *todo.begin();
    todo.erase(MBB);
    BBInfo &MInfo = MBBInfoMap[MBB];
    for (MachineBasicBlock::const_succ_iterator SuI = MBB->succ_begin(),
           SuE = MBB->succ_end(); SuI != SuE; ++SuI) {
      if (*SuI == MBB)
        continue;
      BBInfo &SInfo = MBBInfoMap[*SuI];
      if (SInfo.addPassed(MInfo.vregsPassed))
        todo.insert(*SuI);
    }
  }
}

// Calculate the set of virtual registers that must be passed through each basic
// block in order to satisfy the requirements of successor blocks. This is very
// similar to calcRegsPassed, only backwards.
void MachineVerifier::calcRegsRequired() {
  // First push live-in regs to predecessors' vregsRequired.
  SmallPtrSet<const MachineBasicBlock*, 8> todo;
  for (MachineFunction::const_iterator MFI = MF->begin(), MFE = MF->end();
       MFI != MFE; ++MFI) {
    const MachineBasicBlock &MBB(*MFI);
    BBInfo &MInfo = MBBInfoMap[&MBB];
    for (MachineBasicBlock::const_pred_iterator PrI = MBB.pred_begin(),
           PrE = MBB.pred_end(); PrI != PrE; ++PrI) {
      BBInfo &PInfo = MBBInfoMap[*PrI];
      if (PInfo.addRequired(MInfo.vregsLiveIn))
        todo.insert(*PrI);
    }
  }

  // Iteratively push vregsRequired to predecessors. This will converge to the
  // same final state regardless of DenseSet iteration order.
  while (!todo.empty()) {
    const MachineBasicBlock *MBB = *todo.begin();
    todo.erase(MBB);
    BBInfo &MInfo = MBBInfoMap[MBB];
    for (MachineBasicBlock::const_pred_iterator PrI = MBB->pred_begin(),
           PrE = MBB->pred_end(); PrI != PrE; ++PrI) {
      if (*PrI == MBB)
        continue;
      BBInfo &SInfo = MBBInfoMap[*PrI];
      if (SInfo.addRequired(MInfo.vregsRequired))
        todo.insert(*PrI);
    }
  }
}

// Check PHI instructions at the beginning of MBB. It is assumed that
// calcRegsPassed has been run so BBInfo::isLiveOut is valid.
void MachineVerifier::checkPHIOps(const MachineBasicBlock *MBB) {
  SmallPtrSet<const MachineBasicBlock*, 8> seen;
  for (MachineBasicBlock::const_iterator BBI = MBB->begin(), BBE = MBB->end();
       BBI != BBE && BBI->isPHI(); ++BBI) {
    seen.clear();

    for (unsigned i = 1, e = BBI->getNumOperands(); i != e; i += 2) {
      unsigned Reg = BBI->getOperand(i).getReg();
      const MachineBasicBlock *Pre = BBI->getOperand(i + 1).getMBB();
      if (!Pre->isSuccessor(MBB))
        continue;
      seen.insert(Pre);
      BBInfo &PrInfo = MBBInfoMap[Pre];
      if (PrInfo.reachable && !PrInfo.isLiveOut(Reg))
        report("PHI operand is not live-out from predecessor",
               &BBI->getOperand(i), i);
    }

    // Did we see all predecessors?
    for (MachineBasicBlock::const_pred_iterator PrI = MBB->pred_begin(),
           PrE = MBB->pred_end(); PrI != PrE; ++PrI) {
      if (!seen.count(*PrI)) {
        report("Missing PHI operand", BBI);
        *OS << "BB#" << (*PrI)->getNumber()
            << " is a predecessor according to the CFG.\n";
      }
    }
  }
}

void MachineVerifier::visitMachineFunctionAfter() {
  calcRegsPassed();

  for (MachineFunction::const_iterator MFI = MF->begin(), MFE = MF->end();
       MFI != MFE; ++MFI) {
    BBInfo &MInfo = MBBInfoMap[MFI];

    // Skip unreachable MBBs.
    if (!MInfo.reachable)
      continue;

    checkPHIOps(MFI);
  }

  // Now check liveness info if available
  calcRegsRequired();

  // Check for killed virtual registers that should be live out.
  for (MachineFunction::const_iterator MFI = MF->begin(), MFE = MF->end();
       MFI != MFE; ++MFI) {
    BBInfo &MInfo = MBBInfoMap[MFI];
    for (RegSet::iterator
         I = MInfo.vregsRequired.begin(), E = MInfo.vregsRequired.end(); I != E;
         ++I)
      if (MInfo.regsKilled.count(*I)) {
        report("Virtual register killed in block, but needed live out.", MFI);
        *OS << "Virtual register " << PrintReg(*I)
            << " is used after the block.\n";
      }
  }

  if (!MF->empty()) {
    BBInfo &MInfo = MBBInfoMap[&MF->front()];
    for (RegSet::iterator
         I = MInfo.vregsRequired.begin(), E = MInfo.vregsRequired.end(); I != E;
         ++I)
      report("Virtual register def doesn't dominate all uses.",
             MRI->getVRegDef(*I));
  }

  if (LiveVars)
    verifyLiveVariables();
  if (LiveInts)
    verifyLiveIntervals();
}

void MachineVerifier::verifyLiveVariables() {
  assert(LiveVars && "Don't call verifyLiveVariables without LiveVars");
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    LiveVariables::VarInfo &VI = LiveVars->getVarInfo(Reg);
    for (MachineFunction::const_iterator MFI = MF->begin(), MFE = MF->end();
         MFI != MFE; ++MFI) {
      BBInfo &MInfo = MBBInfoMap[MFI];

      // Our vregsRequired should be identical to LiveVariables' AliveBlocks
      if (MInfo.vregsRequired.count(Reg)) {
        if (!VI.AliveBlocks.test(MFI->getNumber())) {
          report("LiveVariables: Block missing from AliveBlocks", MFI);
          *OS << "Virtual register " << PrintReg(Reg)
              << " must be live through the block.\n";
        }
      } else {
        if (VI.AliveBlocks.test(MFI->getNumber())) {
          report("LiveVariables: Block should not be in AliveBlocks", MFI);
          *OS << "Virtual register " << PrintReg(Reg)
              << " is not needed live through the block.\n";
        }
      }
    }
  }
}

void MachineVerifier::verifyLiveIntervals() {
  assert(LiveInts && "Don't call verifyLiveIntervals without LiveInts");
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);

    // Spilling and splitting may leave unused registers around. Skip them.
    if (MRI->reg_nodbg_empty(Reg))
      continue;

    if (!LiveInts->hasInterval(Reg)) {
      report("Missing live interval for virtual register", MF);
      *OS << PrintReg(Reg, TRI) << " still has defs or uses\n";
      continue;
    }

    const LiveInterval &LI = LiveInts->getInterval(Reg);
    assert(Reg == LI.reg && "Invalid reg to interval mapping");
    verifyLiveInterval(LI);
  }

  // Verify all the cached regunit intervals.
  for (unsigned i = 0, e = TRI->getNumRegUnits(); i != e; ++i)
    if (const LiveInterval *LI = LiveInts->getCachedRegUnit(i))
      verifyLiveInterval(*LI);
}

void MachineVerifier::verifyLiveIntervalValue(const LiveInterval &LI,
                                              VNInfo *VNI) {
  if (VNI->isUnused())
    return;

  const VNInfo *DefVNI = LI.getVNInfoAt(VNI->def);

  if (!DefVNI) {
    report("Valno not live at def and not marked unused", MF, LI);
    *OS << "Valno #" << VNI->id << '\n';
    return;
  }

  if (DefVNI != VNI) {
    report("Live range at def has different valno", MF, LI);
    *OS << "Valno #" << VNI->id << " is defined at " << VNI->def
        << " where valno #" << DefVNI->id << " is live\n";
    return;
  }

  const MachineBasicBlock *MBB = LiveInts->getMBBFromIndex(VNI->def);
  if (!MBB) {
    report("Invalid definition index", MF, LI);
    *OS << "Valno #" << VNI->id << " is defined at " << VNI->def
        << " in " << LI << '\n';
    return;
  }

  if (VNI->isPHIDef()) {
    if (VNI->def != LiveInts->getMBBStartIdx(MBB)) {
      report("PHIDef value is not defined at MBB start", MBB, LI);
      *OS << "Valno #" << VNI->id << " is defined at " << VNI->def
          << ", not at the beginning of BB#" << MBB->getNumber() << '\n';
    }
    return;
  }

  // Non-PHI def.
  const MachineInstr *MI = LiveInts->getInstructionFromIndex(VNI->def);
  if (!MI) {
    report("No instruction at def index", MBB, LI);
    *OS << "Valno #" << VNI->id << " is defined at " << VNI->def << '\n';
    return;
  }

  bool hasDef = false;
  bool isEarlyClobber = false;
  for (ConstMIBundleOperands MOI(MI); MOI.isValid(); ++MOI) {
    if (!MOI->isReg() || !MOI->isDef())
      continue;
    if (TargetRegisterInfo::isVirtualRegister(LI.reg)) {
      if (MOI->getReg() != LI.reg)
        continue;
    } else {
      if (!TargetRegisterInfo::isPhysicalRegister(MOI->getReg()) ||
          !TRI->hasRegUnit(MOI->getReg(), LI.reg))
        continue;
    }
    hasDef = true;
    if (MOI->isEarlyClobber())
      isEarlyClobber = true;
  }

  if (!hasDef) {
    report("Defining instruction does not modify register", MI);
    *OS << "Valno #" << VNI->id << " in " << LI << '\n';
  }

  // Early clobber defs begin at USE slots, but other defs must begin at
  // DEF slots.
  if (isEarlyClobber) {
    if (!VNI->def.isEarlyClobber()) {
      report("Early clobber def must be at an early-clobber slot", MBB, LI);
      *OS << "Valno #" << VNI->id << " is defined at " << VNI->def << '\n';
    }
  } else if (!VNI->def.isRegister()) {
    report("Non-PHI, non-early clobber def must be at a register slot",
           MBB, LI);
    *OS << "Valno #" << VNI->id << " is defined at " << VNI->def << '\n';
  }
}

void
MachineVerifier::verifyLiveIntervalSegment(const LiveInterval &LI,
                                           LiveInterval::const_iterator I) {
  const VNInfo *VNI = I->valno;
  assert(VNI && "Live range has no valno");

  if (VNI->id >= LI.getNumValNums() || VNI != LI.getValNumInfo(VNI->id)) {
    report("Foreign valno in live range", MF, LI);
    *OS << *I << " has a bad valno\n";
  }

  if (VNI->isUnused()) {
    report("Live range valno is marked unused", MF, LI);
    *OS << *I << '\n';
  }

  const MachineBasicBlock *MBB = LiveInts->getMBBFromIndex(I->start);
  if (!MBB) {
    report("Bad start of live segment, no basic block", MF, LI);
    *OS << *I << '\n';
    return;
  }
  SlotIndex MBBStartIdx = LiveInts->getMBBStartIdx(MBB);
  if (I->start != MBBStartIdx && I->start != VNI->def) {
    report("Live segment must begin at MBB entry or valno def", MBB, LI);
    *OS << *I << '\n';
  }

  const MachineBasicBlock *EndMBB =
    LiveInts->getMBBFromIndex(I->end.getPrevSlot());
  if (!EndMBB) {
    report("Bad end of live segment, no basic block", MF, LI);
    *OS << *I << '\n';
    return;
  }

  // No more checks for live-out segments.
  if (I->end == LiveInts->getMBBEndIdx(EndMBB))
    return;

  // RegUnit intervals are allowed dead phis.
  if (!TargetRegisterInfo::isVirtualRegister(LI.reg) && VNI->isPHIDef() &&
      I->start == VNI->def && I->end == VNI->def.getDeadSlot())
    return;

  // The live segment is ending inside EndMBB
  const MachineInstr *MI =
    LiveInts->getInstructionFromIndex(I->end.getPrevSlot());
  if (!MI) {
    report("Live segment doesn't end at a valid instruction", EndMBB, LI);
    *OS << *I << '\n';
    return;
  }

  // The block slot must refer to a basic block boundary.
  if (I->end.isBlock()) {
    report("Live segment ends at B slot of an instruction", EndMBB, LI);
    *OS << *I << '\n';
  }

  if (I->end.isDead()) {
    // Segment ends on the dead slot.
    // That means there must be a dead def.
    if (!SlotIndex::isSameInstr(I->start, I->end)) {
      report("Live segment ending at dead slot spans instructions", EndMBB, LI);
      *OS << *I << '\n';
    }
  }

  // A live segment can only end at an early-clobber slot if it is being
  // redefined by an early-clobber def.
  if (I->end.isEarlyClobber()) {
    if (I+1 == LI.end() || (I+1)->start != I->end) {
      report("Live segment ending at early clobber slot must be "
             "redefined by an EC def in the same instruction", EndMBB, LI);
      *OS << *I << '\n';
    }
  }

  // The following checks only apply to virtual registers. Physreg liveness
  // is too weird to check.
  if (TargetRegisterInfo::isVirtualRegister(LI.reg)) {
    // A live range can end with either a redefinition, a kill flag on a
    // use, or a dead flag on a def.
    bool hasRead = false;
    bool hasDeadDef = false;
    for (ConstMIBundleOperands MOI(MI); MOI.isValid(); ++MOI) {
      if (!MOI->isReg() || MOI->getReg() != LI.reg)
        continue;
      if (MOI->readsReg())
        hasRead = true;
      if (MOI->isDef() && MOI->isDead())
        hasDeadDef = true;
    }

    if (I->end.isDead()) {
      if (!hasDeadDef) {
        report("Instruction doesn't have a dead def operand", MI);
        I->print(*OS);
        *OS << " in " << LI << '\n';
      }
    } else {
      if (!hasRead) {
        report("Instruction ending live range doesn't read the register", MI);
        *OS << *I << " in " << LI << '\n';
      }
    }
  }

  // Now check all the basic blocks in this live segment.
  MachineFunction::const_iterator MFI = MBB;
  // Is this live range the beginning of a non-PHIDef VN?
  if (I->start == VNI->def && !VNI->isPHIDef()) {
    // Not live-in to any blocks.
    if (MBB == EndMBB)
      return;
    // Skip this block.
    ++MFI;
  }
  for (;;) {
    assert(LiveInts->isLiveInToMBB(LI, MFI));
    // We don't know how to track physregs into a landing pad.
    if (!TargetRegisterInfo::isVirtualRegister(LI.reg) &&
        MFI->isLandingPad()) {
      if (&*MFI == EndMBB)
        break;
      ++MFI;
      continue;
    }

    // Is VNI a PHI-def in the current block?
    bool IsPHI = VNI->isPHIDef() &&
      VNI->def == LiveInts->getMBBStartIdx(MFI);

    // Check that VNI is live-out of all predecessors.
    for (MachineBasicBlock::const_pred_iterator PI = MFI->pred_begin(),
         PE = MFI->pred_end(); PI != PE; ++PI) {
      SlotIndex PEnd = LiveInts->getMBBEndIdx(*PI);
      const VNInfo *PVNI = LI.getVNInfoBefore(PEnd);

      // All predecessors must have a live-out value.
      if (!PVNI) {
        report("Register not marked live out of predecessor", *PI, LI);
        *OS << "Valno #" << VNI->id << " live into BB#" << MFI->getNumber()
            << '@' << LiveInts->getMBBStartIdx(MFI) << ", not live before "
            << PEnd << '\n';
        continue;
      }

      // Only PHI-defs can take different predecessor values.
      if (!IsPHI && PVNI != VNI) {
        report("Different value live out of predecessor", *PI, LI);
        *OS << "Valno #" << PVNI->id << " live out of BB#"
            << (*PI)->getNumber() << '@' << PEnd
            << "\nValno #" << VNI->id << " live into BB#" << MFI->getNumber()
            << '@' << LiveInts->getMBBStartIdx(MFI) << '\n';
      }
    }
    if (&*MFI == EndMBB)
      break;
    ++MFI;
  }
}

void MachineVerifier::verifyLiveInterval(const LiveInterval &LI) {
  for (LiveInterval::const_vni_iterator I = LI.vni_begin(), E = LI.vni_end();
       I!=E; ++I)
    verifyLiveIntervalValue(LI, *I);

  for (LiveInterval::const_iterator I = LI.begin(), E = LI.end(); I!=E; ++I)
    verifyLiveIntervalSegment(LI, I);

  // Check the LI only has one connected component.
  if (TargetRegisterInfo::isVirtualRegister(LI.reg)) {
    ConnectedVNInfoEqClasses ConEQ(*LiveInts);
    unsigned NumComp = ConEQ.Classify(&LI);
    if (NumComp > 1) {
      report("Multiple connected components in live interval", MF, LI);
      for (unsigned comp = 0; comp != NumComp; ++comp) {
        *OS << comp << ": valnos";
        for (LiveInterval::const_vni_iterator I = LI.vni_begin(),
             E = LI.vni_end(); I!=E; ++I)
          if (comp == ConEQ.getEqClass(*I))
            *OS << ' ' << (*I)->id;
        *OS << '\n';
      }
    }
  }
}
