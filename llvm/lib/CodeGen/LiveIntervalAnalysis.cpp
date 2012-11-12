//===-- LiveIntervalAnalysis.cpp - Live Interval Analysis -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveInterval analysis pass which is used
// by the Linear Scan Register allocator. This pass linearizes the
// basic blocks of the function in DFS order and uses the
// LiveVariables pass to conservatively compute live intervals for
// each virtual and physical register.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/Value.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "LiveRangeCalc.h"
#include "VirtRegMap.h"
#include <algorithm>
#include <limits>
#include <cmath>
using namespace llvm;

// Switch to the new experimental algorithm for computing live intervals.
static cl::opt<bool>
NewLiveIntervals("new-live-intervals", cl::Hidden,
                 cl::desc("Use new algorithm forcomputing live intervals"));

char LiveIntervals::ID = 0;
char &llvm::LiveIntervalsID = LiveIntervals::ID;
INITIALIZE_PASS_BEGIN(LiveIntervals, "liveintervals",
                "Live Interval Analysis", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(LiveVariables)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_END(LiveIntervals, "liveintervals",
                "Live Interval Analysis", false, false)

void LiveIntervals::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addRequired<LiveVariables>();
  AU.addPreserved<LiveVariables>();
  AU.addPreservedID(MachineLoopInfoID);
  AU.addRequiredTransitiveID(MachineDominatorsID);
  AU.addPreservedID(MachineDominatorsID);
  AU.addPreserved<SlotIndexes>();
  AU.addRequiredTransitive<SlotIndexes>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

LiveIntervals::LiveIntervals() : MachineFunctionPass(ID),
  DomTree(0), LRCalc(0) {
  initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
}

LiveIntervals::~LiveIntervals() {
  delete LRCalc;
}

void LiveIntervals::releaseMemory() {
  // Free the live intervals themselves.
  for (unsigned i = 0, e = VirtRegIntervals.size(); i != e; ++i)
    delete VirtRegIntervals[TargetRegisterInfo::index2VirtReg(i)];
  VirtRegIntervals.clear();
  RegMaskSlots.clear();
  RegMaskBits.clear();
  RegMaskBlocks.clear();

  for (unsigned i = 0, e = RegUnitIntervals.size(); i != e; ++i)
    delete RegUnitIntervals[i];
  RegUnitIntervals.clear();

  // Release VNInfo memory regions, VNInfo objects don't need to be dtor'd.
  VNInfoAllocator.Reset();
}

/// runOnMachineFunction - Register allocate the whole function
///
bool LiveIntervals::runOnMachineFunction(MachineFunction &fn) {
  MF = &fn;
  MRI = &MF->getRegInfo();
  TM = &fn.getTarget();
  TRI = TM->getRegisterInfo();
  TII = TM->getInstrInfo();
  AA = &getAnalysis<AliasAnalysis>();
  LV = &getAnalysis<LiveVariables>();
  Indexes = &getAnalysis<SlotIndexes>();
  DomTree = &getAnalysis<MachineDominatorTree>();
  if (!LRCalc)
    LRCalc = new LiveRangeCalc();

  // Allocate space for all virtual registers.
  VirtRegIntervals.resize(MRI->getNumVirtRegs());

  if (NewLiveIntervals) {
    // This is the new way of computing live intervals.
    // It is independent of LiveVariables, and it can run at any time.
    computeVirtRegs();
    computeRegMasks();
  } else {
    // This is the old way of computing live intervals.
    // It depends on LiveVariables.
    computeIntervals();
  }
  computeLiveInRegUnits();

  DEBUG(dump());
  return true;
}

/// print - Implement the dump method.
void LiveIntervals::print(raw_ostream &OS, const Module* ) const {
  OS << "********** INTERVALS **********\n";

  // Dump the regunits.
  for (unsigned i = 0, e = RegUnitIntervals.size(); i != e; ++i)
    if (LiveInterval *LI = RegUnitIntervals[i])
      OS << PrintRegUnit(i, TRI) << " = " << *LI << '\n';

  // Dump the virtregs.
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (hasInterval(Reg))
      OS << PrintReg(Reg) << " = " << getInterval(Reg) << '\n';
  }

  OS << "RegMasks:";
  for (unsigned i = 0, e = RegMaskSlots.size(); i != e; ++i)
    OS << ' ' << RegMaskSlots[i];
  OS << '\n';

  printInstrs(OS);
}

void LiveIntervals::printInstrs(raw_ostream &OS) const {
  OS << "********** MACHINEINSTRS **********\n";
  MF->print(OS, Indexes);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void LiveIntervals::dumpInstrs() const {
  printInstrs(dbgs());
}
#endif

static
bool MultipleDefsBySameMI(const MachineInstr &MI, unsigned MOIdx) {
  unsigned Reg = MI.getOperand(MOIdx).getReg();
  for (unsigned i = MOIdx+1, e = MI.getNumOperands(); i < e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg())
      continue;
    if (MO.getReg() == Reg && MO.isDef()) {
      assert(MI.getOperand(MOIdx).getSubReg() != MO.getSubReg() &&
             MI.getOperand(MOIdx).getSubReg() &&
             (MO.getSubReg() || MO.isImplicit()));
      return true;
    }
  }
  return false;
}

/// isPartialRedef - Return true if the specified def at the specific index is
/// partially re-defining the specified live interval. A common case of this is
/// a definition of the sub-register.
bool LiveIntervals::isPartialRedef(SlotIndex MIIdx, MachineOperand &MO,
                                   LiveInterval &interval) {
  if (!MO.getSubReg() || MO.isEarlyClobber())
    return false;

  SlotIndex RedefIndex = MIIdx.getRegSlot();
  const LiveRange *OldLR =
    interval.getLiveRangeContaining(RedefIndex.getRegSlot(true));
  MachineInstr *DefMI = getInstructionFromIndex(OldLR->valno->def);
  if (DefMI != 0) {
    return DefMI->findRegisterDefOperandIdx(interval.reg) != -1;
  }
  return false;
}

void LiveIntervals::handleVirtualRegisterDef(MachineBasicBlock *mbb,
                                             MachineBasicBlock::iterator mi,
                                             SlotIndex MIIdx,
                                             MachineOperand& MO,
                                             unsigned MOIdx,
                                             LiveInterval &interval) {
  DEBUG(dbgs() << "\t\tregister: " << PrintReg(interval.reg, TRI));

  // Virtual registers may be defined multiple times (due to phi
  // elimination and 2-addr elimination).  Much of what we do only has to be
  // done once for the vreg.  We use an empty interval to detect the first
  // time we see a vreg.
  LiveVariables::VarInfo& vi = LV->getVarInfo(interval.reg);
  if (interval.empty()) {
    // Get the Idx of the defining instructions.
    SlotIndex defIndex = MIIdx.getRegSlot(MO.isEarlyClobber());

    // Make sure the first definition is not a partial redefinition.
    assert(!MO.readsReg() && "First def cannot also read virtual register "
           "missing <undef> flag?");

    VNInfo *ValNo = interval.getNextValue(defIndex, VNInfoAllocator);
    assert(ValNo->id == 0 && "First value in interval is not 0?");

    // Loop over all of the blocks that the vreg is defined in.  There are
    // two cases we have to handle here.  The most common case is a vreg
    // whose lifetime is contained within a basic block.  In this case there
    // will be a single kill, in MBB, which comes after the definition.
    if (vi.Kills.size() == 1 && vi.Kills[0]->getParent() == mbb) {
      // FIXME: what about dead vars?
      SlotIndex killIdx;
      if (vi.Kills[0] != mi)
        killIdx = getInstructionIndex(vi.Kills[0]).getRegSlot();
      else
        killIdx = defIndex.getDeadSlot();

      // If the kill happens after the definition, we have an intra-block
      // live range.
      if (killIdx > defIndex) {
        assert(vi.AliveBlocks.empty() &&
               "Shouldn't be alive across any blocks!");
        LiveRange LR(defIndex, killIdx, ValNo);
        interval.addRange(LR);
        DEBUG(dbgs() << " +" << LR << "\n");
        return;
      }
    }

    // The other case we handle is when a virtual register lives to the end
    // of the defining block, potentially live across some blocks, then is
    // live into some number of blocks, but gets killed.  Start by adding a
    // range that goes from this definition to the end of the defining block.
    LiveRange NewLR(defIndex, getMBBEndIdx(mbb), ValNo);
    DEBUG(dbgs() << " +" << NewLR);
    interval.addRange(NewLR);

    bool PHIJoin = LV->isPHIJoin(interval.reg);

    if (PHIJoin) {
      // A phi join register is killed at the end of the MBB and revived as a
      // new valno in the killing blocks.
      assert(vi.AliveBlocks.empty() && "Phi join can't pass through blocks");
      DEBUG(dbgs() << " phi-join");
    } else {
      // Iterate over all of the blocks that the variable is completely
      // live in, adding [insrtIndex(begin), instrIndex(end)+4) to the
      // live interval.
      for (SparseBitVector<>::iterator I = vi.AliveBlocks.begin(),
               E = vi.AliveBlocks.end(); I != E; ++I) {
        MachineBasicBlock *aliveBlock = MF->getBlockNumbered(*I);
        LiveRange LR(getMBBStartIdx(aliveBlock), getMBBEndIdx(aliveBlock),
                     ValNo);
        interval.addRange(LR);
        DEBUG(dbgs() << " +" << LR);
      }
    }

    // Finally, this virtual register is live from the start of any killing
    // block to the 'use' slot of the killing instruction.
    for (unsigned i = 0, e = vi.Kills.size(); i != e; ++i) {
      MachineInstr *Kill = vi.Kills[i];
      SlotIndex Start = getMBBStartIdx(Kill->getParent());
      SlotIndex killIdx = getInstructionIndex(Kill).getRegSlot();

      // Create interval with one of a NEW value number.  Note that this value
      // number isn't actually defined by an instruction, weird huh? :)
      if (PHIJoin) {
        assert(getInstructionFromIndex(Start) == 0 &&
               "PHI def index points at actual instruction.");
        ValNo = interval.getNextValue(Start, VNInfoAllocator);
      }
      LiveRange LR(Start, killIdx, ValNo);
      interval.addRange(LR);
      DEBUG(dbgs() << " +" << LR);
    }

  } else {
    if (MultipleDefsBySameMI(*mi, MOIdx))
      // Multiple defs of the same virtual register by the same instruction.
      // e.g. %reg1031:5<def>, %reg1031:6<def> = VLD1q16 %reg1024<kill>, ...
      // This is likely due to elimination of REG_SEQUENCE instructions. Return
      // here since there is nothing to do.
      return;

    // If this is the second time we see a virtual register definition, it
    // must be due to phi elimination or two addr elimination.  If this is
    // the result of two address elimination, then the vreg is one of the
    // def-and-use register operand.

    // It may also be partial redef like this:
    // 80  %reg1041:6<def> = VSHRNv4i16 %reg1034<kill>, 12, pred:14, pred:%reg0
    // 120 %reg1041:5<def> = VSHRNv4i16 %reg1039<kill>, 12, pred:14, pred:%reg0
    bool PartReDef = isPartialRedef(MIIdx, MO, interval);
    if (PartReDef || mi->isRegTiedToUseOperand(MOIdx)) {
      // If this is a two-address definition, then we have already processed
      // the live range.  The only problem is that we didn't realize there
      // are actually two values in the live interval.  Because of this we
      // need to take the LiveRegion that defines this register and split it
      // into two values.
      SlotIndex RedefIndex = MIIdx.getRegSlot(MO.isEarlyClobber());

      const LiveRange *OldLR =
        interval.getLiveRangeContaining(RedefIndex.getRegSlot(true));
      VNInfo *OldValNo = OldLR->valno;
      SlotIndex DefIndex = OldValNo->def.getRegSlot();

      // Delete the previous value, which should be short and continuous,
      // because the 2-addr copy must be in the same MBB as the redef.
      interval.removeRange(DefIndex, RedefIndex);

      // The new value number (#1) is defined by the instruction we claimed
      // defined value #0.
      VNInfo *ValNo = interval.createValueCopy(OldValNo, VNInfoAllocator);

      // Value#0 is now defined by the 2-addr instruction.
      OldValNo->def = RedefIndex;

      // Add the new live interval which replaces the range for the input copy.
      LiveRange LR(DefIndex, RedefIndex, ValNo);
      DEBUG(dbgs() << " replace range with " << LR);
      interval.addRange(LR);

      // If this redefinition is dead, we need to add a dummy unit live
      // range covering the def slot.
      if (MO.isDead())
        interval.addRange(LiveRange(RedefIndex, RedefIndex.getDeadSlot(),
                                    OldValNo));

      DEBUG(dbgs() << " RESULT: " << interval);
    } else if (LV->isPHIJoin(interval.reg)) {
      // In the case of PHI elimination, each variable definition is only
      // live until the end of the block.  We've already taken care of the
      // rest of the live range.

      SlotIndex defIndex = MIIdx.getRegSlot();
      if (MO.isEarlyClobber())
        defIndex = MIIdx.getRegSlot(true);

      VNInfo *ValNo = interval.getNextValue(defIndex, VNInfoAllocator);

      SlotIndex killIndex = getMBBEndIdx(mbb);
      LiveRange LR(defIndex, killIndex, ValNo);
      interval.addRange(LR);
      DEBUG(dbgs() << " phi-join +" << LR);
    } else {
      llvm_unreachable("Multiply defined register");
    }
  }

  DEBUG(dbgs() << '\n');
}

void LiveIntervals::handleRegisterDef(MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator MI,
                                      SlotIndex MIIdx,
                                      MachineOperand& MO,
                                      unsigned MOIdx) {
  if (TargetRegisterInfo::isVirtualRegister(MO.getReg()))
    handleVirtualRegisterDef(MBB, MI, MIIdx, MO, MOIdx,
                             getOrCreateInterval(MO.getReg()));
}

/// computeIntervals - computes the live intervals for virtual
/// registers. for some ordering of the machine instructions [1,N] a
/// live interval is an interval [i, j) where 1 <= i <= j < N for
/// which a variable is live
void LiveIntervals::computeIntervals() {
  DEBUG(dbgs() << "********** COMPUTING LIVE INTERVALS **********\n"
               << "********** Function: " << MF->getName() << '\n');

  RegMaskBlocks.resize(MF->getNumBlockIDs());

  SmallVector<unsigned, 8> UndefUses;
  for (MachineFunction::iterator MBBI = MF->begin(), E = MF->end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    RegMaskBlocks[MBB->getNumber()].first = RegMaskSlots.size();

    if (MBB->empty())
      continue;

    // Track the index of the current machine instr.
    SlotIndex MIIndex = getMBBStartIdx(MBB);
    DEBUG(dbgs() << "BB#" << MBB->getNumber()
          << ":\t\t# derived from " << MBB->getName() << "\n");

    // Skip over empty initial indices.
    if (getInstructionFromIndex(MIIndex) == 0)
      MIIndex = Indexes->getNextNonNullIndex(MIIndex);

    for (MachineBasicBlock::iterator MI = MBB->begin(), miEnd = MBB->end();
         MI != miEnd; ++MI) {
      DEBUG(dbgs() << MIIndex << "\t" << *MI);
      if (MI->isDebugValue())
        continue;
      assert(Indexes->getInstructionFromIndex(MIIndex) == MI &&
             "Lost SlotIndex synchronization");

      // Handle defs.
      for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
        MachineOperand &MO = MI->getOperand(i);

        // Collect register masks.
        if (MO.isRegMask()) {
          RegMaskSlots.push_back(MIIndex.getRegSlot());
          RegMaskBits.push_back(MO.getRegMask());
          continue;
        }

        if (!MO.isReg() || !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
          continue;

        // handle register defs - build intervals
        if (MO.isDef())
          handleRegisterDef(MBB, MI, MIIndex, MO, i);
        else if (MO.isUndef())
          UndefUses.push_back(MO.getReg());
      }

      // Move to the next instr slot.
      MIIndex = Indexes->getNextNonNullIndex(MIIndex);
    }

    // Compute the number of register mask instructions in this block.
    std::pair<unsigned, unsigned> &RMB = RegMaskBlocks[MBB->getNumber()];
    RMB.second = RegMaskSlots.size() - RMB.first;
  }

  // Create empty intervals for registers defined by implicit_def's (except
  // for those implicit_def that define values which are liveout of their
  // blocks.
  for (unsigned i = 0, e = UndefUses.size(); i != e; ++i) {
    unsigned UndefReg = UndefUses[i];
    (void)getOrCreateInterval(UndefReg);
  }
}

LiveInterval* LiveIntervals::createInterval(unsigned reg) {
  float Weight = TargetRegisterInfo::isPhysicalRegister(reg) ? HUGE_VALF : 0.0F;
  return new LiveInterval(reg, Weight);
}


/// computeVirtRegInterval - Compute the live interval of a virtual register,
/// based on defs and uses.
void LiveIntervals::computeVirtRegInterval(LiveInterval *LI) {
  assert(LRCalc && "LRCalc not initialized.");
  assert(LI->empty() && "Should only compute empty intervals.");
  LRCalc->reset(MF, getSlotIndexes(), DomTree, &getVNInfoAllocator());
  LRCalc->createDeadDefs(LI);
  LRCalc->extendToUses(LI);
}

void LiveIntervals::computeVirtRegs() {
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    LiveInterval *LI = createInterval(Reg);
    VirtRegIntervals[Reg] = LI;
    computeVirtRegInterval(LI);
  }
}

void LiveIntervals::computeRegMasks() {
  RegMaskBlocks.resize(MF->getNumBlockIDs());

  // Find all instructions with regmask operands.
  for (MachineFunction::iterator MBBI = MF->begin(), E = MF->end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    std::pair<unsigned, unsigned> &RMB = RegMaskBlocks[MBB->getNumber()];
    RMB.first = RegMaskSlots.size();
    for (MachineBasicBlock::iterator MI = MBB->begin(), ME = MBB->end();
         MI != ME; ++MI)
      for (MIOperands MO(MI); MO.isValid(); ++MO) {
        if (!MO->isRegMask())
          continue;
          RegMaskSlots.push_back(Indexes->getInstructionIndex(MI).getRegSlot());
          RegMaskBits.push_back(MO->getRegMask());
      }
    // Compute the number of register mask instructions in this block.
    RMB.second = RegMaskSlots.size() - RMB.first;
  }
}

//===----------------------------------------------------------------------===//
//                           Register Unit Liveness
//===----------------------------------------------------------------------===//
//
// Fixed interference typically comes from ABI boundaries: Function arguments
// and return values are passed in fixed registers, and so are exception
// pointers entering landing pads. Certain instructions require values to be
// present in specific registers. That is also represented through fixed
// interference.
//

/// computeRegUnitInterval - Compute the live interval of a register unit, based
/// on the uses and defs of aliasing registers.  The interval should be empty,
/// or contain only dead phi-defs from ABI blocks.
void LiveIntervals::computeRegUnitInterval(LiveInterval *LI) {
  unsigned Unit = LI->reg;

  assert(LRCalc && "LRCalc not initialized.");
  LRCalc->reset(MF, getSlotIndexes(), DomTree, &getVNInfoAllocator());

  // The physregs aliasing Unit are the roots and their super-registers.
  // Create all values as dead defs before extending to uses. Note that roots
  // may share super-registers. That's OK because createDeadDefs() is
  // idempotent. It is very rare for a register unit to have multiple roots, so
  // uniquing super-registers is probably not worthwhile.
  for (MCRegUnitRootIterator Roots(Unit, TRI); Roots.isValid(); ++Roots) {
    unsigned Root = *Roots;
    if (!MRI->reg_empty(Root))
      LRCalc->createDeadDefs(LI, Root);
    for (MCSuperRegIterator Supers(Root, TRI); Supers.isValid(); ++Supers) {
      if (!MRI->reg_empty(*Supers))
        LRCalc->createDeadDefs(LI, *Supers);
    }
  }

  // Now extend LI to reach all uses.
  // Ignore uses of reserved registers. We only track defs of those.
  for (MCRegUnitRootIterator Roots(Unit, TRI); Roots.isValid(); ++Roots) {
    unsigned Root = *Roots;
    if (!MRI->isReserved(Root) && !MRI->reg_empty(Root))
      LRCalc->extendToUses(LI, Root);
    for (MCSuperRegIterator Supers(Root, TRI); Supers.isValid(); ++Supers) {
      unsigned Reg = *Supers;
      if (!MRI->isReserved(Reg) && !MRI->reg_empty(Reg))
        LRCalc->extendToUses(LI, Reg);
    }
  }
}


/// computeLiveInRegUnits - Precompute the live ranges of any register units
/// that are live-in to an ABI block somewhere. Register values can appear
/// without a corresponding def when entering the entry block or a landing pad.
///
void LiveIntervals::computeLiveInRegUnits() {
  RegUnitIntervals.resize(TRI->getNumRegUnits());
  DEBUG(dbgs() << "Computing live-in reg-units in ABI blocks.\n");

  // Keep track of the intervals allocated.
  SmallVector<LiveInterval*, 8> NewIntvs;

  // Check all basic blocks for live-ins.
  for (MachineFunction::const_iterator MFI = MF->begin(), MFE = MF->end();
       MFI != MFE; ++MFI) {
    const MachineBasicBlock *MBB = MFI;

    // We only care about ABI blocks: Entry + landing pads.
    if ((MFI != MF->begin() && !MBB->isLandingPad()) || MBB->livein_empty())
      continue;

    // Create phi-defs at Begin for all live-in registers.
    SlotIndex Begin = Indexes->getMBBStartIdx(MBB);
    DEBUG(dbgs() << Begin << "\tBB#" << MBB->getNumber());
    for (MachineBasicBlock::livein_iterator LII = MBB->livein_begin(),
         LIE = MBB->livein_end(); LII != LIE; ++LII) {
      for (MCRegUnitIterator Units(*LII, TRI); Units.isValid(); ++Units) {
        unsigned Unit = *Units;
        LiveInterval *Intv = RegUnitIntervals[Unit];
        if (!Intv) {
          Intv = RegUnitIntervals[Unit] = new LiveInterval(Unit, HUGE_VALF);
          NewIntvs.push_back(Intv);
        }
        VNInfo *VNI = Intv->createDeadDef(Begin, getVNInfoAllocator());
        (void)VNI;
        DEBUG(dbgs() << ' ' << PrintRegUnit(Unit, TRI) << '#' << VNI->id);
      }
    }
    DEBUG(dbgs() << '\n');
  }
  DEBUG(dbgs() << "Created " << NewIntvs.size() << " new intervals.\n");

  // Compute the 'normal' part of the intervals.
  for (unsigned i = 0, e = NewIntvs.size(); i != e; ++i)
    computeRegUnitInterval(NewIntvs[i]);
}


/// shrinkToUses - After removing some uses of a register, shrink its live
/// range to just the remaining uses. This method does not compute reaching
/// defs for new uses, and it doesn't remove dead defs.
bool LiveIntervals::shrinkToUses(LiveInterval *li,
                                 SmallVectorImpl<MachineInstr*> *dead) {
  DEBUG(dbgs() << "Shrink: " << *li << '\n');
  assert(TargetRegisterInfo::isVirtualRegister(li->reg)
         && "Can only shrink virtual registers");
  // Find all the values used, including PHI kills.
  SmallVector<std::pair<SlotIndex, VNInfo*>, 16> WorkList;

  // Blocks that have already been added to WorkList as live-out.
  SmallPtrSet<MachineBasicBlock*, 16> LiveOut;

  // Visit all instructions reading li->reg.
  for (MachineRegisterInfo::reg_iterator I = MRI->reg_begin(li->reg);
       MachineInstr *UseMI = I.skipInstruction();) {
    if (UseMI->isDebugValue() || !UseMI->readsVirtualRegister(li->reg))
      continue;
    SlotIndex Idx = getInstructionIndex(UseMI).getRegSlot();
    LiveRangeQuery LRQ(*li, Idx);
    VNInfo *VNI = LRQ.valueIn();
    if (!VNI) {
      // This shouldn't happen: readsVirtualRegister returns true, but there is
      // no live value. It is likely caused by a target getting <undef> flags
      // wrong.
      DEBUG(dbgs() << Idx << '\t' << *UseMI
                   << "Warning: Instr claims to read non-existent value in "
                    << *li << '\n');
      continue;
    }
    // Special case: An early-clobber tied operand reads and writes the
    // register one slot early.
    if (VNInfo *DefVNI = LRQ.valueDefined())
      Idx = DefVNI->def;

    WorkList.push_back(std::make_pair(Idx, VNI));
  }

  // Create a new live interval with only minimal live segments per def.
  LiveInterval NewLI(li->reg, 0);
  for (LiveInterval::vni_iterator I = li->vni_begin(), E = li->vni_end();
       I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->isUnused())
      continue;
    NewLI.addRange(LiveRange(VNI->def, VNI->def.getDeadSlot(), VNI));
  }

  // Keep track of the PHIs that are in use.
  SmallPtrSet<VNInfo*, 8> UsedPHIs;

  // Extend intervals to reach all uses in WorkList.
  while (!WorkList.empty()) {
    SlotIndex Idx = WorkList.back().first;
    VNInfo *VNI = WorkList.back().second;
    WorkList.pop_back();
    const MachineBasicBlock *MBB = getMBBFromIndex(Idx.getPrevSlot());
    SlotIndex BlockStart = getMBBStartIdx(MBB);

    // Extend the live range for VNI to be live at Idx.
    if (VNInfo *ExtVNI = NewLI.extendInBlock(BlockStart, Idx)) {
      (void)ExtVNI;
      assert(ExtVNI == VNI && "Unexpected existing value number");
      // Is this a PHIDef we haven't seen before?
      if (!VNI->isPHIDef() || VNI->def != BlockStart || !UsedPHIs.insert(VNI))
        continue;
      // The PHI is live, make sure the predecessors are live-out.
      for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
           PE = MBB->pred_end(); PI != PE; ++PI) {
        if (!LiveOut.insert(*PI))
          continue;
        SlotIndex Stop = getMBBEndIdx(*PI);
        // A predecessor is not required to have a live-out value for a PHI.
        if (VNInfo *PVNI = li->getVNInfoBefore(Stop))
          WorkList.push_back(std::make_pair(Stop, PVNI));
      }
      continue;
    }

    // VNI is live-in to MBB.
    DEBUG(dbgs() << " live-in at " << BlockStart << '\n');
    NewLI.addRange(LiveRange(BlockStart, Idx, VNI));

    // Make sure VNI is live-out from the predecessors.
    for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
      if (!LiveOut.insert(*PI))
        continue;
      SlotIndex Stop = getMBBEndIdx(*PI);
      assert(li->getVNInfoBefore(Stop) == VNI &&
             "Wrong value out of predecessor");
      WorkList.push_back(std::make_pair(Stop, VNI));
    }
  }

  // Handle dead values.
  bool CanSeparate = false;
  for (LiveInterval::vni_iterator I = li->vni_begin(), E = li->vni_end();
       I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->isUnused())
      continue;
    LiveInterval::iterator LII = NewLI.FindLiveRangeContaining(VNI->def);
    assert(LII != NewLI.end() && "Missing live range for PHI");
    if (LII->end != VNI->def.getDeadSlot())
      continue;
    if (VNI->isPHIDef()) {
      // This is a dead PHI. Remove it.
      VNI->markUnused();
      NewLI.removeRange(*LII);
      DEBUG(dbgs() << "Dead PHI at " << VNI->def << " may separate interval\n");
      CanSeparate = true;
    } else {
      // This is a dead def. Make sure the instruction knows.
      MachineInstr *MI = getInstructionFromIndex(VNI->def);
      assert(MI && "No instruction defining live value");
      MI->addRegisterDead(li->reg, TRI);
      if (dead && MI->allDefsAreDead()) {
        DEBUG(dbgs() << "All defs dead: " << VNI->def << '\t' << *MI);
        dead->push_back(MI);
      }
    }
  }

  // Move the trimmed ranges back.
  li->ranges.swap(NewLI.ranges);
  DEBUG(dbgs() << "Shrunk: " << *li << '\n');
  return CanSeparate;
}

void LiveIntervals::extendToIndices(LiveInterval *LI,
                                    ArrayRef<SlotIndex> Indices) {
  assert(LRCalc && "LRCalc not initialized.");
  LRCalc->reset(MF, getSlotIndexes(), DomTree, &getVNInfoAllocator());
  for (unsigned i = 0, e = Indices.size(); i != e; ++i)
    LRCalc->extend(LI, Indices[i]);
}

void LiveIntervals::pruneValue(LiveInterval *LI, SlotIndex Kill,
                               SmallVectorImpl<SlotIndex> *EndPoints) {
  LiveRangeQuery LRQ(*LI, Kill);
  VNInfo *VNI = LRQ.valueOut();
  if (!VNI)
    return;

  MachineBasicBlock *KillMBB = Indexes->getMBBFromIndex(Kill);
  SlotIndex MBBStart, MBBEnd;
  tie(MBBStart, MBBEnd) = Indexes->getMBBRange(KillMBB);

  // If VNI isn't live out from KillMBB, the value is trivially pruned.
  if (LRQ.endPoint() < MBBEnd) {
    LI->removeRange(Kill, LRQ.endPoint());
    if (EndPoints) EndPoints->push_back(LRQ.endPoint());
    return;
  }

  // VNI is live out of KillMBB.
  LI->removeRange(Kill, MBBEnd);
  if (EndPoints) EndPoints->push_back(MBBEnd);

  // Find all blocks that are reachable from KillMBB without leaving VNI's live
  // range. It is possible that KillMBB itself is reachable, so start a DFS
  // from each successor.
  typedef SmallPtrSet<MachineBasicBlock*, 9> VisitedTy;
  VisitedTy Visited;
  for (MachineBasicBlock::succ_iterator
       SuccI = KillMBB->succ_begin(), SuccE = KillMBB->succ_end();
       SuccI != SuccE; ++SuccI) {
    for (df_ext_iterator<MachineBasicBlock*, VisitedTy>
         I = df_ext_begin(*SuccI, Visited), E = df_ext_end(*SuccI, Visited);
         I != E;) {
      MachineBasicBlock *MBB = *I;

      // Check if VNI is live in to MBB.
      tie(MBBStart, MBBEnd) = Indexes->getMBBRange(MBB);
      LiveRangeQuery LRQ(*LI, MBBStart);
      if (LRQ.valueIn() != VNI) {
        // This block isn't part of the VNI live range. Prune the search.
        I.skipChildren();
        continue;
      }

      // Prune the search if VNI is killed in MBB.
      if (LRQ.endPoint() < MBBEnd) {
        LI->removeRange(MBBStart, LRQ.endPoint());
        if (EndPoints) EndPoints->push_back(LRQ.endPoint());
        I.skipChildren();
        continue;
      }

      // VNI is live through MBB.
      LI->removeRange(MBBStart, MBBEnd);
      if (EndPoints) EndPoints->push_back(MBBEnd);
      ++I;
    }
  }
}

//===----------------------------------------------------------------------===//
// Register allocator hooks.
//

void LiveIntervals::addKillFlags(const VirtRegMap *VRM) {
  // Keep track of regunit ranges.
  SmallVector<std::pair<LiveInterval*, LiveInterval::iterator>, 8> RU;

  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    LiveInterval *LI = &getInterval(Reg);
    if (LI->empty())
      continue;

    // Find the regunit intervals for the assigned register. They may overlap
    // the virtual register live range, cancelling any kills.
    RU.clear();
    for (MCRegUnitIterator Units(VRM->getPhys(Reg), TRI); Units.isValid();
         ++Units) {
      LiveInterval *RUInt = &getRegUnit(*Units);
      if (RUInt->empty())
        continue;
      RU.push_back(std::make_pair(RUInt, RUInt->find(LI->begin()->end)));
    }

    // Every instruction that kills Reg corresponds to a live range end point.
    for (LiveInterval::iterator RI = LI->begin(), RE = LI->end(); RI != RE;
         ++RI) {
      // A block index indicates an MBB edge.
      if (RI->end.isBlock())
        continue;
      MachineInstr *MI = getInstructionFromIndex(RI->end);
      if (!MI)
        continue;

      // Check if any of the reguints are live beyond the end of RI. That could
      // happen when a physreg is defined as a copy of a virtreg:
      //
      //   %EAX = COPY %vreg5
      //   FOO %vreg5         <--- MI, cancel kill because %EAX is live.
      //   BAR %EAX<kill>
      //
      // There should be no kill flag on FOO when %vreg5 is rewritten as %EAX.
      bool CancelKill = false;
      for (unsigned u = 0, e = RU.size(); u != e; ++u) {
        LiveInterval *RInt = RU[u].first;
        LiveInterval::iterator &I = RU[u].second;
        if (I == RInt->end())
          continue;
        I = RInt->advanceTo(I, RI->end);
        if (I == RInt->end() || I->start >= RI->end)
          continue;
        // I is overlapping RI.
        CancelKill = true;
        break;
      }
      if (CancelKill)
        MI->clearRegisterKills(Reg, NULL);
      else
        MI->addRegisterKilled(Reg, NULL);
    }
  }
}

MachineBasicBlock*
LiveIntervals::intervalIsInOneMBB(const LiveInterval &LI) const {
  // A local live range must be fully contained inside the block, meaning it is
  // defined and killed at instructions, not at block boundaries. It is not
  // live in or or out of any block.
  //
  // It is technically possible to have a PHI-defined live range identical to a
  // single block, but we are going to return false in that case.

  SlotIndex Start = LI.beginIndex();
  if (Start.isBlock())
    return NULL;

  SlotIndex Stop = LI.endIndex();
  if (Stop.isBlock())
    return NULL;

  // getMBBFromIndex doesn't need to search the MBB table when both indexes
  // belong to proper instructions.
  MachineBasicBlock *MBB1 = Indexes->getMBBFromIndex(Start);
  MachineBasicBlock *MBB2 = Indexes->getMBBFromIndex(Stop);
  return MBB1 == MBB2 ? MBB1 : NULL;
}

bool
LiveIntervals::hasPHIKill(const LiveInterval &LI, const VNInfo *VNI) const {
  for (LiveInterval::const_vni_iterator I = LI.vni_begin(), E = LI.vni_end();
       I != E; ++I) {
    const VNInfo *PHI = *I;
    if (PHI->isUnused() || !PHI->isPHIDef())
      continue;
    const MachineBasicBlock *PHIMBB = getMBBFromIndex(PHI->def);
    // Conservatively return true instead of scanning huge predecessor lists.
    if (PHIMBB->pred_size() > 100)
      return true;
    for (MachineBasicBlock::const_pred_iterator
         PI = PHIMBB->pred_begin(), PE = PHIMBB->pred_end(); PI != PE; ++PI)
      if (VNI == LI.getVNInfoBefore(Indexes->getMBBEndIdx(*PI)))
        return true;
  }
  return false;
}

float
LiveIntervals::getSpillWeight(bool isDef, bool isUse, unsigned loopDepth) {
  // Limit the loop depth ridiculousness.
  if (loopDepth > 200)
    loopDepth = 200;

  // The loop depth is used to roughly estimate the number of times the
  // instruction is executed. Something like 10^d is simple, but will quickly
  // overflow a float. This expression behaves like 10^d for small d, but is
  // more tempered for large d. At d=200 we get 6.7e33 which leaves a bit of
  // headroom before overflow.
  // By the way, powf() might be unavailable here. For consistency,
  // We may take pow(double,double).
  float lc = std::pow(1 + (100.0 / (loopDepth + 10)), (double)loopDepth);

  return (isDef + isUse) * lc;
}

LiveRange LiveIntervals::addLiveRangeToEndOfBlock(unsigned reg,
                                                  MachineInstr* startInst) {
  LiveInterval& Interval = getOrCreateInterval(reg);
  VNInfo* VN = Interval.getNextValue(
    SlotIndex(getInstructionIndex(startInst).getRegSlot()),
    getVNInfoAllocator());
  LiveRange LR(
     SlotIndex(getInstructionIndex(startInst).getRegSlot()),
     getMBBEndIdx(startInst->getParent()), VN);
  Interval.addRange(LR);

  return LR;
}


//===----------------------------------------------------------------------===//
//                          Register mask functions
//===----------------------------------------------------------------------===//

bool LiveIntervals::checkRegMaskInterference(LiveInterval &LI,
                                             BitVector &UsableRegs) {
  if (LI.empty())
    return false;
  LiveInterval::iterator LiveI = LI.begin(), LiveE = LI.end();

  // Use a smaller arrays for local live ranges.
  ArrayRef<SlotIndex> Slots;
  ArrayRef<const uint32_t*> Bits;
  if (MachineBasicBlock *MBB = intervalIsInOneMBB(LI)) {
    Slots = getRegMaskSlotsInBlock(MBB->getNumber());
    Bits = getRegMaskBitsInBlock(MBB->getNumber());
  } else {
    Slots = getRegMaskSlots();
    Bits = getRegMaskBits();
  }

  // We are going to enumerate all the register mask slots contained in LI.
  // Start with a binary search of RegMaskSlots to find a starting point.
  ArrayRef<SlotIndex>::iterator SlotI =
    std::lower_bound(Slots.begin(), Slots.end(), LiveI->start);
  ArrayRef<SlotIndex>::iterator SlotE = Slots.end();

  // No slots in range, LI begins after the last call.
  if (SlotI == SlotE)
    return false;

  bool Found = false;
  for (;;) {
    assert(*SlotI >= LiveI->start);
    // Loop over all slots overlapping this segment.
    while (*SlotI < LiveI->end) {
      // *SlotI overlaps LI. Collect mask bits.
      if (!Found) {
        // This is the first overlap. Initialize UsableRegs to all ones.
        UsableRegs.clear();
        UsableRegs.resize(TRI->getNumRegs(), true);
        Found = true;
      }
      // Remove usable registers clobbered by this mask.
      UsableRegs.clearBitsNotInMask(Bits[SlotI-Slots.begin()]);
      if (++SlotI == SlotE)
        return Found;
    }
    // *SlotI is beyond the current LI segment.
    LiveI = LI.advanceTo(LiveI, *SlotI);
    if (LiveI == LiveE)
      return Found;
    // Advance SlotI until it overlaps.
    while (*SlotI < LiveI->start)
      if (++SlotI == SlotE)
        return Found;
  }
}

//===----------------------------------------------------------------------===//
//                         IntervalUpdate class.
//===----------------------------------------------------------------------===//

// HMEditor is a toolkit used by handleMove to trim or extend live intervals.
class LiveIntervals::HMEditor {
private:
  LiveIntervals& LIS;
  const MachineRegisterInfo& MRI;
  const TargetRegisterInfo& TRI;
  SlotIndex OldIdx;
  SlotIndex NewIdx;
  SmallPtrSet<LiveInterval*, 8> Updated;
  bool UpdateFlags;

public:
  HMEditor(LiveIntervals& LIS, const MachineRegisterInfo& MRI,
           const TargetRegisterInfo& TRI,
           SlotIndex OldIdx, SlotIndex NewIdx, bool UpdateFlags)
    : LIS(LIS), MRI(MRI), TRI(TRI), OldIdx(OldIdx), NewIdx(NewIdx),
      UpdateFlags(UpdateFlags) {}

  // FIXME: UpdateFlags is a workaround that creates live intervals for all
  // physregs, even those that aren't needed for regalloc, in order to update
  // kill flags. This is wasteful. Eventually, LiveVariables will strip all kill
  // flags, and postRA passes will use a live register utility instead.
  LiveInterval *getRegUnitLI(unsigned Unit) {
    if (UpdateFlags)
      return &LIS.getRegUnit(Unit);
    return LIS.getCachedRegUnit(Unit);
  }

  /// Update all live ranges touched by MI, assuming a move from OldIdx to
  /// NewIdx.
  void updateAllRanges(MachineInstr *MI) {
    DEBUG(dbgs() << "handleMove " << OldIdx << " -> " << NewIdx << ": " << *MI);
    bool hasRegMask = false;
    for (MIOperands MO(MI); MO.isValid(); ++MO) {
      if (MO->isRegMask())
        hasRegMask = true;
      if (!MO->isReg())
        continue;
      // Aggressively clear all kill flags.
      // They are reinserted by VirtRegRewriter.
      if (MO->isUse())
        MO->setIsKill(false);

      unsigned Reg = MO->getReg();
      if (!Reg)
        continue;
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        updateRange(LIS.getInterval(Reg));
        continue;
      }

      // For physregs, only update the regunits that actually have a
      // precomputed live range.
      for (MCRegUnitIterator Units(Reg, &TRI); Units.isValid(); ++Units)
        if (LiveInterval *LI = getRegUnitLI(*Units))
          updateRange(*LI);
    }
    if (hasRegMask)
      updateRegMaskSlots();
  }

private:
  /// Update a single live range, assuming an instruction has been moved from
  /// OldIdx to NewIdx.
  void updateRange(LiveInterval &LI) {
    if (!Updated.insert(&LI))
      return;
    DEBUG({
      dbgs() << "     ";
      if (TargetRegisterInfo::isVirtualRegister(LI.reg))
        dbgs() << PrintReg(LI.reg);
      else
        dbgs() << PrintRegUnit(LI.reg, &TRI);
      dbgs() << ":\t" << LI << '\n';
    });
    if (SlotIndex::isEarlierInstr(OldIdx, NewIdx))
      handleMoveDown(LI);
    else
      handleMoveUp(LI);
    DEBUG(dbgs() << "        -->\t" << LI << '\n');
    LI.verify();
  }

  /// Update LI to reflect an instruction has been moved downwards from OldIdx
  /// to NewIdx.
  ///
  /// 1. Live def at OldIdx:
  ///    Move def to NewIdx, assert endpoint after NewIdx.
  ///
  /// 2. Live def at OldIdx, killed at NewIdx:
  ///    Change to dead def at NewIdx.
  ///    (Happens when bundling def+kill together).
  ///
  /// 3. Dead def at OldIdx:
  ///    Move def to NewIdx, possibly across another live value.
  ///
  /// 4. Def at OldIdx AND at NewIdx:
  ///    Remove live range [OldIdx;NewIdx) and value defined at OldIdx.
  ///    (Happens when bundling multiple defs together).
  ///
  /// 5. Value read at OldIdx, killed before NewIdx:
  ///    Extend kill to NewIdx.
  ///
  void handleMoveDown(LiveInterval &LI) {
    // First look for a kill at OldIdx.
    LiveInterval::iterator I = LI.find(OldIdx.getBaseIndex());
    LiveInterval::iterator E = LI.end();
    // Is LI even live at OldIdx?
    if (I == E || SlotIndex::isEarlierInstr(OldIdx, I->start))
      return;

    // Handle a live-in value.
    if (!SlotIndex::isSameInstr(I->start, OldIdx)) {
      bool isKill = SlotIndex::isSameInstr(OldIdx, I->end);
      // If the live-in value already extends to NewIdx, there is nothing to do.
      if (!SlotIndex::isEarlierInstr(I->end, NewIdx))
        return;
      // Aggressively remove all kill flags from the old kill point.
      // Kill flags shouldn't be used while live intervals exist, they will be
      // reinserted by VirtRegRewriter.
      if (MachineInstr *KillMI = LIS.getInstructionFromIndex(I->end))
        for (MIBundleOperands MO(KillMI); MO.isValid(); ++MO)
          if (MO->isReg() && MO->isUse())
            MO->setIsKill(false);
      // Adjust I->end to reach NewIdx. This may temporarily make LI invalid by
      // overlapping ranges. Case 5 above.
      I->end = NewIdx.getRegSlot(I->end.isEarlyClobber());
      // If this was a kill, there may also be a def. Otherwise we're done.
      if (!isKill)
        return;
      ++I;
    }

    // Check for a def at OldIdx.
    if (I == E || !SlotIndex::isSameInstr(OldIdx, I->start))
      return;
    // We have a def at OldIdx.
    VNInfo *DefVNI = I->valno;
    assert(DefVNI->def == I->start && "Inconsistent def");
    DefVNI->def = NewIdx.getRegSlot(I->start.isEarlyClobber());
    // If the defined value extends beyond NewIdx, just move the def down.
    // This is case 1 above.
    if (SlotIndex::isEarlierInstr(NewIdx, I->end)) {
      I->start = DefVNI->def;
      return;
    }
    // The remaining possibilities are now:
    // 2. Live def at OldIdx, killed at NewIdx: isSameInstr(I->end, NewIdx).
    // 3. Dead def at OldIdx: I->end = OldIdx.getDeadSlot().
    // In either case, it is possible that there is an existing def at NewIdx.
    assert((I->end == OldIdx.getDeadSlot() ||
            SlotIndex::isSameInstr(I->end, NewIdx)) &&
            "Cannot move def below kill");
    LiveInterval::iterator NewI = LI.advanceTo(I, NewIdx.getRegSlot());
    if (NewI != E && SlotIndex::isSameInstr(NewI->start, NewIdx)) {
      // There is an existing def at NewIdx, case 4 above. The def at OldIdx is
      // coalesced into that value.
      assert(NewI->valno != DefVNI && "Multiple defs of value?");
      LI.removeValNo(DefVNI);
      return;
    }
    // There was no existing def at NewIdx. Turn *I into a dead def at NewIdx.
    // If the def at OldIdx was dead, we allow it to be moved across other LI
    // values. The new range should be placed immediately before NewI, move any
    // intermediate ranges up.
    assert(NewI != I && "Inconsistent iterators");
    std::copy(llvm::next(I), NewI, I);
    *llvm::prior(NewI) = LiveRange(DefVNI->def, NewIdx.getDeadSlot(), DefVNI);
  }

  /// Update LI to reflect an instruction has been moved upwards from OldIdx
  /// to NewIdx.
  ///
  /// 1. Live def at OldIdx:
  ///    Hoist def to NewIdx.
  ///
  /// 2. Dead def at OldIdx:
  ///    Hoist def+end to NewIdx, possibly move across other values.
  ///
  /// 3. Dead def at OldIdx AND existing def at NewIdx:
  ///    Remove value defined at OldIdx, coalescing it with existing value.
  ///
  /// 4. Live def at OldIdx AND existing def at NewIdx:
  ///    Remove value defined at NewIdx, hoist OldIdx def to NewIdx.
  ///    (Happens when bundling multiple defs together).
  ///
  /// 5. Value killed at OldIdx:
  ///    Hoist kill to NewIdx, then scan for last kill between NewIdx and
  ///    OldIdx.
  ///
  void handleMoveUp(LiveInterval &LI) {
    // First look for a kill at OldIdx.
    LiveInterval::iterator I = LI.find(OldIdx.getBaseIndex());
    LiveInterval::iterator E = LI.end();
    // Is LI even live at OldIdx?
    if (I == E || SlotIndex::isEarlierInstr(OldIdx, I->start))
      return;

    // Handle a live-in value.
    if (!SlotIndex::isSameInstr(I->start, OldIdx)) {
      // If the live-in value isn't killed here, there is nothing to do.
      if (!SlotIndex::isSameInstr(OldIdx, I->end))
        return;
      // Adjust I->end to end at NewIdx. If we are hoisting a kill above
      // another use, we need to search for that use. Case 5 above.
      I->end = NewIdx.getRegSlot(I->end.isEarlyClobber());
      ++I;
      // If OldIdx also defines a value, there couldn't have been another use.
      if (I == E || !SlotIndex::isSameInstr(I->start, OldIdx)) {
        // No def, search for the new kill.
        // This can never be an early clobber kill since there is no def.
        llvm::prior(I)->end = findLastUseBefore(LI.reg).getRegSlot();
        return;
      }
    }

    // Now deal with the def at OldIdx.
    assert(I != E && SlotIndex::isSameInstr(I->start, OldIdx) && "No def?");
    VNInfo *DefVNI = I->valno;
    assert(DefVNI->def == I->start && "Inconsistent def");
    DefVNI->def = NewIdx.getRegSlot(I->start.isEarlyClobber());

    // Check for an existing def at NewIdx.
    LiveInterval::iterator NewI = LI.find(NewIdx.getRegSlot());
    if (SlotIndex::isSameInstr(NewI->start, NewIdx)) {
      assert(NewI->valno != DefVNI && "Same value defined more than once?");
      // There is an existing def at NewIdx.
      if (I->end.isDead()) {
        // Case 3: Remove the dead def at OldIdx.
        LI.removeValNo(DefVNI);
        return;
      }
      // Case 4: Replace def at NewIdx with live def at OldIdx.
      I->start = DefVNI->def;
      LI.removeValNo(NewI->valno);
      return;
    }

    // There is no existing def at NewIdx. Hoist DefVNI.
    if (!I->end.isDead()) {
      // Leave the end point of a live def.
      I->start = DefVNI->def;
      return;
    }

    // DefVNI is a dead def. It may have been moved across other values in LI,
    // so move I up to NewI. Slide [NewI;I) down one position.
    std::copy_backward(NewI, I, llvm::next(I));
    *NewI = LiveRange(DefVNI->def, NewIdx.getDeadSlot(), DefVNI);
  }

  void updateRegMaskSlots() {
    SmallVectorImpl<SlotIndex>::iterator RI =
      std::lower_bound(LIS.RegMaskSlots.begin(), LIS.RegMaskSlots.end(),
                       OldIdx);
    assert(RI != LIS.RegMaskSlots.end() && *RI == OldIdx.getRegSlot() &&
           "No RegMask at OldIdx.");
    *RI = NewIdx.getRegSlot();
    assert((RI == LIS.RegMaskSlots.begin() ||
            SlotIndex::isEarlierInstr(*llvm::prior(RI), *RI)) &&
            "Cannot move regmask instruction above another call");
    assert((llvm::next(RI) == LIS.RegMaskSlots.end() ||
            SlotIndex::isEarlierInstr(*RI, *llvm::next(RI))) &&
            "Cannot move regmask instruction below another call");
  }

  // Return the last use of reg between NewIdx and OldIdx.
  SlotIndex findLastUseBefore(unsigned Reg) {
    SlotIndex LastUse = NewIdx;

    if (TargetRegisterInfo::isVirtualRegister(Reg)) {
      for (MachineRegisterInfo::use_nodbg_iterator
             UI = MRI.use_nodbg_begin(Reg),
             UE = MRI.use_nodbg_end();
           UI != UE; UI.skipInstruction()) {
        const MachineInstr* MI = &*UI;
        SlotIndex InstSlot = LIS.getSlotIndexes()->getInstructionIndex(MI);
        if (InstSlot > LastUse && InstSlot < OldIdx)
          LastUse = InstSlot;
      }
    } else {
      MachineInstr* MI = LIS.getSlotIndexes()->getInstructionFromIndex(NewIdx);
      MachineBasicBlock::iterator MII(MI);
      ++MII;
      MachineBasicBlock* MBB = MI->getParent();
      for (; MII != MBB->end() && LIS.getInstructionIndex(MII) < OldIdx; ++MII){
        for (MachineInstr::mop_iterator MOI = MII->operands_begin(),
                                        MOE = MII->operands_end();
             MOI != MOE; ++MOI) {
          const MachineOperand& mop = *MOI;
          if (!mop.isReg() || mop.getReg() == 0 ||
              TargetRegisterInfo::isVirtualRegister(mop.getReg()))
            continue;

          if (TRI.hasRegUnit(mop.getReg(), Reg))
            LastUse = LIS.getInstructionIndex(MII);
        }
      }
    }
    return LastUse;
  }
};

void LiveIntervals::handleMove(MachineInstr* MI, bool UpdateFlags) {
  assert(!MI->isBundled() && "Can't handle bundled instructions yet.");
  SlotIndex OldIndex = Indexes->getInstructionIndex(MI);
  Indexes->removeMachineInstrFromMaps(MI);
  SlotIndex NewIndex = Indexes->insertMachineInstrInMaps(MI);
  assert(getMBBStartIdx(MI->getParent()) <= OldIndex &&
         OldIndex < getMBBEndIdx(MI->getParent()) &&
         "Cannot handle moves across basic block boundaries.");

  HMEditor HME(*this, *MRI, *TRI, OldIndex, NewIndex, UpdateFlags);
  HME.updateAllRanges(MI);
}

void LiveIntervals::handleMoveIntoBundle(MachineInstr* MI,
                                         MachineInstr* BundleStart,
                                         bool UpdateFlags) {
  SlotIndex OldIndex = Indexes->getInstructionIndex(MI);
  SlotIndex NewIndex = Indexes->getInstructionIndex(BundleStart);
  HMEditor HME(*this, *MRI, *TRI, OldIndex, NewIndex, UpdateFlags);
  HME.updateAllRanges(MI);
}
