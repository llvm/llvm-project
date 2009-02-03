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

#define DEBUG_TYPE "liveintervals"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "VirtRegMap.h"
#include "llvm/Value.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

// Hidden options for help debugging.
static cl::opt<bool> DisableReMat("disable-rematerialization", 
                                  cl::init(false), cl::Hidden);

static cl::opt<bool> SplitAtBB("split-intervals-at-bb", 
                               cl::init(true), cl::Hidden);
static cl::opt<int> SplitLimit("split-limit",
                               cl::init(-1), cl::Hidden);

static cl::opt<bool> EnableAggressiveRemat("aggressive-remat", cl::Hidden);

static cl::opt<bool> EnableFastSpilling("fast-spill",
                                        cl::init(false), cl::Hidden);

STATISTIC(numIntervals, "Number of original intervals");
STATISTIC(numFolds    , "Number of loads/stores folded into instructions");
STATISTIC(numSplits   , "Number of intervals split");

char LiveIntervals::ID = 0;
static RegisterPass<LiveIntervals> X("liveintervals", "Live Interval Analysis");

void LiveIntervals::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addPreserved<LiveVariables>();
  AU.addRequired<LiveVariables>();
  AU.addPreservedID(MachineLoopInfoID);
  AU.addPreservedID(MachineDominatorsID);
  
  if (!StrongPHIElim) {
    AU.addPreservedID(PHIEliminationID);
    AU.addRequiredID(PHIEliminationID);
  }
  
  AU.addRequiredID(TwoAddressInstructionPassID);
  MachineFunctionPass::getAnalysisUsage(AU);
}

void LiveIntervals::releaseMemory() {
  // Free the live intervals themselves.
  for (DenseMap<unsigned, LiveInterval*>::iterator I = r2iMap_.begin(),
       E = r2iMap_.end(); I != E; ++I)
    delete I->second;
  
  MBB2IdxMap.clear();
  Idx2MBBMap.clear();
  mi2iMap_.clear();
  i2miMap_.clear();
  r2iMap_.clear();
  // Release VNInfo memroy regions after all VNInfo objects are dtor'd.
  VNInfoAllocator.Reset();
  while (!ClonedMIs.empty()) {
    MachineInstr *MI = ClonedMIs.back();
    ClonedMIs.pop_back();
    mf_->DeleteMachineInstr(MI);
  }
}

void LiveIntervals::computeNumbering() {
  Index2MiMap OldI2MI = i2miMap_;
  std::vector<IdxMBBPair> OldI2MBB = Idx2MBBMap;
  
  Idx2MBBMap.clear();
  MBB2IdxMap.clear();
  mi2iMap_.clear();
  i2miMap_.clear();
  
  FunctionSize = 0;
  
  // Number MachineInstrs and MachineBasicBlocks.
  // Initialize MBB indexes to a sentinal.
  MBB2IdxMap.resize(mf_->getNumBlockIDs(), std::make_pair(~0U,~0U));
  
  unsigned MIIndex = 0;
  for (MachineFunction::iterator MBB = mf_->begin(), E = mf_->end();
       MBB != E; ++MBB) {
    unsigned StartIdx = MIIndex;

    // Insert an empty slot at the beginning of each block.
    MIIndex += InstrSlots::NUM;
    i2miMap_.push_back(0);

    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      bool inserted = mi2iMap_.insert(std::make_pair(I, MIIndex)).second;
      assert(inserted && "multiple MachineInstr -> index mappings");
      inserted = true;
      i2miMap_.push_back(I);
      MIIndex += InstrSlots::NUM;
      FunctionSize++;
      
      // Insert max(1, numdefs) empty slots after every instruction.
      unsigned Slots = I->getDesc().getNumDefs();
      if (Slots == 0)
        Slots = 1;
      MIIndex += InstrSlots::NUM * Slots;
      while (Slots--)
        i2miMap_.push_back(0);
    }
    
    // Set the MBB2IdxMap entry for this MBB.
    MBB2IdxMap[MBB->getNumber()] = std::make_pair(StartIdx, MIIndex - 1);
    Idx2MBBMap.push_back(std::make_pair(StartIdx, MBB));
  }
  std::sort(Idx2MBBMap.begin(), Idx2MBBMap.end(), Idx2MBBCompare());
  
  if (!OldI2MI.empty())
    for (iterator OI = begin(), OE = end(); OI != OE; ++OI) {
      for (LiveInterval::iterator LI = OI->second->begin(),
           LE = OI->second->end(); LI != LE; ++LI) {
        
        // Remap the start index of the live range to the corresponding new
        // number, or our best guess at what it _should_ correspond to if the
        // original instruction has been erased.  This is either the following
        // instruction or its predecessor.
        unsigned index = LI->start / InstrSlots::NUM;
        unsigned offset = LI->start % InstrSlots::NUM;
        if (offset == InstrSlots::LOAD) {
          std::vector<IdxMBBPair>::const_iterator I =
                  std::lower_bound(OldI2MBB.begin(), OldI2MBB.end(), LI->start);
          // Take the pair containing the index
          std::vector<IdxMBBPair>::const_iterator J =
                    (I == OldI2MBB.end() && OldI2MBB.size()>0) ? (I-1): I;
          
          LI->start = getMBBStartIdx(J->second);
        } else {
          LI->start = mi2iMap_[OldI2MI[index]] + offset;
        }
        
        // Remap the ending index in the same way that we remapped the start,
        // except for the final step where we always map to the immediately
        // following instruction.
        index = (LI->end - 1) / InstrSlots::NUM;
        offset  = LI->end % InstrSlots::NUM;
        if (offset == InstrSlots::LOAD) {
          // VReg dies at end of block.
          std::vector<IdxMBBPair>::const_iterator I =
                  std::lower_bound(OldI2MBB.begin(), OldI2MBB.end(), LI->end);
          --I;
          
          LI->end = getMBBEndIdx(I->second) + 1;
        } else {
          unsigned idx = index;
          while (index < OldI2MI.size() && !OldI2MI[index]) ++index;
          
          if (index != OldI2MI.size())
            LI->end = mi2iMap_[OldI2MI[index]] + (idx == index ? offset : 0);
          else
            LI->end = InstrSlots::NUM * i2miMap_.size();
        }
      }
      
      for (LiveInterval::vni_iterator VNI = OI->second->vni_begin(),
           VNE = OI->second->vni_end(); VNI != VNE; ++VNI) { 
        VNInfo* vni = *VNI;
        
        // Remap the VNInfo def index, which works the same as the
        // start indices above. VN's with special sentinel defs
        // don't need to be remapped.
        if (vni->def != ~0U && vni->def != ~1U) {
          unsigned index = vni->def / InstrSlots::NUM;
          unsigned offset = vni->def % InstrSlots::NUM;
          if (offset == InstrSlots::LOAD) {
            std::vector<IdxMBBPair>::const_iterator I =
                  std::lower_bound(OldI2MBB.begin(), OldI2MBB.end(), vni->def);
            // Take the pair containing the index
            std::vector<IdxMBBPair>::const_iterator J =
                    (I == OldI2MBB.end() && OldI2MBB.size()>0) ? (I-1): I;
          
            vni->def = getMBBStartIdx(J->second);
          } else {
            vni->def = mi2iMap_[OldI2MI[index]] + offset;
          }
        }
        
        // Remap the VNInfo kill indices, which works the same as
        // the end indices above.
        for (size_t i = 0; i < vni->kills.size(); ++i) {
          // PHI kills don't need to be remapped.
          if (!vni->kills[i]) continue;
          
          unsigned index = (vni->kills[i]-1) / InstrSlots::NUM;
          unsigned offset = vni->kills[i] % InstrSlots::NUM;
          if (offset == InstrSlots::LOAD) {
            std::vector<IdxMBBPair>::const_iterator I =
             std::lower_bound(OldI2MBB.begin(), OldI2MBB.end(), vni->kills[i]);
            --I;

            vni->kills[i] = getMBBEndIdx(I->second);
          } else {
            unsigned idx = index;
            while (index < OldI2MI.size() && !OldI2MI[index]) ++index;
            
            if (index != OldI2MI.size())
              vni->kills[i] = mi2iMap_[OldI2MI[index]] + 
                              (idx == index ? offset : 0);
            else
              vni->kills[i] = InstrSlots::NUM * i2miMap_.size();
          }
        }
      }
    }
}

/// runOnMachineFunction - Register allocate the whole function
///
bool LiveIntervals::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  mri_ = &mf_->getRegInfo();
  tm_ = &fn.getTarget();
  tri_ = tm_->getRegisterInfo();
  tii_ = tm_->getInstrInfo();
  aa_ = &getAnalysis<AliasAnalysis>();
  lv_ = &getAnalysis<LiveVariables>();
  allocatableRegs_ = tri_->getAllocatableSet(fn);

  computeNumbering();
  computeIntervals();

  numIntervals += getNumIntervals();

  DEBUG(dump());
  return true;
}

/// print - Implement the dump method.
void LiveIntervals::print(std::ostream &O, const Module* ) const {
  O << "********** INTERVALS **********\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    I->second->print(O, tri_);
    O << "\n";
  }

  O << "********** MACHINEINSTRS **********\n";
  for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
       mbbi != mbbe; ++mbbi) {
    O << ((Value*)mbbi->getBasicBlock())->getName() << ":\n";
    for (MachineBasicBlock::iterator mii = mbbi->begin(),
           mie = mbbi->end(); mii != mie; ++mii) {
      O << getInstructionIndex(mii) << '\t' << *mii;
    }
  }
}

/// conflictsWithPhysRegDef - Returns true if the specified register
/// is defined during the duration of the specified interval.
bool LiveIntervals::conflictsWithPhysRegDef(const LiveInterval &li,
                                            VirtRegMap &vrm, unsigned reg) {
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    for (unsigned index = getBaseIndex(I->start),
           end = getBaseIndex(I->end-1) + InstrSlots::NUM; index != end;
         index += InstrSlots::NUM) {
      // skip deleted instructions
      while (index != end && !getInstructionFromIndex(index))
        index += InstrSlots::NUM;
      if (index == end) break;

      MachineInstr *MI = getInstructionFromIndex(index);
      unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
      if (tii_->isMoveInstr(*MI, SrcReg, DstReg, SrcSubReg, DstSubReg))
        if (SrcReg == li.reg || DstReg == li.reg)
          continue;
      for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
        MachineOperand& mop = MI->getOperand(i);
        if (!mop.isReg())
          continue;
        unsigned PhysReg = mop.getReg();
        if (PhysReg == 0 || PhysReg == li.reg)
          continue;
        if (TargetRegisterInfo::isVirtualRegister(PhysReg)) {
          if (!vrm.hasPhys(PhysReg))
            continue;
          PhysReg = vrm.getPhys(PhysReg);
        }
        if (PhysReg && tri_->regsOverlap(PhysReg, reg))
          return true;
      }
    }
  }

  return false;
}

/// conflictsWithPhysRegRef - Similar to conflictsWithPhysRegRef except
/// it can check use as well.
bool LiveIntervals::conflictsWithPhysRegRef(LiveInterval &li,
                                            unsigned Reg, bool CheckUse,
                                  SmallPtrSet<MachineInstr*,32> &JoinedCopies) {
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    for (unsigned index = getBaseIndex(I->start),
           end = getBaseIndex(I->end-1) + InstrSlots::NUM; index != end;
         index += InstrSlots::NUM) {
      // Skip deleted instructions.
      MachineInstr *MI = 0;
      while (index != end) {
        MI = getInstructionFromIndex(index);
        if (MI)
          break;
        index += InstrSlots::NUM;
      }
      if (index == end) break;

      if (JoinedCopies.count(MI))
        continue;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand& MO = MI->getOperand(i);
        if (!MO.isReg())
          continue;
        if (MO.isUse() && !CheckUse)
          continue;
        unsigned PhysReg = MO.getReg();
        if (PhysReg == 0 || TargetRegisterInfo::isVirtualRegister(PhysReg))
          continue;
        if (tri_->isSubRegister(Reg, PhysReg))
          return true;
      }
    }
  }

  return false;
}


void LiveIntervals::printRegName(unsigned reg) const {
  if (TargetRegisterInfo::isPhysicalRegister(reg))
    cerr << tri_->getName(reg);
  else
    cerr << "%reg" << reg;
}

void LiveIntervals::handleVirtualRegisterDef(MachineBasicBlock *mbb,
                                             MachineBasicBlock::iterator mi,
                                             unsigned MIIdx, MachineOperand& MO,
                                             unsigned MOIdx,
                                             LiveInterval &interval) {
  DOUT << "\t\tregister: "; DEBUG(printRegName(interval.reg));
  LiveVariables::VarInfo& vi = lv_->getVarInfo(interval.reg);

  if (mi->getOpcode() == TargetInstrInfo::IMPLICIT_DEF) {
    DOUT << "is a implicit_def\n";
    return;
  }

  // Virtual registers may be defined multiple times (due to phi
  // elimination and 2-addr elimination).  Much of what we do only has to be
  // done once for the vreg.  We use an empty interval to detect the first
  // time we see a vreg.
  if (interval.empty()) {
    // Get the Idx of the defining instructions.
    unsigned defIndex = getDefIndex(MIIdx);
    // Earlyclobbers move back one.
    if (MO.isEarlyClobber())
      defIndex = getUseIndex(MIIdx);
    VNInfo *ValNo;
    MachineInstr *CopyMI = NULL;
    unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
    if (mi->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG ||
        mi->getOpcode() == TargetInstrInfo::INSERT_SUBREG ||
        tii_->isMoveInstr(*mi, SrcReg, DstReg, SrcSubReg, DstSubReg))
      CopyMI = mi;
    // Earlyclobbers move back one.
    ValNo = interval.getNextValue(defIndex, CopyMI, VNInfoAllocator);

    assert(ValNo->id == 0 && "First value in interval is not 0?");

    // Loop over all of the blocks that the vreg is defined in.  There are
    // two cases we have to handle here.  The most common case is a vreg
    // whose lifetime is contained within a basic block.  In this case there
    // will be a single kill, in MBB, which comes after the definition.
    if (vi.Kills.size() == 1 && vi.Kills[0]->getParent() == mbb) {
      // FIXME: what about dead vars?
      unsigned killIdx;
      if (vi.Kills[0] != mi)
        killIdx = getUseIndex(getInstructionIndex(vi.Kills[0]))+1;
      else
        killIdx = defIndex+1;

      // If the kill happens after the definition, we have an intra-block
      // live range.
      if (killIdx > defIndex) {
        assert(vi.AliveBlocks.none() &&
               "Shouldn't be alive across any blocks!");
        LiveRange LR(defIndex, killIdx, ValNo);
        interval.addRange(LR);
        DOUT << " +" << LR << "\n";
        interval.addKill(ValNo, killIdx);
        return;
      }
    }

    // The other case we handle is when a virtual register lives to the end
    // of the defining block, potentially live across some blocks, then is
    // live into some number of blocks, but gets killed.  Start by adding a
    // range that goes from this definition to the end of the defining block.
    LiveRange NewLR(defIndex, getMBBEndIdx(mbb)+1, ValNo);
    DOUT << " +" << NewLR;
    interval.addRange(NewLR);

    // Iterate over all of the blocks that the variable is completely
    // live in, adding [insrtIndex(begin), instrIndex(end)+4) to the
    // live interval.
    for (int i = vi.AliveBlocks.find_first(); i != -1;
         i = vi.AliveBlocks.find_next(i)) {
      LiveRange LR(getMBBStartIdx(i),
                   getMBBEndIdx(i)+1,  // MBB ends at -1.
                   ValNo);
      interval.addRange(LR);
      DOUT << " +" << LR;
    }

    // Finally, this virtual register is live from the start of any killing
    // block to the 'use' slot of the killing instruction.
    for (unsigned i = 0, e = vi.Kills.size(); i != e; ++i) {
      MachineInstr *Kill = vi.Kills[i];
      unsigned killIdx = getUseIndex(getInstructionIndex(Kill))+1;
      LiveRange LR(getMBBStartIdx(Kill->getParent()),
                   killIdx, ValNo);
      interval.addRange(LR);
      interval.addKill(ValNo, killIdx);
      DOUT << " +" << LR;
    }

  } else {
    // If this is the second time we see a virtual register definition, it
    // must be due to phi elimination or two addr elimination.  If this is
    // the result of two address elimination, then the vreg is one of the
    // def-and-use register operand.
    if (mi->isRegReDefinedByTwoAddr(MOIdx)) {
      // If this is a two-address definition, then we have already processed
      // the live range.  The only problem is that we didn't realize there
      // are actually two values in the live interval.  Because of this we
      // need to take the LiveRegion that defines this register and split it
      // into two values.
      assert(interval.containsOneValue());
      unsigned DefIndex = getDefIndex(interval.getValNumInfo(0)->def);
      unsigned RedefIndex = getDefIndex(MIIdx);
      // It cannot be an early clobber MO.
      assert(!MO.isEarlyClobber() && "Unexpected early clobber!");

      const LiveRange *OldLR = interval.getLiveRangeContaining(RedefIndex-1);
      VNInfo *OldValNo = OldLR->valno;

      // Delete the initial value, which should be short and continuous,
      // because the 2-addr copy must be in the same MBB as the redef.
      interval.removeRange(DefIndex, RedefIndex);

      // Two-address vregs should always only be redefined once.  This means
      // that at this point, there should be exactly one value number in it.
      assert(interval.containsOneValue() && "Unexpected 2-addr liveint!");

      // The new value number (#1) is defined by the instruction we claimed
      // defined value #0.
      VNInfo *ValNo = interval.getNextValue(OldValNo->def, OldValNo->copy,
                                            VNInfoAllocator);
      
      // Value#0 is now defined by the 2-addr instruction.
      OldValNo->def  = RedefIndex;
      OldValNo->copy = 0;
      
      // Add the new live interval which replaces the range for the input copy.
      LiveRange LR(DefIndex, RedefIndex, ValNo);
      DOUT << " replace range with " << LR;
      interval.addRange(LR);
      interval.addKill(ValNo, RedefIndex);

      // If this redefinition is dead, we need to add a dummy unit live
      // range covering the def slot.
      if (MO.isDead())
        interval.addRange(LiveRange(RedefIndex, RedefIndex+1, OldValNo));

      DOUT << " RESULT: ";
      interval.print(DOUT, tri_);

    } else {
      // Otherwise, this must be because of phi elimination.  If this is the
      // first redefinition of the vreg that we have seen, go back and change
      // the live range in the PHI block to be a different value number.
      if (interval.containsOneValue()) {
        assert(vi.Kills.size() == 1 &&
               "PHI elimination vreg should have one kill, the PHI itself!");

        // Remove the old range that we now know has an incorrect number.
        VNInfo *VNI = interval.getValNumInfo(0);
        MachineInstr *Killer = vi.Kills[0];
        unsigned Start = getMBBStartIdx(Killer->getParent());
        unsigned End = getUseIndex(getInstructionIndex(Killer))+1;
        DOUT << " Removing [" << Start << "," << End << "] from: ";
        interval.print(DOUT, tri_); DOUT << "\n";
        interval.removeRange(Start, End);
        VNI->hasPHIKill = true;
        DOUT << " RESULT: "; interval.print(DOUT, tri_);

        // Replace the interval with one of a NEW value number.  Note that this
        // value number isn't actually defined by an instruction, weird huh? :)
        LiveRange LR(Start, End, interval.getNextValue(~0, 0, VNInfoAllocator));
        DOUT << " replace range with " << LR;
        interval.addRange(LR);
        interval.addKill(LR.valno, End);
        DOUT << " RESULT: "; interval.print(DOUT, tri_);
      }

      // In the case of PHI elimination, each variable definition is only
      // live until the end of the block.  We've already taken care of the
      // rest of the live range.
      unsigned defIndex = getDefIndex(MIIdx);
      // It cannot be an early clobber MO.
      assert(!MO.isEarlyClobber() && "Unexpected early clobber!");
      
      VNInfo *ValNo;
      MachineInstr *CopyMI = NULL;
      unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
      if (mi->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG ||
          mi->getOpcode() == TargetInstrInfo::INSERT_SUBREG ||
          tii_->isMoveInstr(*mi, SrcReg, DstReg, SrcSubReg, DstSubReg))
        CopyMI = mi;
      ValNo = interval.getNextValue(defIndex, CopyMI, VNInfoAllocator);
      
      unsigned killIndex = getMBBEndIdx(mbb) + 1;
      LiveRange LR(defIndex, killIndex, ValNo);
      interval.addRange(LR);
      interval.addKill(ValNo, killIndex);
      ValNo->hasPHIKill = true;
      DOUT << " +" << LR;
    }
  }

  DOUT << '\n';
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock *MBB,
                                              MachineBasicBlock::iterator mi,
                                              unsigned MIIdx,
                                              MachineOperand& MO,
                                              LiveInterval &interval,
                                              MachineInstr *CopyMI) {
  // A physical register cannot be live across basic block, so its
  // lifetime must end somewhere in its defining basic block.
  DOUT << "\t\tregister: "; DEBUG(printRegName(interval.reg));

  unsigned baseIndex = MIIdx;
  unsigned start = getDefIndex(baseIndex);
  // Earlyclobbers move back one.
  if (MO.isEarlyClobber())
    start = getUseIndex(MIIdx);
  unsigned end = start;

  // If it is not used after definition, it is considered dead at
  // the instruction defining it. Hence its interval is:
  // [defSlot(def), defSlot(def)+1)
  if (MO.isDead()) {
    DOUT << " dead";
    end = start + 1;
    goto exit;
  }

  // If it is not dead on definition, it must be killed by a
  // subsequent instruction. Hence its interval is:
  // [defSlot(def), useSlot(kill)+1)
  baseIndex += InstrSlots::NUM;
  while (++mi != MBB->end()) {
    while (baseIndex / InstrSlots::NUM < i2miMap_.size() &&
           getInstructionFromIndex(baseIndex) == 0)
      baseIndex += InstrSlots::NUM;
    if (mi->killsRegister(interval.reg, tri_)) {
      DOUT << " killed";
      end = getUseIndex(baseIndex) + 1;
      goto exit;
    } else if (mi->modifiesRegister(interval.reg, tri_)) {
      // Another instruction redefines the register before it is ever read.
      // Then the register is essentially dead at the instruction that defines
      // it. Hence its interval is:
      // [defSlot(def), defSlot(def)+1)
      DOUT << " dead";
      end = start + 1;
      goto exit;
    }
    
    baseIndex += InstrSlots::NUM;
  }
  
  // The only case we should have a dead physreg here without a killing or
  // instruction where we know it's dead is if it is live-in to the function
  // and never used.
  assert(!CopyMI && "physreg was not killed in defining block!");
  end = start + 1;

exit:
  assert(start < end && "did not find end of interval?");

  // Already exists? Extend old live interval.
  LiveInterval::iterator OldLR = interval.FindLiveRangeContaining(start);
  bool Extend = OldLR != interval.end();
  VNInfo *ValNo = Extend
    ? OldLR->valno : interval.getNextValue(start, CopyMI, VNInfoAllocator);
  if (MO.isEarlyClobber() && Extend)
    ValNo->redefByEC = true;
  LiveRange LR(start, end, ValNo);
  interval.addRange(LR);
  interval.addKill(LR.valno, end);
  DOUT << " +" << LR << '\n';
}

void LiveIntervals::handleRegisterDef(MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator MI,
                                      unsigned MIIdx,
                                      MachineOperand& MO,
                                      unsigned MOIdx) {
  if (TargetRegisterInfo::isVirtualRegister(MO.getReg()))
    handleVirtualRegisterDef(MBB, MI, MIIdx, MO, MOIdx,
                             getOrCreateInterval(MO.getReg()));
  else if (allocatableRegs_[MO.getReg()]) {
    MachineInstr *CopyMI = NULL;
    unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
    if (MI->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG ||
        MI->getOpcode() == TargetInstrInfo::INSERT_SUBREG ||
        tii_->isMoveInstr(*MI, SrcReg, DstReg, SrcSubReg, DstSubReg))
      CopyMI = MI;
    handlePhysicalRegisterDef(MBB, MI, MIIdx, MO, 
                              getOrCreateInterval(MO.getReg()), CopyMI);
    // Def of a register also defines its sub-registers.
    for (const unsigned* AS = tri_->getSubRegisters(MO.getReg()); *AS; ++AS)
      // If MI also modifies the sub-register explicitly, avoid processing it
      // more than once. Do not pass in TRI here so it checks for exact match.
      if (!MI->modifiesRegister(*AS))
        handlePhysicalRegisterDef(MBB, MI, MIIdx, MO, 
                                  getOrCreateInterval(*AS), 0);
  }
}

void LiveIntervals::handleLiveInRegister(MachineBasicBlock *MBB,
                                         unsigned MIIdx,
                                         LiveInterval &interval, bool isAlias) {
  DOUT << "\t\tlivein register: "; DEBUG(printRegName(interval.reg));

  // Look for kills, if it reaches a def before it's killed, then it shouldn't
  // be considered a livein.
  MachineBasicBlock::iterator mi = MBB->begin();
  unsigned baseIndex = MIIdx;
  unsigned start = baseIndex;
  while (baseIndex / InstrSlots::NUM < i2miMap_.size() && 
         getInstructionFromIndex(baseIndex) == 0)
    baseIndex += InstrSlots::NUM;
  unsigned end = baseIndex;
  
  while (mi != MBB->end()) {
    if (mi->killsRegister(interval.reg, tri_)) {
      DOUT << " killed";
      end = getUseIndex(baseIndex) + 1;
      goto exit;
    } else if (mi->modifiesRegister(interval.reg, tri_)) {
      // Another instruction redefines the register before it is ever read.
      // Then the register is essentially dead at the instruction that defines
      // it. Hence its interval is:
      // [defSlot(def), defSlot(def)+1)
      DOUT << " dead";
      end = getDefIndex(start) + 1;
      goto exit;
    }

    baseIndex += InstrSlots::NUM;
    while (baseIndex / InstrSlots::NUM < i2miMap_.size() && 
           getInstructionFromIndex(baseIndex) == 0)
      baseIndex += InstrSlots::NUM;
    ++mi;
  }

exit:
  // Live-in register might not be used at all.
  if (end == MIIdx) {
    if (isAlias) {
      DOUT << " dead";
      end = getDefIndex(MIIdx) + 1;
    } else {
      DOUT << " live through";
      end = baseIndex;
    }
  }

  LiveRange LR(start, end, interval.getNextValue(~0U, 0, VNInfoAllocator));
  interval.addRange(LR);
  interval.addKill(LR.valno, end);
  DOUT << " +" << LR << '\n';
}

/// computeIntervals - computes the live intervals for virtual
/// registers. for some ordering of the machine instructions [1,N] a
/// live interval is an interval [i, j) where 1 <= i <= j < N for
/// which a variable is live
void LiveIntervals::computeIntervals() { 

  DOUT << "********** COMPUTING LIVE INTERVALS **********\n"
       << "********** Function: "
       << ((Value*)mf_->getFunction())->getName() << '\n';
  
  for (MachineFunction::iterator MBBI = mf_->begin(), E = mf_->end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    // Track the index of the current machine instr.
    unsigned MIIndex = getMBBStartIdx(MBB);
    DOUT << ((Value*)MBB->getBasicBlock())->getName() << ":\n";

    MachineBasicBlock::iterator MI = MBB->begin(), miEnd = MBB->end();

    // Create intervals for live-ins to this BB first.
    for (MachineBasicBlock::const_livein_iterator LI = MBB->livein_begin(),
           LE = MBB->livein_end(); LI != LE; ++LI) {
      handleLiveInRegister(MBB, MIIndex, getOrCreateInterval(*LI));
      // Multiple live-ins can alias the same register.
      for (const unsigned* AS = tri_->getSubRegisters(*LI); *AS; ++AS)
        if (!hasInterval(*AS))
          handleLiveInRegister(MBB, MIIndex, getOrCreateInterval(*AS),
                               true);
    }
    
    // Skip over empty initial indices.
    while (MIIndex / InstrSlots::NUM < i2miMap_.size() &&
           getInstructionFromIndex(MIIndex) == 0)
      MIIndex += InstrSlots::NUM;
    
    for (; MI != miEnd; ++MI) {
      DOUT << MIIndex << "\t" << *MI;

      // Handle defs.
      for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
        MachineOperand &MO = MI->getOperand(i);
        // handle register defs - build intervals
        if (MO.isReg() && MO.getReg() && MO.isDef()) {
          handleRegisterDef(MBB, MI, MIIndex, MO, i);
        }
      }

      // Skip over the empty slots after each instruction.
      unsigned Slots = MI->getDesc().getNumDefs();
      if (Slots == 0)
        Slots = 1;
      MIIndex += InstrSlots::NUM * Slots;
      
      // Skip over empty indices.
      while (MIIndex / InstrSlots::NUM < i2miMap_.size() &&
             getInstructionFromIndex(MIIndex) == 0)
        MIIndex += InstrSlots::NUM;
    }
  }
}

bool LiveIntervals::findLiveInMBBs(unsigned Start, unsigned End,
                              SmallVectorImpl<MachineBasicBlock*> &MBBs) const {
  std::vector<IdxMBBPair>::const_iterator I =
    std::lower_bound(Idx2MBBMap.begin(), Idx2MBBMap.end(), Start);

  bool ResVal = false;
  while (I != Idx2MBBMap.end()) {
    if (I->first >= End)
      break;
    MBBs.push_back(I->second);
    ResVal = true;
    ++I;
  }
  return ResVal;
}

bool LiveIntervals::findReachableMBBs(unsigned Start, unsigned End,
                              SmallVectorImpl<MachineBasicBlock*> &MBBs) const {
  std::vector<IdxMBBPair>::const_iterator I =
    std::lower_bound(Idx2MBBMap.begin(), Idx2MBBMap.end(), Start);

  bool ResVal = false;
  while (I != Idx2MBBMap.end()) {
    if (I->first > End)
      break;
    MachineBasicBlock *MBB = I->second;
    if (getMBBEndIdx(MBB) > End)
      break;
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
           SE = MBB->succ_end(); SI != SE; ++SI)
      MBBs.push_back(*SI);
    ResVal = true;
    ++I;
  }
  return ResVal;
}

LiveInterval* LiveIntervals::createInterval(unsigned reg) {
  float Weight = TargetRegisterInfo::isPhysicalRegister(reg) ?
                       HUGE_VALF : 0.0F;
  return new LiveInterval(reg, Weight);
}

/// getVNInfoSourceReg - Helper function that parses the specified VNInfo
/// copy field and returns the source register that defines it.
unsigned LiveIntervals::getVNInfoSourceReg(const VNInfo *VNI) const {
  if (!VNI->copy)
    return 0;

  if (VNI->copy->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG) {
    // If it's extracting out of a physical register, return the sub-register.
    unsigned Reg = VNI->copy->getOperand(1).getReg();
    if (TargetRegisterInfo::isPhysicalRegister(Reg))
      Reg = tri_->getSubReg(Reg, VNI->copy->getOperand(2).getImm());
    return Reg;
  } else if (VNI->copy->getOpcode() == TargetInstrInfo::INSERT_SUBREG)
    return VNI->copy->getOperand(2).getReg();

  unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
  if (tii_->isMoveInstr(*VNI->copy, SrcReg, DstReg, SrcSubReg, DstSubReg))
    return SrcReg;
  assert(0 && "Unrecognized copy instruction!");
  return 0;
}

//===----------------------------------------------------------------------===//
// Register allocator hooks.
//

/// getReMatImplicitUse - If the remat definition MI has one (for now, we only
/// allow one) virtual register operand, then its uses are implicitly using
/// the register. Returns the virtual register.
unsigned LiveIntervals::getReMatImplicitUse(const LiveInterval &li,
                                            MachineInstr *MI) const {
  unsigned RegOp = 0;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0 || Reg == li.reg)
      continue;
    // FIXME: For now, only remat MI with at most one register operand.
    assert(!RegOp &&
           "Can't rematerialize instruction with multiple register operand!");
    RegOp = MO.getReg();
#ifndef NDEBUG
    break;
#endif
  }
  return RegOp;
}

/// isValNoAvailableAt - Return true if the val# of the specified interval
/// which reaches the given instruction also reaches the specified use index.
bool LiveIntervals::isValNoAvailableAt(const LiveInterval &li, MachineInstr *MI,
                                       unsigned UseIdx) const {
  unsigned Index = getInstructionIndex(MI);  
  VNInfo *ValNo = li.FindLiveRangeContaining(Index)->valno;
  LiveInterval::const_iterator UI = li.FindLiveRangeContaining(UseIdx);
  return UI != li.end() && UI->valno == ValNo;
}

/// isReMaterializable - Returns true if the definition MI of the specified
/// val# of the specified interval is re-materializable.
bool LiveIntervals::isReMaterializable(const LiveInterval &li,
                                       const VNInfo *ValNo, MachineInstr *MI,
                                       SmallVectorImpl<LiveInterval*> &SpillIs,
                                       bool &isLoad) {
  if (DisableReMat)
    return false;

  if (MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF)
    return true;

  int FrameIdx = 0;
  if (tii_->isLoadFromStackSlot(MI, FrameIdx) &&
      mf_->getFrameInfo()->isImmutableObjectIndex(FrameIdx))
    // FIXME: Let target specific isReallyTriviallyReMaterializable determines
    // this but remember this is not safe to fold into a two-address
    // instruction.
    // This is a load from fixed stack slot. It can be rematerialized.
    return true;

  // If the target-specific rules don't identify an instruction as
  // being trivially rematerializable, use some target-independent
  // rules.
  if (!MI->getDesc().isRematerializable() ||
      !tii_->isTriviallyReMaterializable(MI)) {
    if (!EnableAggressiveRemat)
      return false;

    // If the instruction accesses memory but the memoperands have been lost,
    // we can't analyze it.
    const TargetInstrDesc &TID = MI->getDesc();
    if ((TID.mayLoad() || TID.mayStore()) && MI->memoperands_empty())
      return false;

    // Avoid instructions obviously unsafe for remat.
    if (TID.hasUnmodeledSideEffects() || TID.isNotDuplicable())
      return false;

    // If the instruction accesses memory and the memory could be non-constant,
    // assume the instruction is not rematerializable.
    for (std::list<MachineMemOperand>::const_iterator
           I = MI->memoperands_begin(), E = MI->memoperands_end(); I != E; ++I){
      const MachineMemOperand &MMO = *I;
      if (MMO.isVolatile() || MMO.isStore())
        return false;
      const Value *V = MMO.getValue();
      if (!V)
        return false;
      if (const PseudoSourceValue *PSV = dyn_cast<PseudoSourceValue>(V)) {
        if (!PSV->isConstant(mf_->getFrameInfo()))
          return false;
      } else if (!aa_->pointsToConstantMemory(V))
        return false;
    }

    // If any of the registers accessed are non-constant, conservatively assume
    // the instruction is not rematerializable.
    unsigned ImpUse = 0;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg()) {
        unsigned Reg = MO.getReg();
        if (Reg == 0)
          continue;
        if (TargetRegisterInfo::isPhysicalRegister(Reg))
          return false;

        // Only allow one def, and that in the first operand.
        if (MO.isDef() != (i == 0))
          return false;

        // Only allow constant-valued registers.
        bool IsLiveIn = mri_->isLiveIn(Reg);
        MachineRegisterInfo::def_iterator I = mri_->def_begin(Reg),
                                          E = mri_->def_end();

        // For the def, it should be the only def of that register.
        if (MO.isDef() && (next(I) != E || IsLiveIn))
          return false;

        if (MO.isUse()) {
          // Only allow one use other register use, as that's all the
          // remat mechanisms support currently.
          if (Reg != li.reg) {
            if (ImpUse == 0)
              ImpUse = Reg;
            else if (Reg != ImpUse)
              return false;
          }
          // For the use, there should be only one associated def.
          if (I != E && (next(I) != E || IsLiveIn))
            return false;
        }
      }
    }
  }

  unsigned ImpUse = getReMatImplicitUse(li, MI);
  if (ImpUse) {
    const LiveInterval &ImpLi = getInterval(ImpUse);
    for (MachineRegisterInfo::use_iterator ri = mri_->use_begin(li.reg),
           re = mri_->use_end(); ri != re; ++ri) {
      MachineInstr *UseMI = &*ri;
      unsigned UseIdx = getInstructionIndex(UseMI);
      if (li.FindLiveRangeContaining(UseIdx)->valno != ValNo)
        continue;
      if (!isValNoAvailableAt(ImpLi, MI, UseIdx))
        return false;
    }

    // If a register operand of the re-materialized instruction is going to
    // be spilled next, then it's not legal to re-materialize this instruction.
    for (unsigned i = 0, e = SpillIs.size(); i != e; ++i)
      if (ImpUse == SpillIs[i]->reg)
        return false;
  }
  return true;
}

/// isReMaterializable - Returns true if the definition MI of the specified
/// val# of the specified interval is re-materializable.
bool LiveIntervals::isReMaterializable(const LiveInterval &li,
                                       const VNInfo *ValNo, MachineInstr *MI) {
  SmallVector<LiveInterval*, 4> Dummy1;
  bool Dummy2;
  return isReMaterializable(li, ValNo, MI, Dummy1, Dummy2);
}

/// isReMaterializable - Returns true if every definition of MI of every
/// val# of the specified interval is re-materializable.
bool LiveIntervals::isReMaterializable(const LiveInterval &li,
                                       SmallVectorImpl<LiveInterval*> &SpillIs,
                                       bool &isLoad) {
  isLoad = false;
  for (LiveInterval::const_vni_iterator i = li.vni_begin(), e = li.vni_end();
       i != e; ++i) {
    const VNInfo *VNI = *i;
    unsigned DefIdx = VNI->def;
    if (DefIdx == ~1U)
      continue; // Dead val#.
    // Is the def for the val# rematerializable?
    if (DefIdx == ~0u)
      return false;
    MachineInstr *ReMatDefMI = getInstructionFromIndex(DefIdx);
    bool DefIsLoad = false;
    if (!ReMatDefMI ||
        !isReMaterializable(li, VNI, ReMatDefMI, SpillIs, DefIsLoad))
      return false;
    isLoad |= DefIsLoad;
  }
  return true;
}

/// FilterFoldedOps - Filter out two-address use operands. Return
/// true if it finds any issue with the operands that ought to prevent
/// folding.
static bool FilterFoldedOps(MachineInstr *MI,
                            SmallVector<unsigned, 2> &Ops,
                            unsigned &MRInfo,
                            SmallVector<unsigned, 2> &FoldOps) {
  const TargetInstrDesc &TID = MI->getDesc();

  MRInfo = 0;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    unsigned OpIdx = Ops[i];
    MachineOperand &MO = MI->getOperand(OpIdx);
    // FIXME: fold subreg use.
    if (MO.getSubReg())
      return true;
    if (MO.isDef())
      MRInfo |= (unsigned)VirtRegMap::isMod;
    else {
      // Filter out two-address use operand(s).
      if (!MO.isImplicit() &&
          TID.getOperandConstraint(OpIdx, TOI::TIED_TO) != -1) {
        MRInfo = VirtRegMap::isModRef;
        continue;
      }
      MRInfo |= (unsigned)VirtRegMap::isRef;
    }
    FoldOps.push_back(OpIdx);
  }
  return false;
}
                           

/// tryFoldMemoryOperand - Attempts to fold either a spill / restore from
/// slot / to reg or any rematerialized load into ith operand of specified
/// MI. If it is successul, MI is updated with the newly created MI and
/// returns true.
bool LiveIntervals::tryFoldMemoryOperand(MachineInstr* &MI,
                                         VirtRegMap &vrm, MachineInstr *DefMI,
                                         unsigned InstrIdx,
                                         SmallVector<unsigned, 2> &Ops,
                                         bool isSS, int Slot, unsigned Reg) {
  // If it is an implicit def instruction, just delete it.
  if (MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF) {
    RemoveMachineInstrFromMaps(MI);
    vrm.RemoveMachineInstrFromMaps(MI);
    MI->eraseFromParent();
    ++numFolds;
    return true;
  }

  // Filter the list of operand indexes that are to be folded. Abort if
  // any operand will prevent folding.
  unsigned MRInfo = 0;
  SmallVector<unsigned, 2> FoldOps;
  if (FilterFoldedOps(MI, Ops, MRInfo, FoldOps))
    return false;

  // The only time it's safe to fold into a two address instruction is when
  // it's folding reload and spill from / into a spill stack slot.
  if (DefMI && (MRInfo & VirtRegMap::isMod))
    return false;

  MachineInstr *fmi = isSS ? tii_->foldMemoryOperand(*mf_, MI, FoldOps, Slot)
                           : tii_->foldMemoryOperand(*mf_, MI, FoldOps, DefMI);
  if (fmi) {
    // Remember this instruction uses the spill slot.
    if (isSS) vrm.addSpillSlotUse(Slot, fmi);

    // Attempt to fold the memory reference into the instruction. If
    // we can do this, we don't need to insert spill code.
    MachineBasicBlock &MBB = *MI->getParent();
    if (isSS && !mf_->getFrameInfo()->isImmutableObjectIndex(Slot))
      vrm.virtFolded(Reg, MI, fmi, (VirtRegMap::ModRef)MRInfo);
    vrm.transferSpillPts(MI, fmi);
    vrm.transferRestorePts(MI, fmi);
    vrm.transferEmergencySpills(MI, fmi);
    mi2iMap_.erase(MI);
    i2miMap_[InstrIdx /InstrSlots::NUM] = fmi;
    mi2iMap_[fmi] = InstrIdx;
    MI = MBB.insert(MBB.erase(MI), fmi);
    ++numFolds;
    return true;
  }
  return false;
}

/// canFoldMemoryOperand - Returns true if the specified load / store
/// folding is possible.
bool LiveIntervals::canFoldMemoryOperand(MachineInstr *MI,
                                         SmallVector<unsigned, 2> &Ops,
                                         bool ReMat) const {
  // Filter the list of operand indexes that are to be folded. Abort if
  // any operand will prevent folding.
  unsigned MRInfo = 0;
  SmallVector<unsigned, 2> FoldOps;
  if (FilterFoldedOps(MI, Ops, MRInfo, FoldOps))
    return false;

  // It's only legal to remat for a use, not a def.
  if (ReMat && (MRInfo & VirtRegMap::isMod))
    return false;

  return tii_->canFoldMemoryOperand(MI, FoldOps);
}

bool LiveIntervals::intervalIsInOneMBB(const LiveInterval &li) const {
  SmallPtrSet<MachineBasicBlock*, 4> MBBs;
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    std::vector<IdxMBBPair>::const_iterator II =
      std::lower_bound(Idx2MBBMap.begin(), Idx2MBBMap.end(), I->start);
    if (II == Idx2MBBMap.end())
      continue;
    if (I->end > II->first)  // crossing a MBB.
      return false;
    MBBs.insert(II->second);
    if (MBBs.size() > 1)
      return false;
  }
  return true;
}

/// rewriteImplicitOps - Rewrite implicit use operands of MI (i.e. uses of
/// interval on to-be re-materialized operands of MI) with new register.
void LiveIntervals::rewriteImplicitOps(const LiveInterval &li,
                                       MachineInstr *MI, unsigned NewVReg,
                                       VirtRegMap &vrm) {
  // There is an implicit use. That means one of the other operand is
  // being remat'ed and the remat'ed instruction has li.reg as an
  // use operand. Make sure we rewrite that as well.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0 || TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    if (!vrm.isReMaterialized(Reg))
      continue;
    MachineInstr *ReMatMI = vrm.getReMaterializedMI(Reg);
    MachineOperand *UseMO = ReMatMI->findRegisterUseOperand(li.reg);
    if (UseMO)
      UseMO->setReg(NewVReg);
  }
}

/// rewriteInstructionForSpills, rewriteInstructionsForSpills - Helper functions
/// for addIntervalsForSpills to rewrite uses / defs for the given live range.
bool LiveIntervals::
rewriteInstructionForSpills(const LiveInterval &li, const VNInfo *VNI,
                 bool TrySplit, unsigned index, unsigned end,  MachineInstr *MI,
                 MachineInstr *ReMatOrigDefMI, MachineInstr *ReMatDefMI,
                 unsigned Slot, int LdSlot,
                 bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
                 VirtRegMap &vrm,
                 const TargetRegisterClass* rc,
                 SmallVector<int, 4> &ReMatIds,
                 const MachineLoopInfo *loopInfo,
                 unsigned &NewVReg, unsigned ImpUse, bool &HasDef, bool &HasUse,
                 DenseMap<unsigned,unsigned> &MBBVRegsMap,
                 std::vector<LiveInterval*> &NewLIs, float &SSWeight) {
  MachineBasicBlock *MBB = MI->getParent();
  unsigned loopDepth = loopInfo->getLoopDepth(MBB);
  bool CanFold = false;
 RestartInstruction:
  for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
    MachineOperand& mop = MI->getOperand(i);
    if (!mop.isReg())
      continue;
    unsigned Reg = mop.getReg();
    unsigned RegI = Reg;
    if (Reg == 0 || TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    if (Reg != li.reg)
      continue;

    bool TryFold = !DefIsReMat;
    bool FoldSS = true; // Default behavior unless it's a remat.
    int FoldSlot = Slot;
    if (DefIsReMat) {
      // If this is the rematerializable definition MI itself and
      // all of its uses are rematerialized, simply delete it.
      if (MI == ReMatOrigDefMI && CanDelete) {
        DOUT << "\t\t\t\tErasing re-materlizable def: ";
        DOUT << MI << '\n';
        RemoveMachineInstrFromMaps(MI);
        vrm.RemoveMachineInstrFromMaps(MI);
        MI->eraseFromParent();
        break;
      }

      // If def for this use can't be rematerialized, then try folding.
      // If def is rematerializable and it's a load, also try folding.
      TryFold = !ReMatDefMI || (ReMatDefMI && (MI == ReMatOrigDefMI || isLoad));
      if (isLoad) {
        // Try fold loads (from stack slot, constant pool, etc.) into uses.
        FoldSS = isLoadSS;
        FoldSlot = LdSlot;
      }
    }

    // Scan all of the operands of this instruction rewriting operands
    // to use NewVReg instead of li.reg as appropriate.  We do this for
    // two reasons:
    //
    //   1. If the instr reads the same spilled vreg multiple times, we
    //      want to reuse the NewVReg.
    //   2. If the instr is a two-addr instruction, we are required to
    //      keep the src/dst regs pinned.
    //
    // Keep track of whether we replace a use and/or def so that we can
    // create the spill interval with the appropriate range. 

    HasUse = mop.isUse();
    HasDef = mop.isDef();
    SmallVector<unsigned, 2> Ops;
    Ops.push_back(i);
    for (unsigned j = i+1, e = MI->getNumOperands(); j != e; ++j) {
      const MachineOperand &MOj = MI->getOperand(j);
      if (!MOj.isReg())
        continue;
      unsigned RegJ = MOj.getReg();
      if (RegJ == 0 || TargetRegisterInfo::isPhysicalRegister(RegJ))
        continue;
      if (RegJ == RegI) {
        Ops.push_back(j);
        HasUse |= MOj.isUse();
        HasDef |= MOj.isDef();
      }
    }

    if (HasUse && !li.liveAt(getUseIndex(index)))
      // Must be defined by an implicit def. It should not be spilled. Note,
      // this is for correctness reason. e.g.
      // 8   %reg1024<def> = IMPLICIT_DEF
      // 12  %reg1024<def> = INSERT_SUBREG %reg1024<kill>, %reg1025, 2
      // The live range [12, 14) are not part of the r1024 live interval since
      // it's defined by an implicit def. It will not conflicts with live
      // interval of r1025. Now suppose both registers are spilled, you can
      // easily see a situation where both registers are reloaded before
      // the INSERT_SUBREG and both target registers that would overlap.
      HasUse = false;

    // Update stack slot spill weight if we are splitting.
    float Weight = getSpillWeight(HasDef, HasUse, loopDepth);
      if (!TrySplit)
      SSWeight += Weight;

    // Create a new virtual register for the spill interval.
    // Create the new register now so we can map the fold instruction
    // to the new register so when it is unfolded we get the correct
    // answer.
    bool CreatedNewVReg = false;
    if (NewVReg == 0) {
      NewVReg = mri_->createVirtualRegister(rc);
      vrm.grow();
      CreatedNewVReg = true;
    }

    if (!TryFold)
      CanFold = false;
    else {
      // Do not fold load / store here if we are splitting. We'll find an
      // optimal point to insert a load / store later.
      if (!TrySplit) {
        if (tryFoldMemoryOperand(MI, vrm, ReMatDefMI, index,
                                 Ops, FoldSS, FoldSlot, NewVReg)) {
          // Folding the load/store can completely change the instruction in
          // unpredictable ways, rescan it from the beginning.

          if (FoldSS) {
            // We need to give the new vreg the same stack slot as the
            // spilled interval.
            vrm.assignVirt2StackSlot(NewVReg, FoldSlot);
          }

          HasUse = false;
          HasDef = false;
          CanFold = false;
          if (isRemoved(MI)) {
            SSWeight -= Weight;
            break;
          }
          goto RestartInstruction;
        }
      } else {
        // We'll try to fold it later if it's profitable.
        CanFold = canFoldMemoryOperand(MI, Ops, DefIsReMat);
      }
    }

    mop.setReg(NewVReg);
    if (mop.isImplicit())
      rewriteImplicitOps(li, MI, NewVReg, vrm);

    // Reuse NewVReg for other reads.
    for (unsigned j = 0, e = Ops.size(); j != e; ++j) {
      MachineOperand &mopj = MI->getOperand(Ops[j]);
      mopj.setReg(NewVReg);
      if (mopj.isImplicit())
        rewriteImplicitOps(li, MI, NewVReg, vrm);
    }
            
    if (CreatedNewVReg) {
      if (DefIsReMat) {
        vrm.setVirtIsReMaterialized(NewVReg, ReMatDefMI/*, CanDelete*/);
        if (ReMatIds[VNI->id] == VirtRegMap::MAX_STACK_SLOT) {
          // Each valnum may have its own remat id.
          ReMatIds[VNI->id] = vrm.assignVirtReMatId(NewVReg);
        } else {
          vrm.assignVirtReMatId(NewVReg, ReMatIds[VNI->id]);
        }
        if (!CanDelete || (HasUse && HasDef)) {
          // If this is a two-addr instruction then its use operands are
          // rematerializable but its def is not. It should be assigned a
          // stack slot.
          vrm.assignVirt2StackSlot(NewVReg, Slot);
        }
      } else {
        vrm.assignVirt2StackSlot(NewVReg, Slot);
      }
    } else if (HasUse && HasDef &&
               vrm.getStackSlot(NewVReg) == VirtRegMap::NO_STACK_SLOT) {
      // If this interval hasn't been assigned a stack slot (because earlier
      // def is a deleted remat def), do it now.
      assert(Slot != VirtRegMap::NO_STACK_SLOT);
      vrm.assignVirt2StackSlot(NewVReg, Slot);
    }

    // Re-matting an instruction with virtual register use. Add the
    // register as an implicit use on the use MI.
    if (DefIsReMat && ImpUse)
      MI->addOperand(MachineOperand::CreateReg(ImpUse, false, true));

    // create a new register interval for this spill / remat.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    if (CreatedNewVReg) {
      NewLIs.push_back(&nI);
      MBBVRegsMap.insert(std::make_pair(MI->getParent()->getNumber(), NewVReg));
      if (TrySplit)
        vrm.setIsSplitFromReg(NewVReg, li.reg);
    }

    if (HasUse) {
      if (CreatedNewVReg) {
        LiveRange LR(getLoadIndex(index), getUseIndex(index)+1,
                     nI.getNextValue(~0U, 0, VNInfoAllocator));
        DOUT << " +" << LR;
        nI.addRange(LR);
      } else {
        // Extend the split live interval to this def / use.
        unsigned End = getUseIndex(index)+1;
        LiveRange LR(nI.ranges[nI.ranges.size()-1].end, End,
                     nI.getValNumInfo(nI.getNumValNums()-1));
        DOUT << " +" << LR;
        nI.addRange(LR);
      }
    }
    if (HasDef) {
      LiveRange LR(getDefIndex(index), getStoreIndex(index),
                   nI.getNextValue(~0U, 0, VNInfoAllocator));
      DOUT << " +" << LR;
      nI.addRange(LR);
    }

    DOUT << "\t\t\t\tAdded new interval: ";
    nI.print(DOUT, tri_);
    DOUT << '\n';
  }
  return CanFold;
}
bool LiveIntervals::anyKillInMBBAfterIdx(const LiveInterval &li,
                                   const VNInfo *VNI,
                                   MachineBasicBlock *MBB, unsigned Idx) const {
  unsigned End = getMBBEndIdx(MBB);
  for (unsigned j = 0, ee = VNI->kills.size(); j != ee; ++j) {
    unsigned KillIdx = VNI->kills[j];
    if (KillIdx > Idx && KillIdx < End)
      return true;
  }
  return false;
}

/// RewriteInfo - Keep track of machine instrs that will be rewritten
/// during spilling.
namespace {
  struct RewriteInfo {
    unsigned Index;
    MachineInstr *MI;
    bool HasUse;
    bool HasDef;
    RewriteInfo(unsigned i, MachineInstr *mi, bool u, bool d)
      : Index(i), MI(mi), HasUse(u), HasDef(d) {}
  };

  struct RewriteInfoCompare {
    bool operator()(const RewriteInfo &LHS, const RewriteInfo &RHS) const {
      return LHS.Index < RHS.Index;
    }
  };
}

void LiveIntervals::
rewriteInstructionsForSpills(const LiveInterval &li, bool TrySplit,
                    LiveInterval::Ranges::const_iterator &I,
                    MachineInstr *ReMatOrigDefMI, MachineInstr *ReMatDefMI,
                    unsigned Slot, int LdSlot,
                    bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
                    VirtRegMap &vrm,
                    const TargetRegisterClass* rc,
                    SmallVector<int, 4> &ReMatIds,
                    const MachineLoopInfo *loopInfo,
                    BitVector &SpillMBBs,
                    DenseMap<unsigned, std::vector<SRInfo> > &SpillIdxes,
                    BitVector &RestoreMBBs,
                    DenseMap<unsigned, std::vector<SRInfo> > &RestoreIdxes,
                    DenseMap<unsigned,unsigned> &MBBVRegsMap,
                    std::vector<LiveInterval*> &NewLIs, float &SSWeight) {
  bool AllCanFold = true;
  unsigned NewVReg = 0;
  unsigned start = getBaseIndex(I->start);
  unsigned end = getBaseIndex(I->end-1) + InstrSlots::NUM;

  // First collect all the def / use in this live range that will be rewritten.
  // Make sure they are sorted according to instruction index.
  std::vector<RewriteInfo> RewriteMIs;
  for (MachineRegisterInfo::reg_iterator ri = mri_->reg_begin(li.reg),
         re = mri_->reg_end(); ri != re; ) {
    MachineInstr *MI = &*ri;
    MachineOperand &O = ri.getOperand();
    ++ri;
    assert(!O.isImplicit() && "Spilling register that's used as implicit use?");
    unsigned index = getInstructionIndex(MI);
    if (index < start || index >= end)
      continue;
    if (O.isUse() && !li.liveAt(getUseIndex(index)))
      // Must be defined by an implicit def. It should not be spilled. Note,
      // this is for correctness reason. e.g.
      // 8   %reg1024<def> = IMPLICIT_DEF
      // 12  %reg1024<def> = INSERT_SUBREG %reg1024<kill>, %reg1025, 2
      // The live range [12, 14) are not part of the r1024 live interval since
      // it's defined by an implicit def. It will not conflicts with live
      // interval of r1025. Now suppose both registers are spilled, you can
      // easily see a situation where both registers are reloaded before
      // the INSERT_SUBREG and both target registers that would overlap.
      continue;
    RewriteMIs.push_back(RewriteInfo(index, MI, O.isUse(), O.isDef()));
  }
  std::sort(RewriteMIs.begin(), RewriteMIs.end(), RewriteInfoCompare());

  unsigned ImpUse = DefIsReMat ? getReMatImplicitUse(li, ReMatDefMI) : 0;
  // Now rewrite the defs and uses.
  for (unsigned i = 0, e = RewriteMIs.size(); i != e; ) {
    RewriteInfo &rwi = RewriteMIs[i];
    ++i;
    unsigned index = rwi.Index;
    bool MIHasUse = rwi.HasUse;
    bool MIHasDef = rwi.HasDef;
    MachineInstr *MI = rwi.MI;
    // If MI def and/or use the same register multiple times, then there
    // are multiple entries.
    unsigned NumUses = MIHasUse;
    while (i != e && RewriteMIs[i].MI == MI) {
      assert(RewriteMIs[i].Index == index);
      bool isUse = RewriteMIs[i].HasUse;
      if (isUse) ++NumUses;
      MIHasUse |= isUse;
      MIHasDef |= RewriteMIs[i].HasDef;
      ++i;
    }
    MachineBasicBlock *MBB = MI->getParent();

    if (ImpUse && MI != ReMatDefMI) {
      // Re-matting an instruction with virtual register use. Update the
      // register interval's spill weight to HUGE_VALF to prevent it from
      // being spilled.
      LiveInterval &ImpLi = getInterval(ImpUse);
      ImpLi.weight = HUGE_VALF;
    }

    unsigned MBBId = MBB->getNumber();
    unsigned ThisVReg = 0;
    if (TrySplit) {
      DenseMap<unsigned,unsigned>::iterator NVI = MBBVRegsMap.find(MBBId);
      if (NVI != MBBVRegsMap.end()) {
        ThisVReg = NVI->second;
        // One common case:
        // x = use
        // ...
        // ...
        // def = ...
        //     = use
        // It's better to start a new interval to avoid artifically
        // extend the new interval.
        if (MIHasDef && !MIHasUse) {
          MBBVRegsMap.erase(MBB->getNumber());
          ThisVReg = 0;
        }
      }
    }

    bool IsNew = ThisVReg == 0;
    if (IsNew) {
      // This ends the previous live interval. If all of its def / use
      // can be folded, give it a low spill weight.
      if (NewVReg && TrySplit && AllCanFold) {
        LiveInterval &nI = getOrCreateInterval(NewVReg);
        nI.weight /= 10.0F;
      }
      AllCanFold = true;
    }
    NewVReg = ThisVReg;

    bool HasDef = false;
    bool HasUse = false;
    bool CanFold = rewriteInstructionForSpills(li, I->valno, TrySplit,
                         index, end, MI, ReMatOrigDefMI, ReMatDefMI,
                         Slot, LdSlot, isLoad, isLoadSS, DefIsReMat,
                         CanDelete, vrm, rc, ReMatIds, loopInfo, NewVReg,
                         ImpUse, HasDef, HasUse, MBBVRegsMap, NewLIs, SSWeight);
    if (!HasDef && !HasUse)
      continue;

    AllCanFold &= CanFold;

    // Update weight of spill interval.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    if (!TrySplit) {
      // The spill weight is now infinity as it cannot be spilled again.
      nI.weight = HUGE_VALF;
      continue;
    }

    // Keep track of the last def and first use in each MBB.
    if (HasDef) {
      if (MI != ReMatOrigDefMI || !CanDelete) {
        bool HasKill = false;
        if (!HasUse)
          HasKill = anyKillInMBBAfterIdx(li, I->valno, MBB, getDefIndex(index));
        else {
          // If this is a two-address code, then this index starts a new VNInfo.
          const VNInfo *VNI = li.findDefinedVNInfo(getDefIndex(index));
          if (VNI)
            HasKill = anyKillInMBBAfterIdx(li, VNI, MBB, getDefIndex(index));
        }
        DenseMap<unsigned, std::vector<SRInfo> >::iterator SII =
          SpillIdxes.find(MBBId);
        if (!HasKill) {
          if (SII == SpillIdxes.end()) {
            std::vector<SRInfo> S;
            S.push_back(SRInfo(index, NewVReg, true));
            SpillIdxes.insert(std::make_pair(MBBId, S));
          } else if (SII->second.back().vreg != NewVReg) {
            SII->second.push_back(SRInfo(index, NewVReg, true));
          } else if ((int)index > SII->second.back().index) {
            // If there is an earlier def and this is a two-address
            // instruction, then it's not possible to fold the store (which
            // would also fold the load).
            SRInfo &Info = SII->second.back();
            Info.index = index;
            Info.canFold = !HasUse;
          }
          SpillMBBs.set(MBBId);
        } else if (SII != SpillIdxes.end() &&
                   SII->second.back().vreg == NewVReg &&
                   (int)index > SII->second.back().index) {
          // There is an earlier def that's not killed (must be two-address).
          // The spill is no longer needed.
          SII->second.pop_back();
          if (SII->second.empty()) {
            SpillIdxes.erase(MBBId);
            SpillMBBs.reset(MBBId);
          }
        }
      }
    }

    if (HasUse) {
      DenseMap<unsigned, std::vector<SRInfo> >::iterator SII =
        SpillIdxes.find(MBBId);
      if (SII != SpillIdxes.end() &&
          SII->second.back().vreg == NewVReg &&
          (int)index > SII->second.back().index)
        // Use(s) following the last def, it's not safe to fold the spill.
        SII->second.back().canFold = false;
      DenseMap<unsigned, std::vector<SRInfo> >::iterator RII =
        RestoreIdxes.find(MBBId);
      if (RII != RestoreIdxes.end() && RII->second.back().vreg == NewVReg)
        // If we are splitting live intervals, only fold if it's the first
        // use and there isn't another use later in the MBB.
        RII->second.back().canFold = false;
      else if (IsNew) {
        // Only need a reload if there isn't an earlier def / use.
        if (RII == RestoreIdxes.end()) {
          std::vector<SRInfo> Infos;
          Infos.push_back(SRInfo(index, NewVReg, true));
          RestoreIdxes.insert(std::make_pair(MBBId, Infos));
        } else {
          RII->second.push_back(SRInfo(index, NewVReg, true));
        }
        RestoreMBBs.set(MBBId);
      }
    }

    // Update spill weight.
    unsigned loopDepth = loopInfo->getLoopDepth(MBB);
    nI.weight += getSpillWeight(HasDef, HasUse, loopDepth);
  }

  if (NewVReg && TrySplit && AllCanFold) {
    // If all of its def / use can be folded, give it a low spill weight.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    nI.weight /= 10.0F;
  }
}

bool LiveIntervals::alsoFoldARestore(int Id, int index, unsigned vr,
                        BitVector &RestoreMBBs,
                        DenseMap<unsigned,std::vector<SRInfo> > &RestoreIdxes) {
  if (!RestoreMBBs[Id])
    return false;
  std::vector<SRInfo> &Restores = RestoreIdxes[Id];
  for (unsigned i = 0, e = Restores.size(); i != e; ++i)
    if (Restores[i].index == index &&
        Restores[i].vreg == vr &&
        Restores[i].canFold)
      return true;
  return false;
}

void LiveIntervals::eraseRestoreInfo(int Id, int index, unsigned vr,
                        BitVector &RestoreMBBs,
                        DenseMap<unsigned,std::vector<SRInfo> > &RestoreIdxes) {
  if (!RestoreMBBs[Id])
    return;
  std::vector<SRInfo> &Restores = RestoreIdxes[Id];
  for (unsigned i = 0, e = Restores.size(); i != e; ++i)
    if (Restores[i].index == index && Restores[i].vreg)
      Restores[i].index = -1;
}

/// handleSpilledImpDefs - Remove IMPLICIT_DEF instructions which are being
/// spilled and create empty intervals for their uses.
void
LiveIntervals::handleSpilledImpDefs(const LiveInterval &li, VirtRegMap &vrm,
                                    const TargetRegisterClass* rc,
                                    std::vector<LiveInterval*> &NewLIs) {
  for (MachineRegisterInfo::reg_iterator ri = mri_->reg_begin(li.reg),
         re = mri_->reg_end(); ri != re; ) {
    MachineOperand &O = ri.getOperand();
    MachineInstr *MI = &*ri;
    ++ri;
    if (O.isDef()) {
      assert(MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF &&
             "Register def was not rewritten?");
      RemoveMachineInstrFromMaps(MI);
      vrm.RemoveMachineInstrFromMaps(MI);
      MI->eraseFromParent();
    } else {
      // This must be an use of an implicit_def so it's not part of the live
      // interval. Create a new empty live interval for it.
      // FIXME: Can we simply erase some of the instructions? e.g. Stores?
      unsigned NewVReg = mri_->createVirtualRegister(rc);
      vrm.grow();
      vrm.setIsImplicitlyDefined(NewVReg);
      NewLIs.push_back(&getOrCreateInterval(NewVReg));
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (MO.isReg() && MO.getReg() == li.reg)
          MO.setReg(NewVReg);
      }
    }
  }
}

namespace {
  struct LISorter {
    bool operator()(LiveInterval* A, LiveInterval* B) {
      return A->beginNumber() < B->beginNumber();
    }
  };
}

std::vector<LiveInterval*> LiveIntervals::
addIntervalsForSpillsFast(const LiveInterval &li,
                          const MachineLoopInfo *loopInfo,
                          VirtRegMap &vrm, float& SSWeight) {
  unsigned slot = vrm.assignVirt2StackSlot(li.reg);

  std::vector<LiveInterval*> added;

  assert(li.weight != HUGE_VALF &&
         "attempt to spill already spilled interval!");

  DOUT << "\t\t\t\tadding intervals for spills for interval: ";
  DEBUG(li.dump());
  DOUT << '\n';

  const TargetRegisterClass* rc = mri_->getRegClass(li.reg);

  SSWeight = 0.0f;

  MachineRegisterInfo::reg_iterator RI = mri_->reg_begin(li.reg);
  while (RI != mri_->reg_end()) {
    MachineInstr* MI = &*RI;
    
    SmallVector<unsigned, 2> Indices;
    bool HasUse = false;
    bool HasDef = false;
    
    for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
      MachineOperand& mop = MI->getOperand(i);
      if (!mop.isReg() || mop.getReg() != li.reg) continue;
      
      HasUse |= MI->getOperand(i).isUse();
      HasDef |= MI->getOperand(i).isDef();
      
      Indices.push_back(i);
    }
    
    if (!tryFoldMemoryOperand(MI, vrm, NULL, getInstructionIndex(MI),
                              Indices, true, slot, li.reg)) {
      unsigned NewVReg = mri_->createVirtualRegister(rc);
      vrm.grow();
      vrm.assignVirt2StackSlot(NewVReg, slot);
      
      // create a new register for this spill
      LiveInterval &nI = getOrCreateInterval(NewVReg);

      // the spill weight is now infinity as it
      // cannot be spilled again
      nI.weight = HUGE_VALF;
      
      // Rewrite register operands to use the new vreg.
      for (SmallVectorImpl<unsigned>::iterator I = Indices.begin(),
           E = Indices.end(); I != E; ++I) {
        MI->getOperand(*I).setReg(NewVReg);
        
        if (MI->getOperand(*I).isUse())
          MI->getOperand(*I).setIsKill(true);
      }
      
      // Fill in  the new live interval.
      unsigned index = getInstructionIndex(MI);
      if (HasUse) {
        LiveRange LR(getLoadIndex(index), getUseIndex(index),
                     nI.getNextValue(~0U, 0, getVNInfoAllocator()));
        DOUT << " +" << LR;
        nI.addRange(LR);
        vrm.addRestorePoint(NewVReg, MI);
      }
      if (HasDef) {
        LiveRange LR(getDefIndex(index), getStoreIndex(index),
                     nI.getNextValue(~0U, 0, getVNInfoAllocator()));
        DOUT << " +" << LR;
        nI.addRange(LR);
        vrm.addSpillPoint(NewVReg, true, MI);
      }
      
      added.push_back(&nI);
        
      DOUT << "\t\t\t\tadded new interval: ";
      DEBUG(nI.dump());
      DOUT << '\n';
      
      unsigned loopDepth = loopInfo->getLoopDepth(MI->getParent());
      if (HasUse) {
        if (HasDef)
          SSWeight += getSpillWeight(true, true, loopDepth);
        else
          SSWeight += getSpillWeight(false, true, loopDepth);
      } else
        SSWeight += getSpillWeight(true, false, loopDepth);
    }
    
    
    RI = mri_->reg_begin(li.reg);
  }

  // Clients expect the new intervals to be returned in sorted order.
  std::sort(added.begin(), added.end(), LISorter());

  return added;
}

std::vector<LiveInterval*> LiveIntervals::
addIntervalsForSpills(const LiveInterval &li,
                      SmallVectorImpl<LiveInterval*> &SpillIs,
                      const MachineLoopInfo *loopInfo, VirtRegMap &vrm,
                      float &SSWeight) {
  
  if (EnableFastSpilling)
    return addIntervalsForSpillsFast(li, loopInfo, vrm, SSWeight);
  
  assert(li.weight != HUGE_VALF &&
         "attempt to spill already spilled interval!");

  DOUT << "\t\t\t\tadding intervals for spills for interval: ";
  li.print(DOUT, tri_);
  DOUT << '\n';

  // Spill slot weight.
  SSWeight = 0.0f;

  // Each bit specify whether a spill is required in the MBB.
  BitVector SpillMBBs(mf_->getNumBlockIDs());
  DenseMap<unsigned, std::vector<SRInfo> > SpillIdxes;
  BitVector RestoreMBBs(mf_->getNumBlockIDs());
  DenseMap<unsigned, std::vector<SRInfo> > RestoreIdxes;
  DenseMap<unsigned,unsigned> MBBVRegsMap;
  std::vector<LiveInterval*> NewLIs;
  const TargetRegisterClass* rc = mri_->getRegClass(li.reg);

  unsigned NumValNums = li.getNumValNums();
  SmallVector<MachineInstr*, 4> ReMatDefs;
  ReMatDefs.resize(NumValNums, NULL);
  SmallVector<MachineInstr*, 4> ReMatOrigDefs;
  ReMatOrigDefs.resize(NumValNums, NULL);
  SmallVector<int, 4> ReMatIds;
  ReMatIds.resize(NumValNums, VirtRegMap::MAX_STACK_SLOT);
  BitVector ReMatDelete(NumValNums);
  unsigned Slot = VirtRegMap::MAX_STACK_SLOT;

  // Spilling a split live interval. It cannot be split any further. Also,
  // it's also guaranteed to be a single val# / range interval.
  if (vrm.getPreSplitReg(li.reg)) {
    vrm.setIsSplitFromReg(li.reg, 0);
    // Unset the split kill marker on the last use.
    unsigned KillIdx = vrm.getKillPoint(li.reg);
    if (KillIdx) {
      MachineInstr *KillMI = getInstructionFromIndex(KillIdx);
      assert(KillMI && "Last use disappeared?");
      int KillOp = KillMI->findRegisterUseOperandIdx(li.reg, true);
      assert(KillOp != -1 && "Last use disappeared?");
      KillMI->getOperand(KillOp).setIsKill(false);
    }
    vrm.removeKillPoint(li.reg);
    bool DefIsReMat = vrm.isReMaterialized(li.reg);
    Slot = vrm.getStackSlot(li.reg);
    assert(Slot != VirtRegMap::MAX_STACK_SLOT);
    MachineInstr *ReMatDefMI = DefIsReMat ?
      vrm.getReMaterializedMI(li.reg) : NULL;
    int LdSlot = 0;
    bool isLoadSS = DefIsReMat && tii_->isLoadFromStackSlot(ReMatDefMI, LdSlot);
    bool isLoad = isLoadSS ||
      (DefIsReMat && (ReMatDefMI->getDesc().canFoldAsLoad()));
    bool IsFirstRange = true;
    for (LiveInterval::Ranges::const_iterator
           I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
      // If this is a split live interval with multiple ranges, it means there
      // are two-address instructions that re-defined the value. Only the
      // first def can be rematerialized!
      if (IsFirstRange) {
        // Note ReMatOrigDefMI has already been deleted.
        rewriteInstructionsForSpills(li, false, I, NULL, ReMatDefMI,
                             Slot, LdSlot, isLoad, isLoadSS, DefIsReMat,
                             false, vrm, rc, ReMatIds, loopInfo,
                             SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                             MBBVRegsMap, NewLIs, SSWeight);
      } else {
        rewriteInstructionsForSpills(li, false, I, NULL, 0,
                             Slot, 0, false, false, false,
                             false, vrm, rc, ReMatIds, loopInfo,
                             SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                             MBBVRegsMap, NewLIs, SSWeight);
      }
      IsFirstRange = false;
    }

    SSWeight = 0.0f;  // Already accounted for when split.
    handleSpilledImpDefs(li, vrm, rc, NewLIs);
    return NewLIs;
  }

  bool TrySplit = SplitAtBB && !intervalIsInOneMBB(li);
  if (SplitLimit != -1 && (int)numSplits >= SplitLimit)
    TrySplit = false;
  if (TrySplit)
    ++numSplits;
  bool NeedStackSlot = false;
  for (LiveInterval::const_vni_iterator i = li.vni_begin(), e = li.vni_end();
       i != e; ++i) {
    const VNInfo *VNI = *i;
    unsigned VN = VNI->id;
    unsigned DefIdx = VNI->def;
    if (DefIdx == ~1U)
      continue; // Dead val#.
    // Is the def for the val# rematerializable?
    MachineInstr *ReMatDefMI = (DefIdx == ~0u)
      ? 0 : getInstructionFromIndex(DefIdx);
    bool dummy;
    if (ReMatDefMI && isReMaterializable(li, VNI, ReMatDefMI, SpillIs, dummy)) {
      // Remember how to remat the def of this val#.
      ReMatOrigDefs[VN] = ReMatDefMI;
      // Original def may be modified so we have to make a copy here.
      MachineInstr *Clone = mf_->CloneMachineInstr(ReMatDefMI);
      ClonedMIs.push_back(Clone);
      ReMatDefs[VN] = Clone;

      bool CanDelete = true;
      if (VNI->hasPHIKill) {
        // A kill is a phi node, not all of its uses can be rematerialized.
        // It must not be deleted.
        CanDelete = false;
        // Need a stack slot if there is any live range where uses cannot be
        // rematerialized.
        NeedStackSlot = true;
      }
      if (CanDelete)
        ReMatDelete.set(VN);
    } else {
      // Need a stack slot if there is any live range where uses cannot be
      // rematerialized.
      NeedStackSlot = true;
    }
  }

  // One stack slot per live interval.
  if (NeedStackSlot && vrm.getPreSplitReg(li.reg) == 0)
    Slot = vrm.assignVirt2StackSlot(li.reg);

  // Create new intervals and rewrite defs and uses.
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    MachineInstr *ReMatDefMI = ReMatDefs[I->valno->id];
    MachineInstr *ReMatOrigDefMI = ReMatOrigDefs[I->valno->id];
    bool DefIsReMat = ReMatDefMI != NULL;
    bool CanDelete = ReMatDelete[I->valno->id];
    int LdSlot = 0;
    bool isLoadSS = DefIsReMat && tii_->isLoadFromStackSlot(ReMatDefMI, LdSlot);
    bool isLoad = isLoadSS ||
      (DefIsReMat && ReMatDefMI->getDesc().canFoldAsLoad());
    rewriteInstructionsForSpills(li, TrySplit, I, ReMatOrigDefMI, ReMatDefMI,
                               Slot, LdSlot, isLoad, isLoadSS, DefIsReMat,
                               CanDelete, vrm, rc, ReMatIds, loopInfo,
                               SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                               MBBVRegsMap, NewLIs, SSWeight);
  }

  // Insert spills / restores if we are splitting.
  if (!TrySplit) {
    handleSpilledImpDefs(li, vrm, rc, NewLIs);
    return NewLIs;
  }

  SmallPtrSet<LiveInterval*, 4> AddedKill;
  SmallVector<unsigned, 2> Ops;
  if (NeedStackSlot) {
    int Id = SpillMBBs.find_first();
    while (Id != -1) {
      MachineBasicBlock *MBB = mf_->getBlockNumbered(Id);
      unsigned loopDepth = loopInfo->getLoopDepth(MBB);
      std::vector<SRInfo> &spills = SpillIdxes[Id];
      for (unsigned i = 0, e = spills.size(); i != e; ++i) {
        int index = spills[i].index;
        unsigned VReg = spills[i].vreg;
        LiveInterval &nI = getOrCreateInterval(VReg);
        bool isReMat = vrm.isReMaterialized(VReg);
        MachineInstr *MI = getInstructionFromIndex(index);
        bool CanFold = false;
        bool FoundUse = false;
        Ops.clear();
        if (spills[i].canFold) {
          CanFold = true;
          for (unsigned j = 0, ee = MI->getNumOperands(); j != ee; ++j) {
            MachineOperand &MO = MI->getOperand(j);
            if (!MO.isReg() || MO.getReg() != VReg)
              continue;

            Ops.push_back(j);
            if (MO.isDef())
              continue;
            if (isReMat || 
                (!FoundUse && !alsoFoldARestore(Id, index, VReg,
                                                RestoreMBBs, RestoreIdxes))) {
              // MI has two-address uses of the same register. If the use
              // isn't the first and only use in the BB, then we can't fold
              // it. FIXME: Move this to rewriteInstructionsForSpills.
              CanFold = false;
              break;
            }
            FoundUse = true;
          }
        }
        // Fold the store into the def if possible.
        bool Folded = false;
        if (CanFold && !Ops.empty()) {
          if (tryFoldMemoryOperand(MI, vrm, NULL, index, Ops, true, Slot,VReg)){
            Folded = true;
            if (FoundUse > 0) {
              // Also folded uses, do not issue a load.
              eraseRestoreInfo(Id, index, VReg, RestoreMBBs, RestoreIdxes);
              nI.removeRange(getLoadIndex(index), getUseIndex(index)+1);
            }
            nI.removeRange(getDefIndex(index), getStoreIndex(index));
          }
        }

        // Otherwise tell the spiller to issue a spill.
        if (!Folded) {
          LiveRange *LR = &nI.ranges[nI.ranges.size()-1];
          bool isKill = LR->end == getStoreIndex(index);
          if (!MI->registerDefIsDead(nI.reg))
            // No need to spill a dead def.
            vrm.addSpillPoint(VReg, isKill, MI);
          if (isKill)
            AddedKill.insert(&nI);
        }

        // Update spill slot weight.
        if (!isReMat)
          SSWeight += getSpillWeight(true, false, loopDepth);
      }
      Id = SpillMBBs.find_next(Id);
    }
  }

  int Id = RestoreMBBs.find_first();
  while (Id != -1) {
    MachineBasicBlock *MBB = mf_->getBlockNumbered(Id);
    unsigned loopDepth = loopInfo->getLoopDepth(MBB);

    std::vector<SRInfo> &restores = RestoreIdxes[Id];
    for (unsigned i = 0, e = restores.size(); i != e; ++i) {
      int index = restores[i].index;
      if (index == -1)
        continue;
      unsigned VReg = restores[i].vreg;
      LiveInterval &nI = getOrCreateInterval(VReg);
      bool isReMat = vrm.isReMaterialized(VReg);
      MachineInstr *MI = getInstructionFromIndex(index);
      bool CanFold = false;
      Ops.clear();
      if (restores[i].canFold) {
        CanFold = true;
        for (unsigned j = 0, ee = MI->getNumOperands(); j != ee; ++j) {
          MachineOperand &MO = MI->getOperand(j);
          if (!MO.isReg() || MO.getReg() != VReg)
            continue;

          if (MO.isDef()) {
            // If this restore were to be folded, it would have been folded
            // already.
            CanFold = false;
            break;
          }
          Ops.push_back(j);
        }
      }

      // Fold the load into the use if possible.
      bool Folded = false;
      if (CanFold && !Ops.empty()) {
        if (!isReMat)
          Folded = tryFoldMemoryOperand(MI, vrm, NULL,index,Ops,true,Slot,VReg);
        else {
          MachineInstr *ReMatDefMI = vrm.getReMaterializedMI(VReg);
          int LdSlot = 0;
          bool isLoadSS = tii_->isLoadFromStackSlot(ReMatDefMI, LdSlot);
          // If the rematerializable def is a load, also try to fold it.
          if (isLoadSS || ReMatDefMI->getDesc().canFoldAsLoad())
            Folded = tryFoldMemoryOperand(MI, vrm, ReMatDefMI, index,
                                          Ops, isLoadSS, LdSlot, VReg);
          if (!Folded) {
            unsigned ImpUse = getReMatImplicitUse(li, ReMatDefMI);
            if (ImpUse) {
              // Re-matting an instruction with virtual register use. Add the
              // register as an implicit use on the use MI and update the register
              // interval's spill weight to HUGE_VALF to prevent it from being
              // spilled.
              LiveInterval &ImpLi = getInterval(ImpUse);
              ImpLi.weight = HUGE_VALF;
              MI->addOperand(MachineOperand::CreateReg(ImpUse, false, true));
            }
          }
        }
      }
      // If folding is not possible / failed, then tell the spiller to issue a
      // load / rematerialization for us.
      if (Folded)
        nI.removeRange(getLoadIndex(index), getUseIndex(index)+1);
      else
        vrm.addRestorePoint(VReg, MI);

      // Update spill slot weight.
      if (!isReMat)
        SSWeight += getSpillWeight(false, true, loopDepth);
    }
    Id = RestoreMBBs.find_next(Id);
  }

  // Finalize intervals: add kills, finalize spill weights, and filter out
  // dead intervals.
  std::vector<LiveInterval*> RetNewLIs;
  for (unsigned i = 0, e = NewLIs.size(); i != e; ++i) {
    LiveInterval *LI = NewLIs[i];
    if (!LI->empty()) {
      LI->weight /= InstrSlots::NUM * getApproximateInstructionCount(*LI);
      if (!AddedKill.count(LI)) {
        LiveRange *LR = &LI->ranges[LI->ranges.size()-1];
        unsigned LastUseIdx = getBaseIndex(LR->end);
        MachineInstr *LastUse = getInstructionFromIndex(LastUseIdx);
        int UseIdx = LastUse->findRegisterUseOperandIdx(LI->reg, false);
        assert(UseIdx != -1);
        if (LastUse->getOperand(UseIdx).isImplicit() ||
            LastUse->getDesc().getOperandConstraint(UseIdx,TOI::TIED_TO) == -1){
          LastUse->getOperand(UseIdx).setIsKill();
          vrm.addKillPoint(LI->reg, LastUseIdx);
        }
      }
      RetNewLIs.push_back(LI);
    }
  }

  handleSpilledImpDefs(li, vrm, rc, RetNewLIs);
  return RetNewLIs;
}

/// hasAllocatableSuperReg - Return true if the specified physical register has
/// any super register that's allocatable.
bool LiveIntervals::hasAllocatableSuperReg(unsigned Reg) const {
  for (const unsigned* AS = tri_->getSuperRegisters(Reg); *AS; ++AS)
    if (allocatableRegs_[*AS] && hasInterval(*AS))
      return true;
  return false;
}

/// getRepresentativeReg - Find the largest super register of the specified
/// physical register.
unsigned LiveIntervals::getRepresentativeReg(unsigned Reg) const {
  // Find the largest super-register that is allocatable. 
  unsigned BestReg = Reg;
  for (const unsigned* AS = tri_->getSuperRegisters(Reg); *AS; ++AS) {
    unsigned SuperReg = *AS;
    if (!hasAllocatableSuperReg(SuperReg) && hasInterval(SuperReg)) {
      BestReg = SuperReg;
      break;
    }
  }
  return BestReg;
}

/// getNumConflictsWithPhysReg - Return the number of uses and defs of the
/// specified interval that conflicts with the specified physical register.
unsigned LiveIntervals::getNumConflictsWithPhysReg(const LiveInterval &li,
                                                   unsigned PhysReg) const {
  unsigned NumConflicts = 0;
  const LiveInterval &pli = getInterval(getRepresentativeReg(PhysReg));
  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(li.reg),
         E = mri_->reg_end(); I != E; ++I) {
    MachineOperand &O = I.getOperand();
    MachineInstr *MI = O.getParent();
    unsigned Index = getInstructionIndex(MI);
    if (pli.liveAt(Index))
      ++NumConflicts;
  }
  return NumConflicts;
}

/// spillPhysRegAroundRegDefsUses - Spill the specified physical register
/// around all defs and uses of the specified interval.
void LiveIntervals::spillPhysRegAroundRegDefsUses(const LiveInterval &li,
                                            unsigned PhysReg, VirtRegMap &vrm) {
  unsigned SpillReg = getRepresentativeReg(PhysReg);

  for (const unsigned *AS = tri_->getAliasSet(PhysReg); *AS; ++AS)
    // If there are registers which alias PhysReg, but which are not a
    // sub-register of the chosen representative super register. Assert
    // since we can't handle it yet.
    assert(*AS == SpillReg || !allocatableRegs_[*AS] ||
           tri_->isSuperRegister(*AS, SpillReg));

  LiveInterval &pli = getInterval(SpillReg);
  SmallPtrSet<MachineInstr*, 8> SeenMIs;
  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(li.reg),
         E = mri_->reg_end(); I != E; ++I) {
    MachineOperand &O = I.getOperand();
    MachineInstr *MI = O.getParent();
    if (SeenMIs.count(MI))
      continue;
    SeenMIs.insert(MI);
    unsigned Index = getInstructionIndex(MI);
    if (pli.liveAt(Index)) {
      vrm.addEmergencySpill(SpillReg, MI);
      unsigned StartIdx = getLoadIndex(Index);
      unsigned EndIdx = getStoreIndex(Index)+1;
      if (pli.isInOneLiveRange(StartIdx, EndIdx))
        pli.removeRange(StartIdx, EndIdx);
      else {
        cerr << "Ran out of registers during register allocation!\n";
        if (MI->getOpcode() == TargetInstrInfo::INLINEASM) {
          cerr << "Please check your inline asm statement for invalid "
               << "constraints:\n";
          MI->print(cerr.stream(), tm_);
        }
        exit(1);
      }
      for (const unsigned* AS = tri_->getSubRegisters(SpillReg); *AS; ++AS) {
        if (!hasInterval(*AS))
          continue;
        LiveInterval &spli = getInterval(*AS);
        if (spli.liveAt(Index))
          spli.removeRange(getLoadIndex(Index), getStoreIndex(Index)+1);
      }
    }
  }
}

LiveRange LiveIntervals::addLiveRangeToEndOfBlock(unsigned reg,
                                                   MachineInstr* startInst) {
  LiveInterval& Interval = getOrCreateInterval(reg);
  VNInfo* VN = Interval.getNextValue(
            getInstructionIndex(startInst) + InstrSlots::DEF,
            startInst, getVNInfoAllocator());
  VN->hasPHIKill = true;
  VN->kills.push_back(getMBBEndIdx(startInst->getParent()));
  LiveRange LR(getInstructionIndex(startInst) + InstrSlots::DEF,
               getMBBEndIdx(startInst->getParent()) + 1, VN);
  Interval.addRange(LR);
  
  return LR;
}
