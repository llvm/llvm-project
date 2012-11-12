//===- RegisterCoalescer.cpp - Generic Register Coalescing Interface -------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generic RegisterCoalescer interface which
// is used as the common interface used by all clients and
// implementations of register coalescing.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "RegisterCoalescer.h"
#include "LiveDebugVariables.h"
#include "VirtRegMap.h"

#include "llvm/Pass.h"
#include "llvm/Value.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

STATISTIC(numJoins    , "Number of interval joins performed");
STATISTIC(numCrossRCs , "Number of cross class joins performed");
STATISTIC(numCommutes , "Number of instruction commuting performed");
STATISTIC(numExtends  , "Number of copies extended");
STATISTIC(NumReMats   , "Number of instructions re-materialized");
STATISTIC(NumInflated , "Number of register classes inflated");
STATISTIC(NumLaneConflicts, "Number of dead lane conflicts tested");
STATISTIC(NumLaneResolves,  "Number of dead lane conflicts resolved");

static cl::opt<bool>
EnableJoining("join-liveintervals",
              cl::desc("Coalesce copies (default=true)"),
              cl::init(true));

static cl::opt<bool>
VerifyCoalescing("verify-coalescing",
         cl::desc("Verify machine instrs before and after register coalescing"),
         cl::Hidden);

namespace {
  class RegisterCoalescer : public MachineFunctionPass,
                            private LiveRangeEdit::Delegate {
    MachineFunction* MF;
    MachineRegisterInfo* MRI;
    const TargetMachine* TM;
    const TargetRegisterInfo* TRI;
    const TargetInstrInfo* TII;
    LiveIntervals *LIS;
    LiveDebugVariables *LDV;
    const MachineLoopInfo* Loops;
    AliasAnalysis *AA;
    RegisterClassInfo RegClassInfo;

    /// WorkList - Copy instructions yet to be coalesced.
    SmallVector<MachineInstr*, 8> WorkList;

    /// ErasedInstrs - Set of instruction pointers that have been erased, and
    /// that may be present in WorkList.
    SmallPtrSet<MachineInstr*, 8> ErasedInstrs;

    /// Dead instructions that are about to be deleted.
    SmallVector<MachineInstr*, 8> DeadDefs;

    /// Virtual registers to be considered for register class inflation.
    SmallVector<unsigned, 8> InflateRegs;

    /// Recursively eliminate dead defs in DeadDefs.
    void eliminateDeadDefs();

    /// LiveRangeEdit callback.
    void LRE_WillEraseInstruction(MachineInstr *MI);

    /// joinAllIntervals - join compatible live intervals
    void joinAllIntervals();

    /// copyCoalesceInMBB - Coalesce copies in the specified MBB, putting
    /// copies that cannot yet be coalesced into WorkList.
    void copyCoalesceInMBB(MachineBasicBlock *MBB);

    /// copyCoalesceWorkList - Try to coalesce all copies in WorkList after
    /// position From. Return true if any progress was made.
    bool copyCoalesceWorkList(unsigned From = 0);

    /// joinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
    /// which are the src/dst of the copy instruction CopyMI.  This returns
    /// true if the copy was successfully coalesced away. If it is not
    /// currently possible to coalesce this interval, but it may be possible if
    /// other things get coalesced, then it returns true by reference in
    /// 'Again'.
    bool joinCopy(MachineInstr *TheCopy, bool &Again);

    /// joinIntervals - Attempt to join these two intervals.  On failure, this
    /// returns false.  The output "SrcInt" will not have been modified, so we
    /// can use this information below to update aliases.
    bool joinIntervals(CoalescerPair &CP);

    /// Attempt joining two virtual registers. Return true on success.
    bool joinVirtRegs(CoalescerPair &CP);

    /// Attempt joining with a reserved physreg.
    bool joinReservedPhysReg(CoalescerPair &CP);

    /// adjustCopiesBackFrom - We found a non-trivially-coalescable copy. If
    /// the source value number is defined by a copy from the destination reg
    /// see if we can merge these two destination reg valno# into a single
    /// value number, eliminating a copy.
    bool adjustCopiesBackFrom(const CoalescerPair &CP, MachineInstr *CopyMI);

    /// hasOtherReachingDefs - Return true if there are definitions of IntB
    /// other than BValNo val# that can reach uses of AValno val# of IntA.
    bool hasOtherReachingDefs(LiveInterval &IntA, LiveInterval &IntB,
                              VNInfo *AValNo, VNInfo *BValNo);

    /// removeCopyByCommutingDef - We found a non-trivially-coalescable copy.
    /// If the source value number is defined by a commutable instruction and
    /// its other operand is coalesced to the copy dest register, see if we
    /// can transform the copy into a noop by commuting the definition.
    bool removeCopyByCommutingDef(const CoalescerPair &CP,MachineInstr *CopyMI);

    /// reMaterializeTrivialDef - If the source of a copy is defined by a
    /// trivial computation, replace the copy by rematerialize the definition.
    bool reMaterializeTrivialDef(LiveInterval &SrcInt, unsigned DstReg,
                                 MachineInstr *CopyMI);

    /// canJoinPhys - Return true if a physreg copy should be joined.
    bool canJoinPhys(CoalescerPair &CP);

    /// updateRegDefsUses - Replace all defs and uses of SrcReg to DstReg and
    /// update the subregister number if it is not zero. If DstReg is a
    /// physical register and the existing subregister number of the def / use
    /// being updated is not zero, make sure to set it to the correct physical
    /// subregister.
    void updateRegDefsUses(unsigned SrcReg, unsigned DstReg, unsigned SubIdx);

    /// eliminateUndefCopy - Handle copies of undef values.
    bool eliminateUndefCopy(MachineInstr *CopyMI, const CoalescerPair &CP);

  public:
    static char ID; // Class identification, replacement for typeinfo
    RegisterCoalescer() : MachineFunctionPass(ID) {
      initializeRegisterCoalescerPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;

    virtual void releaseMemory();

    /// runOnMachineFunction - pass entry point
    virtual bool runOnMachineFunction(MachineFunction&);

    /// print - Implement the dump method.
    virtual void print(raw_ostream &O, const Module* = 0) const;
  };
} /// end anonymous namespace

char &llvm::RegisterCoalescerID = RegisterCoalescer::ID;

INITIALIZE_PASS_BEGIN(RegisterCoalescer, "simple-register-coalescing",
                      "Simple Register Coalescing", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariables)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(RegisterCoalescer, "simple-register-coalescing",
                    "Simple Register Coalescing", false, false)

char RegisterCoalescer::ID = 0;

static bool isMoveInstr(const TargetRegisterInfo &tri, const MachineInstr *MI,
                        unsigned &Src, unsigned &Dst,
                        unsigned &SrcSub, unsigned &DstSub) {
  if (MI->isCopy()) {
    Dst = MI->getOperand(0).getReg();
    DstSub = MI->getOperand(0).getSubReg();
    Src = MI->getOperand(1).getReg();
    SrcSub = MI->getOperand(1).getSubReg();
  } else if (MI->isSubregToReg()) {
    Dst = MI->getOperand(0).getReg();
    DstSub = tri.composeSubRegIndices(MI->getOperand(0).getSubReg(),
                                      MI->getOperand(3).getImm());
    Src = MI->getOperand(2).getReg();
    SrcSub = MI->getOperand(2).getSubReg();
  } else
    return false;
  return true;
}

bool CoalescerPair::setRegisters(const MachineInstr *MI) {
  SrcReg = DstReg = 0;
  SrcIdx = DstIdx = 0;
  NewRC = 0;
  Flipped = CrossClass = false;

  unsigned Src, Dst, SrcSub, DstSub;
  if (!isMoveInstr(TRI, MI, Src, Dst, SrcSub, DstSub))
    return false;
  Partial = SrcSub || DstSub;

  // If one register is a physreg, it must be Dst.
  if (TargetRegisterInfo::isPhysicalRegister(Src)) {
    if (TargetRegisterInfo::isPhysicalRegister(Dst))
      return false;
    std::swap(Src, Dst);
    std::swap(SrcSub, DstSub);
    Flipped = true;
  }

  const MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();

  if (TargetRegisterInfo::isPhysicalRegister(Dst)) {
    // Eliminate DstSub on a physreg.
    if (DstSub) {
      Dst = TRI.getSubReg(Dst, DstSub);
      if (!Dst) return false;
      DstSub = 0;
    }

    // Eliminate SrcSub by picking a corresponding Dst superregister.
    if (SrcSub) {
      Dst = TRI.getMatchingSuperReg(Dst, SrcSub, MRI.getRegClass(Src));
      if (!Dst) return false;
      SrcSub = 0;
    } else if (!MRI.getRegClass(Src)->contains(Dst)) {
      return false;
    }
  } else {
    // Both registers are virtual.
    const TargetRegisterClass *SrcRC = MRI.getRegClass(Src);
    const TargetRegisterClass *DstRC = MRI.getRegClass(Dst);

    // Both registers have subreg indices.
    if (SrcSub && DstSub) {
      // Copies between different sub-registers are never coalescable.
      if (Src == Dst && SrcSub != DstSub)
        return false;

      NewRC = TRI.getCommonSuperRegClass(SrcRC, SrcSub, DstRC, DstSub,
                                         SrcIdx, DstIdx);
      if (!NewRC)
        return false;
    } else if (DstSub) {
      // SrcReg will be merged with a sub-register of DstReg.
      SrcIdx = DstSub;
      NewRC = TRI.getMatchingSuperRegClass(DstRC, SrcRC, DstSub);
    } else if (SrcSub) {
      // DstReg will be merged with a sub-register of SrcReg.
      DstIdx = SrcSub;
      NewRC = TRI.getMatchingSuperRegClass(SrcRC, DstRC, SrcSub);
    } else {
      // This is a straight copy without sub-registers.
      NewRC = TRI.getCommonSubClass(DstRC, SrcRC);
    }

    // The combined constraint may be impossible to satisfy.
    if (!NewRC)
      return false;

    // Prefer SrcReg to be a sub-register of DstReg.
    // FIXME: Coalescer should support subregs symmetrically.
    if (DstIdx && !SrcIdx) {
      std::swap(Src, Dst);
      std::swap(SrcIdx, DstIdx);
      Flipped = !Flipped;
    }

    CrossClass = NewRC != DstRC || NewRC != SrcRC;
  }
  // Check our invariants
  assert(TargetRegisterInfo::isVirtualRegister(Src) && "Src must be virtual");
  assert(!(TargetRegisterInfo::isPhysicalRegister(Dst) && DstSub) &&
         "Cannot have a physical SubIdx");
  SrcReg = Src;
  DstReg = Dst;
  return true;
}

bool CoalescerPair::flip() {
  if (TargetRegisterInfo::isPhysicalRegister(DstReg))
    return false;
  std::swap(SrcReg, DstReg);
  std::swap(SrcIdx, DstIdx);
  Flipped = !Flipped;
  return true;
}

bool CoalescerPair::isCoalescable(const MachineInstr *MI) const {
  if (!MI)
    return false;
  unsigned Src, Dst, SrcSub, DstSub;
  if (!isMoveInstr(TRI, MI, Src, Dst, SrcSub, DstSub))
    return false;

  // Find the virtual register that is SrcReg.
  if (Dst == SrcReg) {
    std::swap(Src, Dst);
    std::swap(SrcSub, DstSub);
  } else if (Src != SrcReg) {
    return false;
  }

  // Now check that Dst matches DstReg.
  if (TargetRegisterInfo::isPhysicalRegister(DstReg)) {
    if (!TargetRegisterInfo::isPhysicalRegister(Dst))
      return false;
    assert(!DstIdx && !SrcIdx && "Inconsistent CoalescerPair state.");
    // DstSub could be set for a physreg from INSERT_SUBREG.
    if (DstSub)
      Dst = TRI.getSubReg(Dst, DstSub);
    // Full copy of Src.
    if (!SrcSub)
      return DstReg == Dst;
    // This is a partial register copy. Check that the parts match.
    return TRI.getSubReg(DstReg, SrcSub) == Dst;
  } else {
    // DstReg is virtual.
    if (DstReg != Dst)
      return false;
    // Registers match, do the subregisters line up?
    return TRI.composeSubRegIndices(SrcIdx, SrcSub) ==
           TRI.composeSubRegIndices(DstIdx, DstSub);
  }
}

void RegisterCoalescer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  AU.addRequired<LiveDebugVariables>();
  AU.addPreserved<LiveDebugVariables>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addPreservedID(MachineDominatorsID);
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RegisterCoalescer::eliminateDeadDefs() {
  SmallVector<LiveInterval*, 8> NewRegs;
  LiveRangeEdit(0, NewRegs, *MF, *LIS, 0, this).eliminateDeadDefs(DeadDefs);
}

// Callback from eliminateDeadDefs().
void RegisterCoalescer::LRE_WillEraseInstruction(MachineInstr *MI) {
  // MI may be in WorkList. Make sure we don't visit it.
  ErasedInstrs.insert(MI);
}

/// adjustCopiesBackFrom - We found a non-trivially-coalescable copy with IntA
/// being the source and IntB being the dest, thus this defines a value number
/// in IntB.  If the source value number (in IntA) is defined by a copy from B,
/// see if we can merge these two pieces of B into a single value number,
/// eliminating a copy.  For example:
///
///  A3 = B0
///    ...
///  B1 = A3      <- this copy
///
/// In this case, B0 can be extended to where the B1 copy lives, allowing the B1
/// value number to be replaced with B0 (which simplifies the B liveinterval).
///
/// This returns true if an interval was modified.
///
bool RegisterCoalescer::adjustCopiesBackFrom(const CoalescerPair &CP,
                                             MachineInstr *CopyMI) {
  assert(!CP.isPartial() && "This doesn't work for partial copies.");
  assert(!CP.isPhys() && "This doesn't work for physreg copies.");

  LiveInterval &IntA =
    LIS->getInterval(CP.isFlipped() ? CP.getDstReg() : CP.getSrcReg());
  LiveInterval &IntB =
    LIS->getInterval(CP.isFlipped() ? CP.getSrcReg() : CP.getDstReg());
  SlotIndex CopyIdx = LIS->getInstructionIndex(CopyMI).getRegSlot();

  // BValNo is a value number in B that is defined by a copy from A.  'B3' in
  // the example above.
  LiveInterval::iterator BLR = IntB.FindLiveRangeContaining(CopyIdx);
  if (BLR == IntB.end()) return false;
  VNInfo *BValNo = BLR->valno;

  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we
  // can't process it.
  if (BValNo->def != CopyIdx) return false;

  // AValNo is the value number in A that defines the copy, A3 in the example.
  SlotIndex CopyUseIdx = CopyIdx.getRegSlot(true);
  LiveInterval::iterator ALR = IntA.FindLiveRangeContaining(CopyUseIdx);
  // The live range might not exist after fun with physreg coalescing.
  if (ALR == IntA.end()) return false;
  VNInfo *AValNo = ALR->valno;

  // If AValNo is defined as a copy from IntB, we can potentially process this.
  // Get the instruction that defines this value number.
  MachineInstr *ACopyMI = LIS->getInstructionFromIndex(AValNo->def);
  // Don't allow any partial copies, even if isCoalescable() allows them.
  if (!CP.isCoalescable(ACopyMI) || !ACopyMI->isFullCopy())
    return false;

  // Get the LiveRange in IntB that this value number starts with.
  LiveInterval::iterator ValLR =
    IntB.FindLiveRangeContaining(AValNo->def.getPrevSlot());
  if (ValLR == IntB.end())
    return false;

  // Make sure that the end of the live range is inside the same block as
  // CopyMI.
  MachineInstr *ValLREndInst =
    LIS->getInstructionFromIndex(ValLR->end.getPrevSlot());
  if (!ValLREndInst || ValLREndInst->getParent() != CopyMI->getParent())
    return false;

  // Okay, we now know that ValLR ends in the same block that the CopyMI
  // live-range starts.  If there are no intervening live ranges between them in
  // IntB, we can merge them.
  if (ValLR+1 != BLR) return false;

  DEBUG(dbgs() << "Extending: " << PrintReg(IntB.reg, TRI));

  SlotIndex FillerStart = ValLR->end, FillerEnd = BLR->start;
  // We are about to delete CopyMI, so need to remove it as the 'instruction
  // that defines this value #'. Update the valnum with the new defining
  // instruction #.
  BValNo->def = FillerStart;

  // Okay, we can merge them.  We need to insert a new liverange:
  // [ValLR.end, BLR.begin) of either value number, then we merge the
  // two value numbers.
  IntB.addRange(LiveRange(FillerStart, FillerEnd, BValNo));

  // Okay, merge "B1" into the same value number as "B0".
  if (BValNo != ValLR->valno)
    IntB.MergeValueNumberInto(BValNo, ValLR->valno);
  DEBUG(dbgs() << "   result = " << IntB << '\n');

  // If the source instruction was killing the source register before the
  // merge, unset the isKill marker given the live range has been extended.
  int UIdx = ValLREndInst->findRegisterUseOperandIdx(IntB.reg, true);
  if (UIdx != -1) {
    ValLREndInst->getOperand(UIdx).setIsKill(false);
  }

  // Rewrite the copy. If the copy instruction was killing the destination
  // register before the merge, find the last use and trim the live range. That
  // will also add the isKill marker.
  CopyMI->substituteRegister(IntA.reg, IntB.reg, 0, *TRI);
  if (ALR->end == CopyIdx)
    LIS->shrinkToUses(&IntA);

  ++numExtends;
  return true;
}

/// hasOtherReachingDefs - Return true if there are definitions of IntB
/// other than BValNo val# that can reach uses of AValno val# of IntA.
bool RegisterCoalescer::hasOtherReachingDefs(LiveInterval &IntA,
                                             LiveInterval &IntB,
                                             VNInfo *AValNo,
                                             VNInfo *BValNo) {
  // If AValNo has PHI kills, conservatively assume that IntB defs can reach
  // the PHI values.
  if (LIS->hasPHIKill(IntA, AValNo))
    return true;

  for (LiveInterval::iterator AI = IntA.begin(), AE = IntA.end();
       AI != AE; ++AI) {
    if (AI->valno != AValNo) continue;
    LiveInterval::Ranges::iterator BI =
      std::upper_bound(IntB.ranges.begin(), IntB.ranges.end(), AI->start);
    if (BI != IntB.ranges.begin())
      --BI;
    for (; BI != IntB.ranges.end() && AI->end >= BI->start; ++BI) {
      if (BI->valno == BValNo)
        continue;
      if (BI->start <= AI->start && BI->end > AI->start)
        return true;
      if (BI->start > AI->start && BI->start < AI->end)
        return true;
    }
  }
  return false;
}

/// removeCopyByCommutingDef - We found a non-trivially-coalescable copy with
/// IntA being the source and IntB being the dest, thus this defines a value
/// number in IntB.  If the source value number (in IntA) is defined by a
/// commutable instruction and its other operand is coalesced to the copy dest
/// register, see if we can transform the copy into a noop by commuting the
/// definition. For example,
///
///  A3 = op A2 B0<kill>
///    ...
///  B1 = A3      <- this copy
///    ...
///     = op A3   <- more uses
///
/// ==>
///
///  B2 = op B0 A2<kill>
///    ...
///  B1 = B2      <- now an identify copy
///    ...
///     = op B2   <- more uses
///
/// This returns true if an interval was modified.
///
bool RegisterCoalescer::removeCopyByCommutingDef(const CoalescerPair &CP,
                                                 MachineInstr *CopyMI) {
  assert (!CP.isPhys());

  SlotIndex CopyIdx = LIS->getInstructionIndex(CopyMI).getRegSlot();

  LiveInterval &IntA =
    LIS->getInterval(CP.isFlipped() ? CP.getDstReg() : CP.getSrcReg());
  LiveInterval &IntB =
    LIS->getInterval(CP.isFlipped() ? CP.getSrcReg() : CP.getDstReg());

  // BValNo is a value number in B that is defined by a copy from A. 'B3' in
  // the example above.
  VNInfo *BValNo = IntB.getVNInfoAt(CopyIdx);
  if (!BValNo || BValNo->def != CopyIdx)
    return false;

  assert(BValNo->def == CopyIdx && "Copy doesn't define the value?");

  // AValNo is the value number in A that defines the copy, A3 in the example.
  VNInfo *AValNo = IntA.getVNInfoAt(CopyIdx.getRegSlot(true));
  assert(AValNo && "COPY source not live");
  if (AValNo->isPHIDef() || AValNo->isUnused())
    return false;
  MachineInstr *DefMI = LIS->getInstructionFromIndex(AValNo->def);
  if (!DefMI)
    return false;
  if (!DefMI->isCommutable())
    return false;
  // If DefMI is a two-address instruction then commuting it will change the
  // destination register.
  int DefIdx = DefMI->findRegisterDefOperandIdx(IntA.reg);
  assert(DefIdx != -1);
  unsigned UseOpIdx;
  if (!DefMI->isRegTiedToUseOperand(DefIdx, &UseOpIdx))
    return false;
  unsigned Op1, Op2, NewDstIdx;
  if (!TII->findCommutedOpIndices(DefMI, Op1, Op2))
    return false;
  if (Op1 == UseOpIdx)
    NewDstIdx = Op2;
  else if (Op2 == UseOpIdx)
    NewDstIdx = Op1;
  else
    return false;

  MachineOperand &NewDstMO = DefMI->getOperand(NewDstIdx);
  unsigned NewReg = NewDstMO.getReg();
  if (NewReg != IntB.reg || !LiveRangeQuery(IntB, AValNo->def).isKill())
    return false;

  // Make sure there are no other definitions of IntB that would reach the
  // uses which the new definition can reach.
  if (hasOtherReachingDefs(IntA, IntB, AValNo, BValNo))
    return false;

  // If some of the uses of IntA.reg is already coalesced away, return false.
  // It's not possible to determine whether it's safe to perform the coalescing.
  for (MachineRegisterInfo::use_nodbg_iterator UI =
         MRI->use_nodbg_begin(IntA.reg),
       UE = MRI->use_nodbg_end(); UI != UE; ++UI) {
    MachineInstr *UseMI = &*UI;
    SlotIndex UseIdx = LIS->getInstructionIndex(UseMI);
    LiveInterval::iterator ULR = IntA.FindLiveRangeContaining(UseIdx);
    if (ULR == IntA.end() || ULR->valno != AValNo)
      continue;
    // If this use is tied to a def, we can't rewrite the register.
    if (UseMI->isRegTiedToDefOperand(UI.getOperandNo()))
      return false;
  }

  DEBUG(dbgs() << "\tremoveCopyByCommutingDef: " << AValNo->def << '\t'
               << *DefMI);

  // At this point we have decided that it is legal to do this
  // transformation.  Start by commuting the instruction.
  MachineBasicBlock *MBB = DefMI->getParent();
  MachineInstr *NewMI = TII->commuteInstruction(DefMI);
  if (!NewMI)
    return false;
  if (TargetRegisterInfo::isVirtualRegister(IntA.reg) &&
      TargetRegisterInfo::isVirtualRegister(IntB.reg) &&
      !MRI->constrainRegClass(IntB.reg, MRI->getRegClass(IntA.reg)))
    return false;
  if (NewMI != DefMI) {
    LIS->ReplaceMachineInstrInMaps(DefMI, NewMI);
    MachineBasicBlock::iterator Pos = DefMI;
    MBB->insert(Pos, NewMI);
    MBB->erase(DefMI);
  }
  unsigned OpIdx = NewMI->findRegisterUseOperandIdx(IntA.reg, false);
  NewMI->getOperand(OpIdx).setIsKill();

  // If ALR and BLR overlaps and end of BLR extends beyond end of ALR, e.g.
  // A = or A, B
  // ...
  // B = A
  // ...
  // C = A<kill>
  // ...
  //   = B

  // Update uses of IntA of the specific Val# with IntB.
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(IntA.reg),
         UE = MRI->use_end(); UI != UE;) {
    MachineOperand &UseMO = UI.getOperand();
    MachineInstr *UseMI = &*UI;
    ++UI;
    if (UseMI->isDebugValue()) {
      // FIXME These don't have an instruction index.  Not clear we have enough
      // info to decide whether to do this replacement or not.  For now do it.
      UseMO.setReg(NewReg);
      continue;
    }
    SlotIndex UseIdx = LIS->getInstructionIndex(UseMI).getRegSlot(true);
    LiveInterval::iterator ULR = IntA.FindLiveRangeContaining(UseIdx);
    if (ULR == IntA.end() || ULR->valno != AValNo)
      continue;
    // Kill flags are no longer accurate. They are recomputed after RA.
    UseMO.setIsKill(false);
    if (TargetRegisterInfo::isPhysicalRegister(NewReg))
      UseMO.substPhysReg(NewReg, *TRI);
    else
      UseMO.setReg(NewReg);
    if (UseMI == CopyMI)
      continue;
    if (!UseMI->isCopy())
      continue;
    if (UseMI->getOperand(0).getReg() != IntB.reg ||
        UseMI->getOperand(0).getSubReg())
      continue;

    // This copy will become a noop. If it's defining a new val#, merge it into
    // BValNo.
    SlotIndex DefIdx = UseIdx.getRegSlot();
    VNInfo *DVNI = IntB.getVNInfoAt(DefIdx);
    if (!DVNI)
      continue;
    DEBUG(dbgs() << "\t\tnoop: " << DefIdx << '\t' << *UseMI);
    assert(DVNI->def == DefIdx);
    BValNo = IntB.MergeValueNumberInto(BValNo, DVNI);
    ErasedInstrs.insert(UseMI);
    LIS->RemoveMachineInstrFromMaps(UseMI);
    UseMI->eraseFromParent();
  }

  // Extend BValNo by merging in IntA live ranges of AValNo. Val# definition
  // is updated.
  VNInfo *ValNo = BValNo;
  ValNo->def = AValNo->def;
  for (LiveInterval::iterator AI = IntA.begin(), AE = IntA.end();
       AI != AE; ++AI) {
    if (AI->valno != AValNo) continue;
    IntB.addRange(LiveRange(AI->start, AI->end, ValNo));
  }
  DEBUG(dbgs() << "\t\textended: " << IntB << '\n');

  IntA.removeValNo(AValNo);
  DEBUG(dbgs() << "\t\ttrimmed:  " << IntA << '\n');
  ++numCommutes;
  return true;
}

/// reMaterializeTrivialDef - If the source of a copy is defined by a trivial
/// computation, replace the copy by rematerialize the definition.
bool RegisterCoalescer::reMaterializeTrivialDef(LiveInterval &SrcInt,
                                                unsigned DstReg,
                                                MachineInstr *CopyMI) {
  SlotIndex CopyIdx = LIS->getInstructionIndex(CopyMI).getRegSlot(true);
  LiveInterval::iterator SrcLR = SrcInt.FindLiveRangeContaining(CopyIdx);
  assert(SrcLR != SrcInt.end() && "Live range not found!");
  VNInfo *ValNo = SrcLR->valno;
  if (ValNo->isPHIDef() || ValNo->isUnused())
    return false;
  MachineInstr *DefMI = LIS->getInstructionFromIndex(ValNo->def);
  if (!DefMI)
    return false;
  assert(DefMI && "Defining instruction disappeared");
  if (!DefMI->isAsCheapAsAMove())
    return false;
  if (!TII->isTriviallyReMaterializable(DefMI, AA))
    return false;
  bool SawStore = false;
  if (!DefMI->isSafeToMove(TII, AA, SawStore))
    return false;
  const MCInstrDesc &MCID = DefMI->getDesc();
  if (MCID.getNumDefs() != 1)
    return false;
  if (!DefMI->isImplicitDef()) {
    // Make sure the copy destination register class fits the instruction
    // definition register class. The mismatch can happen as a result of earlier
    // extract_subreg, insert_subreg, subreg_to_reg coalescing.
    const TargetRegisterClass *RC = TII->getRegClass(MCID, 0, TRI, *MF);
    if (TargetRegisterInfo::isVirtualRegister(DstReg)) {
      if (MRI->getRegClass(DstReg) != RC)
        return false;
    } else if (!RC->contains(DstReg))
      return false;
  }

  MachineBasicBlock *MBB = CopyMI->getParent();
  MachineBasicBlock::iterator MII =
    llvm::next(MachineBasicBlock::iterator(CopyMI));
  TII->reMaterialize(*MBB, MII, DstReg, 0, DefMI, *TRI);
  MachineInstr *NewMI = prior(MII);

  // NewMI may have dead implicit defs (E.g. EFLAGS for MOV<bits>r0 on X86).
  // We need to remember these so we can add intervals once we insert
  // NewMI into SlotIndexes.
  SmallVector<unsigned, 4> NewMIImplDefs;
  for (unsigned i = NewMI->getDesc().getNumOperands(),
         e = NewMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = NewMI->getOperand(i);
    if (MO.isReg()) {
      assert(MO.isDef() && MO.isImplicit() && MO.isDead() &&
             TargetRegisterInfo::isPhysicalRegister(MO.getReg()));
      NewMIImplDefs.push_back(MO.getReg());
    }
  }

  // CopyMI may have implicit operands, transfer them over to the newly
  // rematerialized instruction. And update implicit def interval valnos.
  for (unsigned i = CopyMI->getDesc().getNumOperands(),
         e = CopyMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = CopyMI->getOperand(i);
    if (MO.isReg()) {
      assert(MO.isImplicit() && "No explicit operands after implict operands.");
      // Discard VReg implicit defs.
      if (TargetRegisterInfo::isPhysicalRegister(MO.getReg())) {
        NewMI->addOperand(MO);
      }
    }
  }

  LIS->ReplaceMachineInstrInMaps(CopyMI, NewMI);

  SlotIndex NewMIIdx = LIS->getInstructionIndex(NewMI);
  for (unsigned i = 0, e = NewMIImplDefs.size(); i != e; ++i) {
    unsigned Reg = NewMIImplDefs[i];
    for (MCRegUnitIterator Units(Reg, TRI); Units.isValid(); ++Units)
      if (LiveInterval *LI = LIS->getCachedRegUnit(*Units))
        LI->createDeadDef(NewMIIdx.getRegSlot(), LIS->getVNInfoAllocator());
  }

  CopyMI->eraseFromParent();
  ErasedInstrs.insert(CopyMI);
  DEBUG(dbgs() << "Remat: " << *NewMI);
  ++NumReMats;

  // The source interval can become smaller because we removed a use.
  LIS->shrinkToUses(&SrcInt, &DeadDefs);
  if (!DeadDefs.empty())
    eliminateDeadDefs();

  return true;
}

/// eliminateUndefCopy - ProcessImpicitDefs may leave some copies of <undef>
/// values, it only removes local variables. When we have a copy like:
///
///   %vreg1 = COPY %vreg2<undef>
///
/// We delete the copy and remove the corresponding value number from %vreg1.
/// Any uses of that value number are marked as <undef>.
bool RegisterCoalescer::eliminateUndefCopy(MachineInstr *CopyMI,
                                           const CoalescerPair &CP) {
  SlotIndex Idx = LIS->getInstructionIndex(CopyMI);
  LiveInterval *SrcInt = &LIS->getInterval(CP.getSrcReg());
  if (SrcInt->liveAt(Idx))
    return false;
  LiveInterval *DstInt = &LIS->getInterval(CP.getDstReg());
  if (DstInt->liveAt(Idx))
    return false;

  // No intervals are live-in to CopyMI - it is undef.
  if (CP.isFlipped())
    DstInt = SrcInt;
  SrcInt = 0;

  VNInfo *DeadVNI = DstInt->getVNInfoAt(Idx.getRegSlot());
  assert(DeadVNI && "No value defined in DstInt");
  DstInt->removeValNo(DeadVNI);

  // Find new undef uses.
  for (MachineRegisterInfo::reg_nodbg_iterator
         I = MRI->reg_nodbg_begin(DstInt->reg), E = MRI->reg_nodbg_end();
       I != E; ++I) {
    MachineOperand &MO = I.getOperand();
    if (MO.isDef() || MO.isUndef())
      continue;
    MachineInstr *MI = MO.getParent();
    SlotIndex Idx = LIS->getInstructionIndex(MI);
    if (DstInt->liveAt(Idx))
      continue;
    MO.setIsUndef(true);
    DEBUG(dbgs() << "\tnew undef: " << Idx << '\t' << *MI);
  }
  return true;
}

/// updateRegDefsUses - Replace all defs and uses of SrcReg to DstReg and
/// update the subregister number if it is not zero. If DstReg is a
/// physical register and the existing subregister number of the def / use
/// being updated is not zero, make sure to set it to the correct physical
/// subregister.
void RegisterCoalescer::updateRegDefsUses(unsigned SrcReg,
                                          unsigned DstReg,
                                          unsigned SubIdx) {
  bool DstIsPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);
  LiveInterval *DstInt = DstIsPhys ? 0 : &LIS->getInterval(DstReg);

  // Update LiveDebugVariables.
  LDV->renameRegister(SrcReg, DstReg, SubIdx);

  for (MachineRegisterInfo::reg_iterator I = MRI->reg_begin(SrcReg);
       MachineInstr *UseMI = I.skipInstruction();) {
    SmallVector<unsigned,8> Ops;
    bool Reads, Writes;
    tie(Reads, Writes) = UseMI->readsWritesVirtualRegister(SrcReg, &Ops);

    // If SrcReg wasn't read, it may still be the case that DstReg is live-in
    // because SrcReg is a sub-register.
    if (DstInt && !Reads && SubIdx)
      Reads = DstInt->liveAt(LIS->getInstructionIndex(UseMI));

    // Replace SrcReg with DstReg in all UseMI operands.
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      MachineOperand &MO = UseMI->getOperand(Ops[i]);

      // Adjust <undef> flags in case of sub-register joins. We don't want to
      // turn a full def into a read-modify-write sub-register def and vice
      // versa.
      if (SubIdx && MO.isDef())
        MO.setIsUndef(!Reads);

      if (DstIsPhys)
        MO.substPhysReg(DstReg, *TRI);
      else
        MO.substVirtReg(DstReg, SubIdx, *TRI);
    }

    DEBUG({
        dbgs() << "\t\tupdated: ";
        if (!UseMI->isDebugValue())
          dbgs() << LIS->getInstructionIndex(UseMI) << "\t";
        dbgs() << *UseMI;
      });
  }
}

/// canJoinPhys - Return true if a copy involving a physreg should be joined.
bool RegisterCoalescer::canJoinPhys(CoalescerPair &CP) {
  /// Always join simple intervals that are defined by a single copy from a
  /// reserved register. This doesn't increase register pressure, so it is
  /// always beneficial.
  if (!MRI->isReserved(CP.getDstReg())) {
    DEBUG(dbgs() << "\tCan only merge into reserved registers.\n");
    return false;
  }

  LiveInterval &JoinVInt = LIS->getInterval(CP.getSrcReg());
  if (CP.isFlipped() && JoinVInt.containsOneValue())
    return true;

  DEBUG(dbgs() << "\tCannot join defs into reserved register.\n");
  return false;
}

/// joinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
/// which are the src/dst of the copy instruction CopyMI.  This returns true
/// if the copy was successfully coalesced away. If it is not currently
/// possible to coalesce this interval, but it may be possible if other
/// things get coalesced, then it returns true by reference in 'Again'.
bool RegisterCoalescer::joinCopy(MachineInstr *CopyMI, bool &Again) {

  Again = false;
  DEBUG(dbgs() << LIS->getInstructionIndex(CopyMI) << '\t' << *CopyMI);

  CoalescerPair CP(*TRI);
  if (!CP.setRegisters(CopyMI)) {
    DEBUG(dbgs() << "\tNot coalescable.\n");
    return false;
  }

  // Dead code elimination. This really should be handled by MachineDCE, but
  // sometimes dead copies slip through, and we can't generate invalid live
  // ranges.
  if (!CP.isPhys() && CopyMI->allDefsAreDead()) {
    DEBUG(dbgs() << "\tCopy is dead.\n");
    DeadDefs.push_back(CopyMI);
    eliminateDeadDefs();
    return true;
  }

  // Eliminate undefs.
  if (!CP.isPhys() && eliminateUndefCopy(CopyMI, CP)) {
    DEBUG(dbgs() << "\tEliminated copy of <undef> value.\n");
    LIS->RemoveMachineInstrFromMaps(CopyMI);
    CopyMI->eraseFromParent();
    return false;  // Not coalescable.
  }

  // Coalesced copies are normally removed immediately, but transformations
  // like removeCopyByCommutingDef() can inadvertently create identity copies.
  // When that happens, just join the values and remove the copy.
  if (CP.getSrcReg() == CP.getDstReg()) {
    LiveInterval &LI = LIS->getInterval(CP.getSrcReg());
    DEBUG(dbgs() << "\tCopy already coalesced: " << LI << '\n');
    LiveRangeQuery LRQ(LI, LIS->getInstructionIndex(CopyMI));
    if (VNInfo *DefVNI = LRQ.valueDefined()) {
      VNInfo *ReadVNI = LRQ.valueIn();
      assert(ReadVNI && "No value before copy and no <undef> flag.");
      assert(ReadVNI != DefVNI && "Cannot read and define the same value.");
      LI.MergeValueNumberInto(DefVNI, ReadVNI);
      DEBUG(dbgs() << "\tMerged values:          " << LI << '\n');
    }
    LIS->RemoveMachineInstrFromMaps(CopyMI);
    CopyMI->eraseFromParent();
    return true;
  }

  // Enforce policies.
  if (CP.isPhys()) {
    DEBUG(dbgs() << "\tConsidering merging " << PrintReg(CP.getSrcReg(), TRI)
                 << " with " << PrintReg(CP.getDstReg(), TRI, CP.getSrcIdx())
                 << '\n');
    if (!canJoinPhys(CP)) {
      // Before giving up coalescing, if definition of source is defined by
      // trivial computation, try rematerializing it.
      if (!CP.isFlipped() &&
          reMaterializeTrivialDef(LIS->getInterval(CP.getSrcReg()),
                                  CP.getDstReg(), CopyMI))
        return true;
      return false;
    }
  } else {
    DEBUG({
      dbgs() << "\tConsidering merging to " << CP.getNewRC()->getName()
             << " with ";
      if (CP.getDstIdx() && CP.getSrcIdx())
        dbgs() << PrintReg(CP.getDstReg()) << " in "
               << TRI->getSubRegIndexName(CP.getDstIdx()) << " and "
               << PrintReg(CP.getSrcReg()) << " in "
               << TRI->getSubRegIndexName(CP.getSrcIdx()) << '\n';
      else
        dbgs() << PrintReg(CP.getSrcReg(), TRI) << " in "
               << PrintReg(CP.getDstReg(), TRI, CP.getSrcIdx()) << '\n';
    });

    // When possible, let DstReg be the larger interval.
    if (!CP.isPartial() && LIS->getInterval(CP.getSrcReg()).ranges.size() >
                           LIS->getInterval(CP.getDstReg()).ranges.size())
      CP.flip();
  }

  // Okay, attempt to join these two intervals.  On failure, this returns false.
  // Otherwise, if one of the intervals being joined is a physreg, this method
  // always canonicalizes DstInt to be it.  The output "SrcInt" will not have
  // been modified, so we can use this information below to update aliases.
  if (!joinIntervals(CP)) {
    // Coalescing failed.

    // If definition of source is defined by trivial computation, try
    // rematerializing it.
    if (!CP.isFlipped() &&
        reMaterializeTrivialDef(LIS->getInterval(CP.getSrcReg()),
                                CP.getDstReg(), CopyMI))
      return true;

    // If we can eliminate the copy without merging the live ranges, do so now.
    if (!CP.isPartial() && !CP.isPhys()) {
      if (adjustCopiesBackFrom(CP, CopyMI) ||
          removeCopyByCommutingDef(CP, CopyMI)) {
        LIS->RemoveMachineInstrFromMaps(CopyMI);
        CopyMI->eraseFromParent();
        DEBUG(dbgs() << "\tTrivial!\n");
        return true;
      }
    }

    // Otherwise, we are unable to join the intervals.
    DEBUG(dbgs() << "\tInterference!\n");
    Again = true;  // May be possible to coalesce later.
    return false;
  }

  // Coalescing to a virtual register that is of a sub-register class of the
  // other. Make sure the resulting register is set to the right register class.
  if (CP.isCrossClass()) {
    ++numCrossRCs;
    MRI->setRegClass(CP.getDstReg(), CP.getNewRC());
  }

  // Removing sub-register copies can ease the register class constraints.
  // Make sure we attempt to inflate the register class of DstReg.
  if (!CP.isPhys() && RegClassInfo.isProperSubClass(CP.getNewRC()))
    InflateRegs.push_back(CP.getDstReg());

  // CopyMI has been erased by joinIntervals at this point. Remove it from
  // ErasedInstrs since copyCoalesceWorkList() won't add a successful join back
  // to the work list. This keeps ErasedInstrs from growing needlessly.
  ErasedInstrs.erase(CopyMI);

  // Rewrite all SrcReg operands to DstReg.
  // Also update DstReg operands to include DstIdx if it is set.
  if (CP.getDstIdx())
    updateRegDefsUses(CP.getDstReg(), CP.getDstReg(), CP.getDstIdx());
  updateRegDefsUses(CP.getSrcReg(), CP.getDstReg(), CP.getSrcIdx());

  // SrcReg is guaranteed to be the register whose live interval that is
  // being merged.
  LIS->removeInterval(CP.getSrcReg());

  // Update regalloc hint.
  TRI->UpdateRegAllocHint(CP.getSrcReg(), CP.getDstReg(), *MF);

  DEBUG({
    dbgs() << "\tJoined. Result = " << PrintReg(CP.getDstReg(), TRI);
    if (!CP.isPhys())
      dbgs() << LIS->getInterval(CP.getDstReg());
     dbgs() << '\n';
  });

  ++numJoins;
  return true;
}

/// Attempt joining with a reserved physreg.
bool RegisterCoalescer::joinReservedPhysReg(CoalescerPair &CP) {
  assert(CP.isPhys() && "Must be a physreg copy");
  assert(MRI->isReserved(CP.getDstReg()) && "Not a reserved register");
  LiveInterval &RHS = LIS->getInterval(CP.getSrcReg());
  DEBUG(dbgs() << "\t\tRHS = " << PrintReg(CP.getSrcReg()) << ' ' << RHS
               << '\n');

  assert(CP.isFlipped() && RHS.containsOneValue() &&
         "Invalid join with reserved register");

  // Optimization for reserved registers like ESP. We can only merge with a
  // reserved physreg if RHS has a single value that is a copy of CP.DstReg().
  // The live range of the reserved register will look like a set of dead defs
  // - we don't properly track the live range of reserved registers.

  // Deny any overlapping intervals.  This depends on all the reserved
  // register live ranges to look like dead defs.
  for (MCRegUnitIterator UI(CP.getDstReg(), TRI); UI.isValid(); ++UI)
    if (RHS.overlaps(LIS->getRegUnit(*UI))) {
      DEBUG(dbgs() << "\t\tInterference: " << PrintRegUnit(*UI, TRI) << '\n');
      return false;
    }

  // Skip any value computations, we are not adding new values to the
  // reserved register.  Also skip merging the live ranges, the reserved
  // register live range doesn't need to be accurate as long as all the
  // defs are there.

  // Delete the identity copy.
  MachineInstr *CopyMI = MRI->getVRegDef(RHS.reg);
  LIS->RemoveMachineInstrFromMaps(CopyMI);
  CopyMI->eraseFromParent();

  // We don't track kills for reserved registers.
  MRI->clearKillFlags(CP.getSrcReg());

  return true;
}

//===----------------------------------------------------------------------===//
//                 Interference checking and interval joining
//===----------------------------------------------------------------------===//
//
// In the easiest case, the two live ranges being joined are disjoint, and
// there is no interference to consider. It is quite common, though, to have
// overlapping live ranges, and we need to check if the interference can be
// resolved.
//
// The live range of a single SSA value forms a sub-tree of the dominator tree.
// This means that two SSA values overlap if and only if the def of one value
// is contained in the live range of the other value. As a special case, the
// overlapping values can be defined at the same index.
//
// The interference from an overlapping def can be resolved in these cases:
//
// 1. Coalescable copies. The value is defined by a copy that would become an
//    identity copy after joining SrcReg and DstReg. The copy instruction will
//    be removed, and the value will be merged with the source value.
//
//    There can be several copies back and forth, causing many values to be
//    merged into one. We compute a list of ultimate values in the joined live
//    range as well as a mappings from the old value numbers.
//
// 2. IMPLICIT_DEF. This instruction is only inserted to ensure all PHI
//    predecessors have a live out value. It doesn't cause real interference,
//    and can be merged into the value it overlaps. Like a coalescable copy, it
//    can be erased after joining.
//
// 3. Copy of external value. The overlapping def may be a copy of a value that
//    is already in the other register. This is like a coalescable copy, but
//    the live range of the source register must be trimmed after erasing the
//    copy instruction:
//
//      %src = COPY %ext
//      %dst = COPY %ext  <-- Remove this COPY, trim the live range of %ext.
//
// 4. Clobbering undefined lanes. Vector registers are sometimes built by
//    defining one lane at a time:
//
//      %dst:ssub0<def,read-undef> = FOO
//      %src = BAR
//      %dst:ssub1<def> = COPY %src
//
//    The live range of %src overlaps the %dst value defined by FOO, but
//    merging %src into %dst:ssub1 is only going to clobber the ssub1 lane
//    which was undef anyway.
//
//    The value mapping is more complicated in this case. The final live range
//    will have different value numbers for both FOO and BAR, but there is no
//    simple mapping from old to new values. It may even be necessary to add
//    new PHI values.
//
// 5. Clobbering dead lanes. A def may clobber a lane of a vector register that
//    is live, but never read. This can happen because we don't compute
//    individual live ranges per lane.
//
//      %dst<def> = FOO
//      %src = BAR
//      %dst:ssub1<def> = COPY %src
//
//    This kind of interference is only resolved locally. If the clobbered
//    lane value escapes the block, the join is aborted.

namespace {
/// Track information about values in a single virtual register about to be
/// joined. Objects of this class are always created in pairs - one for each
/// side of the CoalescerPair.
class JoinVals {
  LiveInterval &LI;

  // Location of this register in the final joined register.
  // Either CP.DstIdx or CP.SrcIdx.
  unsigned SubIdx;

  // Values that will be present in the final live range.
  SmallVectorImpl<VNInfo*> &NewVNInfo;

  const CoalescerPair &CP;
  LiveIntervals *LIS;
  SlotIndexes *Indexes;
  const TargetRegisterInfo *TRI;

  // Value number assignments. Maps value numbers in LI to entries in NewVNInfo.
  // This is suitable for passing to LiveInterval::join().
  SmallVector<int, 8> Assignments;

  // Conflict resolution for overlapping values.
  enum ConflictResolution {
    // No overlap, simply keep this value.
    CR_Keep,

    // Merge this value into OtherVNI and erase the defining instruction.
    // Used for IMPLICIT_DEF, coalescable copies, and copies from external
    // values.
    CR_Erase,

    // Merge this value into OtherVNI but keep the defining instruction.
    // This is for the special case where OtherVNI is defined by the same
    // instruction.
    CR_Merge,

    // Keep this value, and have it replace OtherVNI where possible. This
    // complicates value mapping since OtherVNI maps to two different values
    // before and after this def.
    // Used when clobbering undefined or dead lanes.
    CR_Replace,

    // Unresolved conflict. Visit later when all values have been mapped.
    CR_Unresolved,

    // Unresolvable conflict. Abort the join.
    CR_Impossible
  };

  // Per-value info for LI. The lane bit masks are all relative to the final
  // joined register, so they can be compared directly between SrcReg and
  // DstReg.
  struct Val {
    ConflictResolution Resolution;

    // Lanes written by this def, 0 for unanalyzed values.
    unsigned WriteLanes;

    // Lanes with defined values in this register. Other lanes are undef and
    // safe to clobber.
    unsigned ValidLanes;

    // Value in LI being redefined by this def.
    VNInfo *RedefVNI;

    // Value in the other live range that overlaps this def, if any.
    VNInfo *OtherVNI;

    // Is this value an IMPLICIT_DEF?
    bool IsImplicitDef;

    // True when the live range of this value will be pruned because of an
    // overlapping CR_Replace value in the other live range.
    bool Pruned;

    // True once Pruned above has been computed.
    bool PrunedComputed;

    Val() : Resolution(CR_Keep), WriteLanes(0), ValidLanes(0),
            RedefVNI(0), OtherVNI(0), IsImplicitDef(false), Pruned(false),
            PrunedComputed(false) {}

    bool isAnalyzed() const { return WriteLanes != 0; }
  };

  // One entry per value number in LI.
  SmallVector<Val, 8> Vals;

  unsigned computeWriteLanes(const MachineInstr *DefMI, bool &Redef);
  VNInfo *stripCopies(VNInfo *VNI);
  ConflictResolution analyzeValue(unsigned ValNo, JoinVals &Other);
  void computeAssignment(unsigned ValNo, JoinVals &Other);
  bool taintExtent(unsigned, unsigned, JoinVals&,
                   SmallVectorImpl<std::pair<SlotIndex, unsigned> >&);
  bool usesLanes(MachineInstr *MI, unsigned, unsigned, unsigned);
  bool isPrunedValue(unsigned ValNo, JoinVals &Other);

public:
  JoinVals(LiveInterval &li, unsigned subIdx,
           SmallVectorImpl<VNInfo*> &newVNInfo,
           const CoalescerPair &cp,
           LiveIntervals *lis,
           const TargetRegisterInfo *tri)
    : LI(li), SubIdx(subIdx), NewVNInfo(newVNInfo), CP(cp), LIS(lis),
      Indexes(LIS->getSlotIndexes()), TRI(tri),
      Assignments(LI.getNumValNums(), -1), Vals(LI.getNumValNums())
  {}

  /// Analyze defs in LI and compute a value mapping in NewVNInfo.
  /// Returns false if any conflicts were impossible to resolve.
  bool mapValues(JoinVals &Other);

  /// Try to resolve conflicts that require all values to be mapped.
  /// Returns false if any conflicts were impossible to resolve.
  bool resolveConflicts(JoinVals &Other);

  /// Prune the live range of values in Other.LI where they would conflict with
  /// CR_Replace values in LI. Collect end points for restoring the live range
  /// after joining.
  void pruneValues(JoinVals &Other, SmallVectorImpl<SlotIndex> &EndPoints);

  /// Erase any machine instructions that have been coalesced away.
  /// Add erased instructions to ErasedInstrs.
  /// Add foreign virtual registers to ShrinkRegs if their live range ended at
  /// the erased instrs.
  void eraseInstrs(SmallPtrSet<MachineInstr*, 8> &ErasedInstrs,
                   SmallVectorImpl<unsigned> &ShrinkRegs);

  /// Get the value assignments suitable for passing to LiveInterval::join.
  const int *getAssignments() const { return Assignments.data(); }
};
} // end anonymous namespace

/// Compute the bitmask of lanes actually written by DefMI.
/// Set Redef if there are any partial register definitions that depend on the
/// previous value of the register.
unsigned JoinVals::computeWriteLanes(const MachineInstr *DefMI, bool &Redef) {
  unsigned L = 0;
  for (ConstMIOperands MO(DefMI); MO.isValid(); ++MO) {
    if (!MO->isReg() || MO->getReg() != LI.reg || !MO->isDef())
      continue;
    L |= TRI->getSubRegIndexLaneMask(
           TRI->composeSubRegIndices(SubIdx, MO->getSubReg()));
    if (MO->readsReg())
      Redef = true;
  }
  return L;
}

/// Find the ultimate value that VNI was copied from.
VNInfo *JoinVals::stripCopies(VNInfo *VNI) {
  while (!VNI->isPHIDef()) {
    MachineInstr *MI = Indexes->getInstructionFromIndex(VNI->def);
    assert(MI && "No defining instruction");
    if (!MI->isFullCopy())
      break;
    unsigned Reg = MI->getOperand(1).getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      break;
    LiveRangeQuery LRQ(LIS->getInterval(Reg), VNI->def);
    if (!LRQ.valueIn())
      break;
    VNI = LRQ.valueIn();
  }
  return VNI;
}

/// Analyze ValNo in this live range, and set all fields of Vals[ValNo].
/// Return a conflict resolution when possible, but leave the hard cases as
/// CR_Unresolved.
/// Recursively calls computeAssignment() on this and Other, guaranteeing that
/// both OtherVNI and RedefVNI have been analyzed and mapped before returning.
/// The recursion always goes upwards in the dominator tree, making loops
/// impossible.
JoinVals::ConflictResolution
JoinVals::analyzeValue(unsigned ValNo, JoinVals &Other) {
  Val &V = Vals[ValNo];
  assert(!V.isAnalyzed() && "Value has already been analyzed!");
  VNInfo *VNI = LI.getValNumInfo(ValNo);
  if (VNI->isUnused()) {
    V.WriteLanes = ~0u;
    return CR_Keep;
  }

  // Get the instruction defining this value, compute the lanes written.
  const MachineInstr *DefMI = 0;
  if (VNI->isPHIDef()) {
    // Conservatively assume that all lanes in a PHI are valid.
    V.ValidLanes = V.WriteLanes = TRI->getSubRegIndexLaneMask(SubIdx);
  } else {
    DefMI = Indexes->getInstructionFromIndex(VNI->def);
    bool Redef = false;
    V.ValidLanes = V.WriteLanes = computeWriteLanes(DefMI, Redef);

    // If this is a read-modify-write instruction, there may be more valid
    // lanes than the ones written by this instruction.
    // This only covers partial redef operands. DefMI may have normal use
    // operands reading the register. They don't contribute valid lanes.
    //
    // This adds ssub1 to the set of valid lanes in %src:
    //
    //   %src:ssub1<def> = FOO
    //
    // This leaves only ssub1 valid, making any other lanes undef:
    //
    //   %src:ssub1<def,read-undef> = FOO %src:ssub2
    //
    // The <read-undef> flag on the def operand means that old lane values are
    // not important.
    if (Redef) {
      V.RedefVNI = LiveRangeQuery(LI, VNI->def).valueIn();
      assert(V.RedefVNI && "Instruction is reading nonexistent value");
      computeAssignment(V.RedefVNI->id, Other);
      V.ValidLanes |= Vals[V.RedefVNI->id].ValidLanes;
    }

    // An IMPLICIT_DEF writes undef values.
    if (DefMI->isImplicitDef()) {
      V.IsImplicitDef = true;
      V.ValidLanes &= ~V.WriteLanes;
    }
  }

  // Find the value in Other that overlaps VNI->def, if any.
  LiveRangeQuery OtherLRQ(Other.LI, VNI->def);

  // It is possible that both values are defined by the same instruction, or
  // the values are PHIs defined in the same block. When that happens, the two
  // values should be merged into one, but not into any preceding value.
  // The first value defined or visited gets CR_Keep, the other gets CR_Merge.
  if (VNInfo *OtherVNI = OtherLRQ.valueDefined()) {
    assert(SlotIndex::isSameInstr(VNI->def, OtherVNI->def) && "Broken LRQ");

    // One value stays, the other is merged. Keep the earlier one, or the first
    // one we see.
    if (OtherVNI->def < VNI->def)
      Other.computeAssignment(OtherVNI->id, *this);
    else if (VNI->def < OtherVNI->def && OtherLRQ.valueIn()) {
      // This is an early-clobber def overlapping a live-in value in the other
      // register. Not mergeable.
      V.OtherVNI = OtherLRQ.valueIn();
      return CR_Impossible;
    }
    V.OtherVNI = OtherVNI;
    Val &OtherV = Other.Vals[OtherVNI->id];
    // Keep this value, check for conflicts when analyzing OtherVNI.
    if (!OtherV.isAnalyzed())
      return CR_Keep;
    // Both sides have been analyzed now.
    // Allow overlapping PHI values. Any real interference would show up in a
    // predecessor, the PHI itself can't introduce any conflicts.
    if (VNI->isPHIDef())
      return CR_Merge;
    if (V.ValidLanes & OtherV.ValidLanes)
      // Overlapping lanes can't be resolved.
      return CR_Impossible;
    else
      return CR_Merge;
  }

  // No simultaneous def. Is Other live at the def?
  V.OtherVNI = OtherLRQ.valueIn();
  if (!V.OtherVNI)
    // No overlap, no conflict.
    return CR_Keep;

  assert(!SlotIndex::isSameInstr(VNI->def, V.OtherVNI->def) && "Broken LRQ");

  // We have overlapping values, or possibly a kill of Other.
  // Recursively compute assignments up the dominator tree.
  Other.computeAssignment(V.OtherVNI->id, *this);
  const Val &OtherV = Other.Vals[V.OtherVNI->id];

  // Allow overlapping PHI values. Any real interference would show up in a
  // predecessor, the PHI itself can't introduce any conflicts.
  if (VNI->isPHIDef())
    return CR_Replace;

  // Check for simple erasable conflicts.
  if (DefMI->isImplicitDef())
    return CR_Erase;

  // Include the non-conflict where DefMI is a coalescable copy that kills
  // OtherVNI. We still want the copy erased and value numbers merged.
  if (CP.isCoalescable(DefMI)) {
    // Some of the lanes copied from OtherVNI may be undef, making them undef
    // here too.
    V.ValidLanes &= ~V.WriteLanes | OtherV.ValidLanes;
    return CR_Erase;
  }

  // This may not be a real conflict if DefMI simply kills Other and defines
  // VNI.
  if (OtherLRQ.isKill() && OtherLRQ.endPoint() <= VNI->def)
    return CR_Keep;

  // Handle the case where VNI and OtherVNI can be proven to be identical:
  //
  //   %other = COPY %ext
  //   %this  = COPY %ext <-- Erase this copy
  //
  if (DefMI->isFullCopy() && !CP.isPartial() &&
      stripCopies(VNI) == stripCopies(V.OtherVNI))
    return CR_Erase;

  // If the lanes written by this instruction were all undef in OtherVNI, it is
  // still safe to join the live ranges. This can't be done with a simple value
  // mapping, though - OtherVNI will map to multiple values:
  //
  //   1 %dst:ssub0 = FOO                <-- OtherVNI
  //   2 %src = BAR                      <-- VNI
  //   3 %dst:ssub1 = COPY %src<kill>    <-- Eliminate this copy.
  //   4 BAZ %dst<kill>
  //   5 QUUX %src<kill>
  //
  // Here OtherVNI will map to itself in [1;2), but to VNI in [2;5). CR_Replace
  // handles this complex value mapping.
  if ((V.WriteLanes & OtherV.ValidLanes) == 0)
    return CR_Replace;

  // If the other live range is killed by DefMI and the live ranges are still
  // overlapping, it must be because we're looking at an early clobber def:
  //
  //   %dst<def,early-clobber> = ASM %src<kill>
  //
  // In this case, it is illegal to merge the two live ranges since the early
  // clobber def would clobber %src before it was read.
  if (OtherLRQ.isKill()) {
    // This case where the def doesn't overlap the kill is handled above.
    assert(VNI->def.isEarlyClobber() &&
           "Only early clobber defs can overlap a kill");
    return CR_Impossible;
  }

  // VNI is clobbering live lanes in OtherVNI, but there is still the
  // possibility that no instructions actually read the clobbered lanes.
  // If we're clobbering all the lanes in OtherVNI, at least one must be read.
  // Otherwise Other.LI wouldn't be live here.
  if ((TRI->getSubRegIndexLaneMask(Other.SubIdx) & ~V.WriteLanes) == 0)
    return CR_Impossible;

  // We need to verify that no instructions are reading the clobbered lanes. To
  // save compile time, we'll only check that locally. Don't allow the tainted
  // value to escape the basic block.
  MachineBasicBlock *MBB = Indexes->getMBBFromIndex(VNI->def);
  if (OtherLRQ.endPoint() >= Indexes->getMBBEndIdx(MBB))
    return CR_Impossible;

  // There are still some things that could go wrong besides clobbered lanes
  // being read, for example OtherVNI may be only partially redefined in MBB,
  // and some clobbered lanes could escape the block. Save this analysis for
  // resolveConflicts() when all values have been mapped. We need to know
  // RedefVNI and WriteLanes for any later defs in MBB, and we can't compute
  // that now - the recursive analyzeValue() calls must go upwards in the
  // dominator tree.
  return CR_Unresolved;
}

/// Compute the value assignment for ValNo in LI.
/// This may be called recursively by analyzeValue(), but never for a ValNo on
/// the stack.
void JoinVals::computeAssignment(unsigned ValNo, JoinVals &Other) {
  Val &V = Vals[ValNo];
  if (V.isAnalyzed()) {
    // Recursion should always move up the dominator tree, so ValNo is not
    // supposed to reappear before it has been assigned.
    assert(Assignments[ValNo] != -1 && "Bad recursion?");
    return;
  }
  switch ((V.Resolution = analyzeValue(ValNo, Other))) {
  case CR_Erase:
  case CR_Merge:
    // Merge this ValNo into OtherVNI.
    assert(V.OtherVNI && "OtherVNI not assigned, can't merge.");
    assert(Other.Vals[V.OtherVNI->id].isAnalyzed() && "Missing recursion");
    Assignments[ValNo] = Other.Assignments[V.OtherVNI->id];
    DEBUG(dbgs() << "\t\tmerge " << PrintReg(LI.reg) << ':' << ValNo << '@'
                 << LI.getValNumInfo(ValNo)->def << " into "
                 << PrintReg(Other.LI.reg) << ':' << V.OtherVNI->id << '@'
                 << V.OtherVNI->def << " --> @"
                 << NewVNInfo[Assignments[ValNo]]->def << '\n');
    break;
  case CR_Replace:
  case CR_Unresolved:
    // The other value is going to be pruned if this join is successful.
    assert(V.OtherVNI && "OtherVNI not assigned, can't prune");
    Other.Vals[V.OtherVNI->id].Pruned = true;
    // Fall through.
  default:
    // This value number needs to go in the final joined live range.
    Assignments[ValNo] = NewVNInfo.size();
    NewVNInfo.push_back(LI.getValNumInfo(ValNo));
    break;
  }
}

bool JoinVals::mapValues(JoinVals &Other) {
  for (unsigned i = 0, e = LI.getNumValNums(); i != e; ++i) {
    computeAssignment(i, Other);
    if (Vals[i].Resolution == CR_Impossible) {
      DEBUG(dbgs() << "\t\tinterference at " << PrintReg(LI.reg) << ':' << i
                   << '@' << LI.getValNumInfo(i)->def << '\n');
      return false;
    }
  }
  return true;
}

/// Assuming ValNo is going to clobber some valid lanes in Other.LI, compute
/// the extent of the tainted lanes in the block.
///
/// Multiple values in Other.LI can be affected since partial redefinitions can
/// preserve previously tainted lanes.
///
///   1 %dst = VLOAD           <-- Define all lanes in %dst
///   2 %src = FOO             <-- ValNo to be joined with %dst:ssub0
///   3 %dst:ssub1 = BAR       <-- Partial redef doesn't clear taint in ssub0
///   4 %dst:ssub0 = COPY %src <-- Conflict resolved, ssub0 wasn't read
///
/// For each ValNo in Other that is affected, add an (EndIndex, TaintedLanes)
/// entry to TaintedVals.
///
/// Returns false if the tainted lanes extend beyond the basic block.
bool JoinVals::
taintExtent(unsigned ValNo, unsigned TaintedLanes, JoinVals &Other,
            SmallVectorImpl<std::pair<SlotIndex, unsigned> > &TaintExtent) {
  VNInfo *VNI = LI.getValNumInfo(ValNo);
  MachineBasicBlock *MBB = Indexes->getMBBFromIndex(VNI->def);
  SlotIndex MBBEnd = Indexes->getMBBEndIdx(MBB);

  // Scan Other.LI from VNI.def to MBBEnd.
  LiveInterval::iterator OtherI = Other.LI.find(VNI->def);
  assert(OtherI != Other.LI.end() && "No conflict?");
  do {
    // OtherI is pointing to a tainted value. Abort the join if the tainted
    // lanes escape the block.
    SlotIndex End = OtherI->end;
    if (End >= MBBEnd) {
      DEBUG(dbgs() << "\t\ttaints global " << PrintReg(Other.LI.reg) << ':'
                   << OtherI->valno->id << '@' << OtherI->start << '\n');
      return false;
    }
    DEBUG(dbgs() << "\t\ttaints local " << PrintReg(Other.LI.reg) << ':'
                 << OtherI->valno->id << '@' << OtherI->start
                 << " to " << End << '\n');
    // A dead def is not a problem.
    if (End.isDead())
      break;
    TaintExtent.push_back(std::make_pair(End, TaintedLanes));

    // Check for another def in the MBB.
    if (++OtherI == Other.LI.end() || OtherI->start >= MBBEnd)
      break;

    // Lanes written by the new def are no longer tainted.
    const Val &OV = Other.Vals[OtherI->valno->id];
    TaintedLanes &= ~OV.WriteLanes;
    if (!OV.RedefVNI)
      break;
  } while (TaintedLanes);
  return true;
}

/// Return true if MI uses any of the given Lanes from Reg.
/// This does not include partial redefinitions of Reg.
bool JoinVals::usesLanes(MachineInstr *MI, unsigned Reg, unsigned SubIdx,
                         unsigned Lanes) {
  if (MI->isDebugValue())
    return false;
  for (ConstMIOperands MO(MI); MO.isValid(); ++MO) {
    if (!MO->isReg() || MO->isDef() || MO->getReg() != Reg)
      continue;
    if (!MO->readsReg())
      continue;
    if (Lanes & TRI->getSubRegIndexLaneMask(
                  TRI->composeSubRegIndices(SubIdx, MO->getSubReg())))
      return true;
  }
  return false;
}

bool JoinVals::resolveConflicts(JoinVals &Other) {
  for (unsigned i = 0, e = LI.getNumValNums(); i != e; ++i) {
    Val &V = Vals[i];
    assert (V.Resolution != CR_Impossible && "Unresolvable conflict");
    if (V.Resolution != CR_Unresolved)
      continue;
    DEBUG(dbgs() << "\t\tconflict at " << PrintReg(LI.reg) << ':' << i
                 << '@' << LI.getValNumInfo(i)->def << '\n');
    ++NumLaneConflicts;
    assert(V.OtherVNI && "Inconsistent conflict resolution.");
    VNInfo *VNI = LI.getValNumInfo(i);
    const Val &OtherV = Other.Vals[V.OtherVNI->id];

    // VNI is known to clobber some lanes in OtherVNI. If we go ahead with the
    // join, those lanes will be tainted with a wrong value. Get the extent of
    // the tainted lanes.
    unsigned TaintedLanes = V.WriteLanes & OtherV.ValidLanes;
    SmallVector<std::pair<SlotIndex, unsigned>, 8> TaintExtent;
    if (!taintExtent(i, TaintedLanes, Other, TaintExtent))
      // Tainted lanes would extend beyond the basic block.
      return false;

    assert(!TaintExtent.empty() && "There should be at least one conflict.");

    // Now look at the instructions from VNI->def to TaintExtent (inclusive).
    MachineBasicBlock *MBB = Indexes->getMBBFromIndex(VNI->def);
    MachineBasicBlock::iterator MI = MBB->begin();
    if (!VNI->isPHIDef()) {
      MI = Indexes->getInstructionFromIndex(VNI->def);
      // No need to check the instruction defining VNI for reads.
      ++MI;
    }
    assert(!SlotIndex::isSameInstr(VNI->def, TaintExtent.front().first) &&
           "Interference ends on VNI->def. Should have been handled earlier");
    MachineInstr *LastMI =
      Indexes->getInstructionFromIndex(TaintExtent.front().first);
    assert(LastMI && "Range must end at a proper instruction");
    unsigned TaintNum = 0;
    for(;;) {
      assert(MI != MBB->end() && "Bad LastMI");
      if (usesLanes(MI, Other.LI.reg, Other.SubIdx, TaintedLanes)) {
        DEBUG(dbgs() << "\t\ttainted lanes used by: " << *MI);
        return false;
      }
      // LastMI is the last instruction to use the current value.
      if (&*MI == LastMI) {
        if (++TaintNum == TaintExtent.size())
          break;
        LastMI = Indexes->getInstructionFromIndex(TaintExtent[TaintNum].first);
        assert(LastMI && "Range must end at a proper instruction");
        TaintedLanes = TaintExtent[TaintNum].second;
      }
      ++MI;
    }

    // The tainted lanes are unused.
    V.Resolution = CR_Replace;
    ++NumLaneResolves;
  }
  return true;
}

// Determine if ValNo is a copy of a value number in LI or Other.LI that will
// be pruned:
//
//   %dst = COPY %src
//   %src = COPY %dst  <-- This value to be pruned.
//   %dst = COPY %src  <-- This value is a copy of a pruned value.
//
bool JoinVals::isPrunedValue(unsigned ValNo, JoinVals &Other) {
  Val &V = Vals[ValNo];
  if (V.Pruned || V.PrunedComputed)
    return V.Pruned;

  if (V.Resolution != CR_Erase && V.Resolution != CR_Merge)
    return V.Pruned;

  // Follow copies up the dominator tree and check if any intermediate value
  // has been pruned.
  V.PrunedComputed = true;
  V.Pruned = Other.isPrunedValue(V.OtherVNI->id, *this);
  return V.Pruned;
}

void JoinVals::pruneValues(JoinVals &Other,
                           SmallVectorImpl<SlotIndex> &EndPoints) {
  for (unsigned i = 0, e = LI.getNumValNums(); i != e; ++i) {
    SlotIndex Def = LI.getValNumInfo(i)->def;
    switch (Vals[i].Resolution) {
    case CR_Keep:
      break;
    case CR_Replace: {
      // This value takes precedence over the value in Other.LI.
      LIS->pruneValue(&Other.LI, Def, &EndPoints);
      // Check if we're replacing an IMPLICIT_DEF value. The IMPLICIT_DEF
      // instructions are only inserted to provide a live-out value for PHI
      // predecessors, so the instruction should simply go away once its value
      // has been replaced.
      Val &OtherV = Other.Vals[Vals[i].OtherVNI->id];
      bool EraseImpDef = OtherV.IsImplicitDef && OtherV.Resolution == CR_Keep;
      if (!Def.isBlock()) {
        // Remove <def,read-undef> flags. This def is now a partial redef.
        // Also remove <def,dead> flags since the joined live range will
        // continue past this instruction.
        for (MIOperands MO(Indexes->getInstructionFromIndex(Def));
             MO.isValid(); ++MO)
          if (MO->isReg() && MO->isDef() && MO->getReg() == LI.reg) {
            MO->setIsUndef(EraseImpDef);
            MO->setIsDead(false);
          }
        // This value will reach instructions below, but we need to make sure
        // the live range also reaches the instruction at Def.
        if (!EraseImpDef)
          EndPoints.push_back(Def);
      }
      DEBUG(dbgs() << "\t\tpruned " << PrintReg(Other.LI.reg) << " at " << Def
                   << ": " << Other.LI << '\n');
      break;
    }
    case CR_Erase:
    case CR_Merge:
      if (isPrunedValue(i, Other)) {
        // This value is ultimately a copy of a pruned value in LI or Other.LI.
        // We can no longer trust the value mapping computed by
        // computeAssignment(), the value that was originally copied could have
        // been replaced.
        LIS->pruneValue(&LI, Def, &EndPoints);
        DEBUG(dbgs() << "\t\tpruned all of " << PrintReg(LI.reg) << " at "
                     << Def << ": " << LI << '\n');
      }
      break;
    case CR_Unresolved:
    case CR_Impossible:
      llvm_unreachable("Unresolved conflicts");
    }
  }
}

void JoinVals::eraseInstrs(SmallPtrSet<MachineInstr*, 8> &ErasedInstrs,
                           SmallVectorImpl<unsigned> &ShrinkRegs) {
  for (unsigned i = 0, e = LI.getNumValNums(); i != e; ++i) {
    // Get the def location before markUnused() below invalidates it.
    SlotIndex Def = LI.getValNumInfo(i)->def;
    switch (Vals[i].Resolution) {
    case CR_Keep:
      // If an IMPLICIT_DEF value is pruned, it doesn't serve a purpose any
      // longer. The IMPLICIT_DEF instructions are only inserted by
      // PHIElimination to guarantee that all PHI predecessors have a value.
      if (!Vals[i].IsImplicitDef || !Vals[i].Pruned)
        break;
      // Remove value number i from LI. Note that this VNInfo is still present
      // in NewVNInfo, so it will appear as an unused value number in the final
      // joined interval.
      LI.getValNumInfo(i)->markUnused();
      LI.removeValNo(LI.getValNumInfo(i));
      DEBUG(dbgs() << "\t\tremoved " << i << '@' << Def << ": " << LI << '\n');
      // FALL THROUGH.

    case CR_Erase: {
      MachineInstr *MI = Indexes->getInstructionFromIndex(Def);
      assert(MI && "No instruction to erase");
      if (MI->isCopy()) {
        unsigned Reg = MI->getOperand(1).getReg();
        if (TargetRegisterInfo::isVirtualRegister(Reg) &&
            Reg != CP.getSrcReg() && Reg != CP.getDstReg())
          ShrinkRegs.push_back(Reg);
      }
      ErasedInstrs.insert(MI);
      DEBUG(dbgs() << "\t\terased:\t" << Def << '\t' << *MI);
      LIS->RemoveMachineInstrFromMaps(MI);
      MI->eraseFromParent();
      break;
    }
    default:
      break;
    }
  }
}

bool RegisterCoalescer::joinVirtRegs(CoalescerPair &CP) {
  SmallVector<VNInfo*, 16> NewVNInfo;
  LiveInterval &RHS = LIS->getInterval(CP.getSrcReg());
  LiveInterval &LHS = LIS->getInterval(CP.getDstReg());
  JoinVals RHSVals(RHS, CP.getSrcIdx(), NewVNInfo, CP, LIS, TRI);
  JoinVals LHSVals(LHS, CP.getDstIdx(), NewVNInfo, CP, LIS, TRI);

  DEBUG(dbgs() << "\t\tRHS = " << PrintReg(CP.getSrcReg()) << ' ' << RHS
               << "\n\t\tLHS = " << PrintReg(CP.getDstReg()) << ' ' << LHS
               << '\n');

  // First compute NewVNInfo and the simple value mappings.
  // Detect impossible conflicts early.
  if (!LHSVals.mapValues(RHSVals) || !RHSVals.mapValues(LHSVals))
    return false;

  // Some conflicts can only be resolved after all values have been mapped.
  if (!LHSVals.resolveConflicts(RHSVals) || !RHSVals.resolveConflicts(LHSVals))
    return false;

  // All clear, the live ranges can be merged.

  // The merging algorithm in LiveInterval::join() can't handle conflicting
  // value mappings, so we need to remove any live ranges that overlap a
  // CR_Replace resolution. Collect a set of end points that can be used to
  // restore the live range after joining.
  SmallVector<SlotIndex, 8> EndPoints;
  LHSVals.pruneValues(RHSVals, EndPoints);
  RHSVals.pruneValues(LHSVals, EndPoints);

  // Erase COPY and IMPLICIT_DEF instructions. This may cause some external
  // registers to require trimming.
  SmallVector<unsigned, 8> ShrinkRegs;
  LHSVals.eraseInstrs(ErasedInstrs, ShrinkRegs);
  RHSVals.eraseInstrs(ErasedInstrs, ShrinkRegs);
  while (!ShrinkRegs.empty())
    LIS->shrinkToUses(&LIS->getInterval(ShrinkRegs.pop_back_val()));

  // Join RHS into LHS.
  LHS.join(RHS, LHSVals.getAssignments(), RHSVals.getAssignments(), NewVNInfo,
           MRI);

  // Kill flags are going to be wrong if the live ranges were overlapping.
  // Eventually, we should simply clear all kill flags when computing live
  // ranges. They are reinserted after register allocation.
  MRI->clearKillFlags(LHS.reg);
  MRI->clearKillFlags(RHS.reg);

  if (EndPoints.empty())
    return true;

  // Recompute the parts of the live range we had to remove because of
  // CR_Replace conflicts.
  DEBUG(dbgs() << "\t\trestoring liveness to " << EndPoints.size()
               << " points: " << LHS << '\n');
  LIS->extendToIndices(&LHS, EndPoints);
  return true;
}

/// joinIntervals - Attempt to join these two intervals.  On failure, this
/// returns false.
bool RegisterCoalescer::joinIntervals(CoalescerPair &CP) {
  return CP.isPhys() ? joinReservedPhysReg(CP) : joinVirtRegs(CP);
}

namespace {
  // DepthMBBCompare - Comparison predicate that sort first based on the loop
  // depth of the basic block (the unsigned), and then on the MBB number.
  struct DepthMBBCompare {
    typedef std::pair<unsigned, MachineBasicBlock*> DepthMBBPair;
    bool operator()(const DepthMBBPair &LHS, const DepthMBBPair &RHS) const {
      // Deeper loops first
      if (LHS.first != RHS.first)
        return LHS.first > RHS.first;

      // Prefer blocks that are more connected in the CFG. This takes care of
      // the most difficult copies first while intervals are short.
      unsigned cl = LHS.second->pred_size() + LHS.second->succ_size();
      unsigned cr = RHS.second->pred_size() + RHS.second->succ_size();
      if (cl != cr)
        return cl > cr;

      // As a last resort, sort by block number.
      return LHS.second->getNumber() < RHS.second->getNumber();
    }
  };
}

// Try joining WorkList copies starting from index From.
// Null out any successful joins.
bool RegisterCoalescer::copyCoalesceWorkList(unsigned From) {
  assert(From <= WorkList.size() && "Out of range");
  bool Progress = false;
  for (unsigned i = From, e = WorkList.size(); i != e; ++i) {
    if (!WorkList[i])
      continue;
    // Skip instruction pointers that have already been erased, for example by
    // dead code elimination.
    if (ErasedInstrs.erase(WorkList[i])) {
      WorkList[i] = 0;
      continue;
    }
    bool Again = false;
    bool Success = joinCopy(WorkList[i], Again);
    Progress |= Success;
    if (Success || !Again)
      WorkList[i] = 0;
  }
  return Progress;
}

void
RegisterCoalescer::copyCoalesceInMBB(MachineBasicBlock *MBB) {
  DEBUG(dbgs() << MBB->getName() << ":\n");

  // Collect all copy-like instructions in MBB. Don't start coalescing anything
  // yet, it might invalidate the iterator.
  const unsigned PrevSize = WorkList.size();
  for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
       MII != E; ++MII)
    if (MII->isCopyLike())
      WorkList.push_back(MII);

  // Try coalescing the collected copies immediately, and remove the nulls.
  // This prevents the WorkList from getting too large since most copies are
  // joinable on the first attempt.
  if (copyCoalesceWorkList(PrevSize))
    WorkList.erase(std::remove(WorkList.begin() + PrevSize, WorkList.end(),
                               (MachineInstr*)0), WorkList.end());
}

void RegisterCoalescer::joinAllIntervals() {
  DEBUG(dbgs() << "********** JOINING INTERVALS ***********\n");
  assert(WorkList.empty() && "Old data still around.");

  if (Loops->empty()) {
    // If there are no loops in the function, join intervals in function order.
    for (MachineFunction::iterator I = MF->begin(), E = MF->end();
         I != E; ++I)
      copyCoalesceInMBB(I);
  } else {
    // Otherwise, join intervals in inner loops before other intervals.
    // Unfortunately we can't just iterate over loop hierarchy here because
    // there may be more MBB's than BB's.  Collect MBB's for sorting.

    // Join intervals in the function prolog first. We want to join physical
    // registers with virtual registers before the intervals got too long.
    std::vector<std::pair<unsigned, MachineBasicBlock*> > MBBs;
    for (MachineFunction::iterator I = MF->begin(), E = MF->end();I != E;++I){
      MachineBasicBlock *MBB = I;
      MBBs.push_back(std::make_pair(Loops->getLoopDepth(MBB), I));
    }

    // Sort by loop depth.
    std::sort(MBBs.begin(), MBBs.end(), DepthMBBCompare());

    // Finally, join intervals in loop nest order.
    for (unsigned i = 0, e = MBBs.size(); i != e; ++i)
      copyCoalesceInMBB(MBBs[i].second);
  }

  // Joining intervals can allow other intervals to be joined.  Iteratively join
  // until we make no progress.
  while (copyCoalesceWorkList())
    /* empty */ ;
}

void RegisterCoalescer::releaseMemory() {
  ErasedInstrs.clear();
  WorkList.clear();
  DeadDefs.clear();
  InflateRegs.clear();
}

bool RegisterCoalescer::runOnMachineFunction(MachineFunction &fn) {
  MF = &fn;
  MRI = &fn.getRegInfo();
  TM = &fn.getTarget();
  TRI = TM->getRegisterInfo();
  TII = TM->getInstrInfo();
  LIS = &getAnalysis<LiveIntervals>();
  LDV = &getAnalysis<LiveDebugVariables>();
  AA = &getAnalysis<AliasAnalysis>();
  Loops = &getAnalysis<MachineLoopInfo>();

  DEBUG(dbgs() << "********** SIMPLE REGISTER COALESCING **********\n"
               << "********** Function: " << MF->getName() << '\n');

  if (VerifyCoalescing)
    MF->verify(this, "Before register coalescing");

  RegClassInfo.runOnMachineFunction(fn);

  // Join (coalesce) intervals if requested.
  if (EnableJoining)
    joinAllIntervals();

  // After deleting a lot of copies, register classes may be less constrained.
  // Removing sub-register operands may allow GR32_ABCD -> GR32 and DPR_VFP2 ->
  // DPR inflation.
  array_pod_sort(InflateRegs.begin(), InflateRegs.end());
  InflateRegs.erase(std::unique(InflateRegs.begin(), InflateRegs.end()),
                    InflateRegs.end());
  DEBUG(dbgs() << "Trying to inflate " << InflateRegs.size() << " regs.\n");
  for (unsigned i = 0, e = InflateRegs.size(); i != e; ++i) {
    unsigned Reg = InflateRegs[i];
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    if (MRI->recomputeRegClass(Reg, *TM)) {
      DEBUG(dbgs() << PrintReg(Reg) << " inflated to "
                   << MRI->getRegClass(Reg)->getName() << '\n');
      ++NumInflated;
    }
  }

  DEBUG(dump());
  DEBUG(LDV->dump());
  if (VerifyCoalescing)
    MF->verify(this, "After register coalescing");
  return true;
}

/// print - Implement the dump method.
void RegisterCoalescer::print(raw_ostream &O, const Module* m) const {
   LIS->print(O, m);
}
