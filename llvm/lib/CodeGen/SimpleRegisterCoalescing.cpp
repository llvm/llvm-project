//===-- SimpleRegisterCoalescing.cpp - Register Coalescing ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple register coalescing pass that attempts to
// aggressively coalesce every register copy that it can.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regcoalescing"
#include "SimpleRegisterCoalescing.h"
#include "VirtRegMap.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/Value.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

STATISTIC(numJoins    , "Number of interval joins performed");
STATISTIC(numCrossRCs , "Number of cross class joins performed");
STATISTIC(numCommutes , "Number of instruction commuting performed");
STATISTIC(numExtends  , "Number of copies extended");
STATISTIC(NumReMats   , "Number of instructions re-materialized");
STATISTIC(numPeep     , "Number of identity moves eliminated after coalescing");
STATISTIC(numAborts   , "Number of times interval joining aborted");
STATISTIC(numDeadValNo, "Number of valno def marked dead");

char SimpleRegisterCoalescing::ID = 0;
static cl::opt<bool>
EnableJoining("join-liveintervals",
              cl::desc("Coalesce copies (default=true)"),
              cl::init(true));

static cl::opt<bool>
NewHeuristic("new-coalescer-heuristic",
             cl::desc("Use new coalescer heuristic"),
             cl::init(false), cl::Hidden);

static cl::opt<bool>
DisableCrossClassJoin("disable-cross-class-join",
               cl::desc("Avoid coalescing cross register class copies"),
               cl::init(false), cl::Hidden);

static cl::opt<bool>
PhysJoinTweak("tweak-phys-join-heuristics",
               cl::desc("Tweak heuristics for joining phys reg with vr"),
               cl::init(false), cl::Hidden);

static RegisterPass<SimpleRegisterCoalescing> 
X("simple-register-coalescing", "Simple Register Coalescing");

// Declare that we implement the RegisterCoalescer interface
static RegisterAnalysisGroup<RegisterCoalescer, true/*The Default*/> V(X);

const PassInfo *const llvm::SimpleRegisterCoalescingID = &X;

void SimpleRegisterCoalescing::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addPreservedID(MachineDominatorsID);
  if (StrongPHIElim)
    AU.addPreservedID(StrongPHIEliminationID);
  else
    AU.addPreservedID(PHIEliminationID);
  AU.addPreservedID(TwoAddressInstructionPassID);
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// AdjustCopiesBackFrom - We found a non-trivially-coalescable copy with IntA
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
bool SimpleRegisterCoalescing::AdjustCopiesBackFrom(LiveInterval &IntA,
                                                    LiveInterval &IntB,
                                                    MachineInstr *CopyMI) {
  unsigned CopyIdx = li_->getDefIndex(li_->getInstructionIndex(CopyMI));

  // BValNo is a value number in B that is defined by a copy from A.  'B3' in
  // the example above.
  LiveInterval::iterator BLR = IntB.FindLiveRangeContaining(CopyIdx);
  assert(BLR != IntB.end() && "Live range not found!");
  VNInfo *BValNo = BLR->valno;
  
  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we 
  // can't process it.
  if (!BValNo->getCopy()) return false;
  assert(BValNo->def == CopyIdx && "Copy doesn't define the value?");
  
  // AValNo is the value number in A that defines the copy, A3 in the example.
  unsigned CopyUseIdx = li_->getUseIndex(CopyIdx);
  LiveInterval::iterator ALR = IntA.FindLiveRangeContaining(CopyUseIdx);
  assert(ALR != IntA.end() && "Live range not found!");
  VNInfo *AValNo = ALR->valno;
  // If it's re-defined by an early clobber somewhere in the live range, then
  // it's not safe to eliminate the copy. FIXME: This is a temporary workaround.
  // See PR3149:
  // 172     %ECX<def> = MOV32rr %reg1039<kill>
  // 180     INLINEASM <es:subl $5,$1
  //         sbbl $3,$0>, 10, %EAX<def>, 14, %ECX<earlyclobber,def>, 9, %EAX<kill>,
  // 36, <fi#0>, 1, %reg0, 0, 9, %ECX<kill>, 36, <fi#1>, 1, %reg0, 0
  // 188     %EAX<def> = MOV32rr %EAX<kill>
  // 196     %ECX<def> = MOV32rr %ECX<kill>
  // 204     %ECX<def> = MOV32rr %ECX<kill>
  // 212     %EAX<def> = MOV32rr %EAX<kill>
  // 220     %EAX<def> = MOV32rr %EAX
  // 228     %reg1039<def> = MOV32rr %ECX<kill>
  // The early clobber operand ties ECX input to the ECX def.
  //
  // The live interval of ECX is represented as this:
  // %reg20,inf = [46,47:1)[174,230:0)  0@174-(230) 1@46-(47)
  // The coalescer has no idea there was a def in the middle of [174,230].
  if (AValNo->hasRedefByEC())
    return false;
  
  // If AValNo is defined as a copy from IntB, we can potentially process this.  
  // Get the instruction that defines this value number.
  unsigned SrcReg = li_->getVNInfoSourceReg(AValNo);
  if (!SrcReg) return false;  // Not defined by a copy.
    
  // If the value number is not defined by a copy instruction, ignore it.

  // If the source register comes from an interval other than IntB, we can't
  // handle this.
  if (SrcReg != IntB.reg) return false;
  
  // Get the LiveRange in IntB that this value number starts with.
  LiveInterval::iterator ValLR = IntB.FindLiveRangeContaining(AValNo->def-1);
  assert(ValLR != IntB.end() && "Live range not found!");
  
  // Make sure that the end of the live range is inside the same block as
  // CopyMI.
  MachineInstr *ValLREndInst = li_->getInstructionFromIndex(ValLR->end-1);
  if (!ValLREndInst || 
      ValLREndInst->getParent() != CopyMI->getParent()) return false;

  // Okay, we now know that ValLR ends in the same block that the CopyMI
  // live-range starts.  If there are no intervening live ranges between them in
  // IntB, we can merge them.
  if (ValLR+1 != BLR) return false;

  // If a live interval is a physical register, conservatively check if any
  // of its sub-registers is overlapping the live interval of the virtual
  // register. If so, do not coalesce.
  if (TargetRegisterInfo::isPhysicalRegister(IntB.reg) &&
      *tri_->getSubRegisters(IntB.reg)) {
    for (const unsigned* SR = tri_->getSubRegisters(IntB.reg); *SR; ++SR)
      if (li_->hasInterval(*SR) && IntA.overlaps(li_->getInterval(*SR))) {
        DOUT << "Interfere with sub-register ";
        DEBUG(li_->getInterval(*SR).print(DOUT, tri_));
        return false;
      }
  }
  
  DOUT << "\nExtending: "; IntB.print(DOUT, tri_);
  
  unsigned FillerStart = ValLR->end, FillerEnd = BLR->start;
  // We are about to delete CopyMI, so need to remove it as the 'instruction
  // that defines this value #'. Update the the valnum with the new defining
  // instruction #.
  BValNo->def  = FillerStart;
  BValNo->setCopy(0);
  
  // Okay, we can merge them.  We need to insert a new liverange:
  // [ValLR.end, BLR.begin) of either value number, then we merge the
  // two value numbers.
  IntB.addRange(LiveRange(FillerStart, FillerEnd, BValNo));

  // If the IntB live range is assigned to a physical register, and if that
  // physreg has sub-registers, update their live intervals as well. 
  if (TargetRegisterInfo::isPhysicalRegister(IntB.reg)) {
    for (const unsigned *SR = tri_->getSubRegisters(IntB.reg); *SR; ++SR) {
      LiveInterval &SRLI = li_->getInterval(*SR);
      SRLI.addRange(LiveRange(FillerStart, FillerEnd,
                              SRLI.getNextValue(FillerStart, 0, true,
                                                li_->getVNInfoAllocator())));
    }
  }

  // Okay, merge "B1" into the same value number as "B0".
  if (BValNo != ValLR->valno) {
    IntB.addKills(ValLR->valno, BValNo->kills);
    IntB.MergeValueNumberInto(BValNo, ValLR->valno);
  }
  DOUT << "   result = "; IntB.print(DOUT, tri_);
  DOUT << "\n";

  // If the source instruction was killing the source register before the
  // merge, unset the isKill marker given the live range has been extended.
  int UIdx = ValLREndInst->findRegisterUseOperandIdx(IntB.reg, true);
  if (UIdx != -1) {
    ValLREndInst->getOperand(UIdx).setIsKill(false);
    IntB.removeKill(ValLR->valno, FillerStart);
  }

  // If the copy instruction was killing the destination register before the
  // merge, find the last use and trim the live range. That will also add the
  // isKill marker.
  if (CopyMI->killsRegister(IntA.reg))
    TrimLiveIntervalToLastUse(CopyUseIdx, CopyMI->getParent(), IntA, ALR);

  ++numExtends;
  return true;
}

/// HasOtherReachingDefs - Return true if there are definitions of IntB
/// other than BValNo val# that can reach uses of AValno val# of IntA.
bool SimpleRegisterCoalescing::HasOtherReachingDefs(LiveInterval &IntA,
                                                    LiveInterval &IntB,
                                                    VNInfo *AValNo,
                                                    VNInfo *BValNo) {
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

/// RemoveCopyByCommutingDef - We found a non-trivially-coalescable copy with IntA
/// being the source and IntB being the dest, thus this defines a value number
/// in IntB.  If the source value number (in IntA) is defined by a commutable
/// instruction and its other operand is coalesced to the copy dest register,
/// see if we can transform the copy into a noop by commuting the definition. For
/// example,
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
bool SimpleRegisterCoalescing::RemoveCopyByCommutingDef(LiveInterval &IntA,
                                                        LiveInterval &IntB,
                                                        MachineInstr *CopyMI) {
  unsigned CopyIdx = li_->getDefIndex(li_->getInstructionIndex(CopyMI));

  // FIXME: For now, only eliminate the copy by commuting its def when the
  // source register is a virtual register. We want to guard against cases
  // where the copy is a back edge copy and commuting the def lengthen the
  // live interval of the source register to the entire loop.
  if (TargetRegisterInfo::isPhysicalRegister(IntA.reg))
    return false;

  // BValNo is a value number in B that is defined by a copy from A. 'B3' in
  // the example above.
  LiveInterval::iterator BLR = IntB.FindLiveRangeContaining(CopyIdx);
  assert(BLR != IntB.end() && "Live range not found!");
  VNInfo *BValNo = BLR->valno;
  
  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we 
  // can't process it.
  if (!BValNo->getCopy()) return false;
  assert(BValNo->def == CopyIdx && "Copy doesn't define the value?");
  
  // AValNo is the value number in A that defines the copy, A3 in the example.
  LiveInterval::iterator ALR = IntA.FindLiveRangeContaining(CopyIdx-1);
  assert(ALR != IntA.end() && "Live range not found!");
  VNInfo *AValNo = ALR->valno;
  // If other defs can reach uses of this def, then it's not safe to perform
  // the optimization. FIXME: Do isPHIDef and isDefAccurate both need to be
  // tested?
  if (AValNo->isPHIDef() || !AValNo->isDefAccurate() ||
      AValNo->isUnused() || AValNo->hasPHIKill())
    return false;
  MachineInstr *DefMI = li_->getInstructionFromIndex(AValNo->def);
  const TargetInstrDesc &TID = DefMI->getDesc();
  if (!TID.isCommutable())
    return false;
  // If DefMI is a two-address instruction then commuting it will change the
  // destination register.
  int DefIdx = DefMI->findRegisterDefOperandIdx(IntA.reg);
  assert(DefIdx != -1);
  unsigned UseOpIdx;
  if (!DefMI->isRegTiedToUseOperand(DefIdx, &UseOpIdx))
    return false;
  unsigned Op1, Op2, NewDstIdx;
  if (!tii_->findCommutedOpIndices(DefMI, Op1, Op2))
    return false;
  if (Op1 == UseOpIdx)
    NewDstIdx = Op2;
  else if (Op2 == UseOpIdx)
    NewDstIdx = Op1;
  else
    return false;

  MachineOperand &NewDstMO = DefMI->getOperand(NewDstIdx);
  unsigned NewReg = NewDstMO.getReg();
  if (NewReg != IntB.reg || !NewDstMO.isKill())
    return false;

  // Make sure there are no other definitions of IntB that would reach the
  // uses which the new definition can reach.
  if (HasOtherReachingDefs(IntA, IntB, AValNo, BValNo))
    return false;

  // If some of the uses of IntA.reg is already coalesced away, return false.
  // It's not possible to determine whether it's safe to perform the coalescing.
  for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(IntA.reg),
         UE = mri_->use_end(); UI != UE; ++UI) {
    MachineInstr *UseMI = &*UI;
    unsigned UseIdx = li_->getInstructionIndex(UseMI);
    LiveInterval::iterator ULR = IntA.FindLiveRangeContaining(UseIdx);
    if (ULR == IntA.end())
      continue;
    if (ULR->valno == AValNo && JoinedCopies.count(UseMI))
      return false;
  }

  // At this point we have decided that it is legal to do this
  // transformation.  Start by commuting the instruction.
  MachineBasicBlock *MBB = DefMI->getParent();
  MachineInstr *NewMI = tii_->commuteInstruction(DefMI);
  if (!NewMI)
    return false;
  if (NewMI != DefMI) {
    li_->ReplaceMachineInstrInMaps(DefMI, NewMI);
    MBB->insert(DefMI, NewMI);
    MBB->erase(DefMI);
  }
  unsigned OpIdx = NewMI->findRegisterUseOperandIdx(IntA.reg, false);
  NewMI->getOperand(OpIdx).setIsKill();

  bool BHasPHIKill = BValNo->hasPHIKill();
  SmallVector<VNInfo*, 4> BDeadValNos;
  VNInfo::KillSet BKills;
  std::map<unsigned, unsigned> BExtend;

  // If ALR and BLR overlaps and end of BLR extends beyond end of ALR, e.g.
  // A = or A, B
  // ...
  // B = A
  // ...
  // C = A<kill>
  // ...
  //   = B
  //
  // then do not add kills of A to the newly created B interval.
  bool Extended = BLR->end > ALR->end && ALR->end != ALR->start;
  if (Extended)
    BExtend[ALR->end] = BLR->end;

  // Update uses of IntA of the specific Val# with IntB.
  bool BHasSubRegs = false;
  if (TargetRegisterInfo::isPhysicalRegister(IntB.reg))
    BHasSubRegs = *tri_->getSubRegisters(IntB.reg);
  for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(IntA.reg),
         UE = mri_->use_end(); UI != UE;) {
    MachineOperand &UseMO = UI.getOperand();
    MachineInstr *UseMI = &*UI;
    ++UI;
    if (JoinedCopies.count(UseMI))
      continue;
    unsigned UseIdx = li_->getInstructionIndex(UseMI);
    LiveInterval::iterator ULR = IntA.FindLiveRangeContaining(UseIdx);
    if (ULR == IntA.end() || ULR->valno != AValNo)
      continue;
    UseMO.setReg(NewReg);
    if (UseMI == CopyMI)
      continue;
    if (UseMO.isKill()) {
      if (Extended)
        UseMO.setIsKill(false);
      else
        BKills.push_back(VNInfo::KillInfo(false, li_->getUseIndex(UseIdx)+1));
    }
    unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
    if (!tii_->isMoveInstr(*UseMI, SrcReg, DstReg, SrcSubIdx, DstSubIdx))
      continue;
    if (DstReg == IntB.reg) {
      // This copy will become a noop. If it's defining a new val#,
      // remove that val# as well. However this live range is being
      // extended to the end of the existing live range defined by the copy.
      unsigned DefIdx = li_->getDefIndex(UseIdx);
      const LiveRange *DLR = IntB.getLiveRangeContaining(DefIdx);
      BHasPHIKill |= DLR->valno->hasPHIKill();
      assert(DLR->valno->def == DefIdx);
      BDeadValNos.push_back(DLR->valno);
      BExtend[DLR->start] = DLR->end;
      JoinedCopies.insert(UseMI);
      // If this is a kill but it's going to be removed, the last use
      // of the same val# is the new kill.
      if (UseMO.isKill())
        BKills.pop_back();
    }
  }

  // We need to insert a new liverange: [ALR.start, LastUse). It may be we can
  // simply extend BLR if CopyMI doesn't end the range.
  DOUT << "\nExtending: "; IntB.print(DOUT, tri_);

  // Remove val#'s defined by copies that will be coalesced away.
  for (unsigned i = 0, e = BDeadValNos.size(); i != e; ++i) {
    VNInfo *DeadVNI = BDeadValNos[i];
    if (BHasSubRegs) {
      for (const unsigned *SR = tri_->getSubRegisters(IntB.reg); *SR; ++SR) {
        LiveInterval &SRLI = li_->getInterval(*SR);
        const LiveRange *SRLR = SRLI.getLiveRangeContaining(DeadVNI->def);
        SRLI.removeValNo(SRLR->valno);
      }
    }
    IntB.removeValNo(BDeadValNos[i]);
  }

  // Extend BValNo by merging in IntA live ranges of AValNo. Val# definition
  // is updated. Kills are also updated.
  VNInfo *ValNo = BValNo;
  ValNo->def = AValNo->def;
  ValNo->setCopy(0);
  for (unsigned j = 0, ee = ValNo->kills.size(); j != ee; ++j) {
    unsigned Kill = ValNo->kills[j].killIdx;
    if (Kill != BLR->end)
      BKills.push_back(VNInfo::KillInfo(ValNo->kills[j].isPHIKill, Kill));
  }
  ValNo->kills.clear();
  for (LiveInterval::iterator AI = IntA.begin(), AE = IntA.end();
       AI != AE; ++AI) {
    if (AI->valno != AValNo) continue;
    unsigned End = AI->end;
    std::map<unsigned, unsigned>::iterator EI = BExtend.find(End);
    if (EI != BExtend.end())
      End = EI->second;
    IntB.addRange(LiveRange(AI->start, End, ValNo));

    // If the IntB live range is assigned to a physical register, and if that
    // physreg has sub-registers, update their live intervals as well. 
    if (BHasSubRegs) {
      for (const unsigned *SR = tri_->getSubRegisters(IntB.reg); *SR; ++SR) {
        LiveInterval &SRLI = li_->getInterval(*SR);
        SRLI.MergeInClobberRange(AI->start, End, li_->getVNInfoAllocator());
      }
    }
  }
  IntB.addKills(ValNo, BKills);
  ValNo->setHasPHIKill(BHasPHIKill);

  DOUT << "   result = "; IntB.print(DOUT, tri_);
  DOUT << "\n";

  DOUT << "\nShortening: "; IntA.print(DOUT, tri_);
  IntA.removeValNo(AValNo);
  DOUT << "   result = "; IntA.print(DOUT, tri_);
  DOUT << "\n";

  ++numCommutes;
  return true;
}

/// isSameOrFallThroughBB - Return true if MBB == SuccMBB or MBB simply
/// fallthoughs to SuccMBB.
static bool isSameOrFallThroughBB(MachineBasicBlock *MBB,
                                  MachineBasicBlock *SuccMBB,
                                  const TargetInstrInfo *tii_) {
  if (MBB == SuccMBB)
    return true;
  MachineBasicBlock *TBB = 0, *FBB = 0;
  SmallVector<MachineOperand, 4> Cond;
  return !tii_->AnalyzeBranch(*MBB, TBB, FBB, Cond) && !TBB && !FBB &&
    MBB->isSuccessor(SuccMBB);
}

/// removeRange - Wrapper for LiveInterval::removeRange. This removes a range
/// from a physical register live interval as well as from the live intervals
/// of its sub-registers.
static void removeRange(LiveInterval &li, unsigned Start, unsigned End,
                        LiveIntervals *li_, const TargetRegisterInfo *tri_) {
  li.removeRange(Start, End, true);
  if (TargetRegisterInfo::isPhysicalRegister(li.reg)) {
    for (const unsigned* SR = tri_->getSubRegisters(li.reg); *SR; ++SR) {
      if (!li_->hasInterval(*SR))
        continue;
      LiveInterval &sli = li_->getInterval(*SR);
      unsigned RemoveStart = Start;
      unsigned RemoveEnd = Start;
      while (RemoveEnd != End) {
        LiveInterval::iterator LR = sli.FindLiveRangeContaining(RemoveStart);
        if (LR == sli.end())
          break;
        RemoveEnd = (LR->end < End) ? LR->end : End;
        sli.removeRange(RemoveStart, RemoveEnd, true);
        RemoveStart = RemoveEnd;
      }
    }
  }
}

/// TrimLiveIntervalToLastUse - If there is a last use in the same basic block
/// as the copy instruction, trim the live interval to the last use and return
/// true.
bool
SimpleRegisterCoalescing::TrimLiveIntervalToLastUse(unsigned CopyIdx,
                                                    MachineBasicBlock *CopyMBB,
                                                    LiveInterval &li,
                                                    const LiveRange *LR) {
  unsigned MBBStart = li_->getMBBStartIdx(CopyMBB);
  unsigned LastUseIdx;
  MachineOperand *LastUse = lastRegisterUse(LR->start, CopyIdx-1, li.reg,
                                            LastUseIdx);
  if (LastUse) {
    MachineInstr *LastUseMI = LastUse->getParent();
    if (!isSameOrFallThroughBB(LastUseMI->getParent(), CopyMBB, tii_)) {
      // r1024 = op
      // ...
      // BB1:
      //       = r1024
      //
      // BB2:
      // r1025<dead> = r1024<kill>
      if (MBBStart < LR->end)
        removeRange(li, MBBStart, LR->end, li_, tri_);
      return true;
    }

    // There are uses before the copy, just shorten the live range to the end
    // of last use.
    LastUse->setIsKill();
    removeRange(li, li_->getDefIndex(LastUseIdx), LR->end, li_, tri_);
    li.addKill(LR->valno, LastUseIdx+1, false);
    unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
    if (tii_->isMoveInstr(*LastUseMI, SrcReg, DstReg, SrcSubIdx, DstSubIdx) &&
        DstReg == li.reg) {
      // Last use is itself an identity code.
      int DeadIdx = LastUseMI->findRegisterDefOperandIdx(li.reg, false, tri_);
      LastUseMI->getOperand(DeadIdx).setIsDead();
    }
    return true;
  }

  // Is it livein?
  if (LR->start <= MBBStart && LR->end > MBBStart) {
    if (LR->start == 0) {
      assert(TargetRegisterInfo::isPhysicalRegister(li.reg));
      // Live-in to the function but dead. Remove it from entry live-in set.
      mf_->begin()->removeLiveIn(li.reg);
    }
    // FIXME: Shorten intervals in BBs that reaches this BB.
  }

  return false;
}

/// ReMaterializeTrivialDef - If the source of a copy is defined by a trivial
/// computation, replace the copy by rematerialize the definition.
bool SimpleRegisterCoalescing::ReMaterializeTrivialDef(LiveInterval &SrcInt,
                                                       unsigned DstReg,
                                                       unsigned DstSubIdx,
                                                       MachineInstr *CopyMI) {
  unsigned CopyIdx = li_->getUseIndex(li_->getInstructionIndex(CopyMI));
  LiveInterval::iterator SrcLR = SrcInt.FindLiveRangeContaining(CopyIdx);
  assert(SrcLR != SrcInt.end() && "Live range not found!");
  VNInfo *ValNo = SrcLR->valno;
  // If other defs can reach uses of this def, then it's not safe to perform
  // the optimization. FIXME: Do isPHIDef and isDefAccurate both need to be
  // tested?
  if (ValNo->isPHIDef() || !ValNo->isDefAccurate() ||
      ValNo->isUnused() || ValNo->hasPHIKill())
    return false;
  MachineInstr *DefMI = li_->getInstructionFromIndex(ValNo->def);
  const TargetInstrDesc &TID = DefMI->getDesc();
  if (!TID.isAsCheapAsAMove())
    return false;
  if (!DefMI->getDesc().isRematerializable() ||
      !tii_->isTriviallyReMaterializable(DefMI))
    return false;
  bool SawStore = false;
  if (!DefMI->isSafeToMove(tii_, SawStore))
    return false;
  if (TID.getNumDefs() != 1)
    return false;
  if (DefMI->getOpcode() != TargetInstrInfo::IMPLICIT_DEF) {
    // Make sure the copy destination register class fits the instruction
    // definition register class. The mismatch can happen as a result of earlier
    // extract_subreg, insert_subreg, subreg_to_reg coalescing.
    const TargetRegisterClass *RC = TID.OpInfo[0].getRegClass(tri_);
    if (TargetRegisterInfo::isVirtualRegister(DstReg)) {
      if (mri_->getRegClass(DstReg) != RC)
        return false;
    } else if (!RC->contains(DstReg))
      return false;
  }

  // If destination register has a sub-register index on it, make sure it mtches
  // the instruction register class.
  if (DstSubIdx) {
    const TargetInstrDesc &TID = DefMI->getDesc();
    if (TID.getNumDefs() != 1)
      return false;
    const TargetRegisterClass *DstRC = mri_->getRegClass(DstReg);
    const TargetRegisterClass *DstSubRC =
      DstRC->getSubRegisterRegClass(DstSubIdx);
    const TargetRegisterClass *DefRC = TID.OpInfo[0].getRegClass(tri_);
    if (DefRC == DstRC)
      DstSubIdx = 0;
    else if (DefRC != DstSubRC)
      return false;
  }

  unsigned DefIdx = li_->getDefIndex(CopyIdx);
  const LiveRange *DLR= li_->getInterval(DstReg).getLiveRangeContaining(DefIdx);
  DLR->valno->setCopy(0);
  // Don't forget to update sub-register intervals.
  if (TargetRegisterInfo::isPhysicalRegister(DstReg)) {
    for (const unsigned* SR = tri_->getSubRegisters(DstReg); *SR; ++SR) {
      if (!li_->hasInterval(*SR))
        continue;
      DLR = li_->getInterval(*SR).getLiveRangeContaining(DefIdx);
      if (DLR && DLR->valno->getCopy() == CopyMI)
        DLR->valno->setCopy(0);
    }
  }

  // If copy kills the source register, find the last use and propagate
  // kill.
  bool checkForDeadDef = false;
  MachineBasicBlock *MBB = CopyMI->getParent();
  if (CopyMI->killsRegister(SrcInt.reg))
    if (!TrimLiveIntervalToLastUse(CopyIdx, MBB, SrcInt, SrcLR)) {
      checkForDeadDef = true;
    }

  MachineBasicBlock::iterator MII = next(MachineBasicBlock::iterator(CopyMI));
  tii_->reMaterialize(*MBB, MII, DstReg, DstSubIdx, DefMI);
  MachineInstr *NewMI = prior(MII);

  if (checkForDeadDef) {
    // PR4090 fix: Trim interval failed because there was no use of the
    // source interval in this MBB. If the def is in this MBB too then we
    // should mark it dead:
    if (DefMI->getParent() == MBB) {
      DefMI->addRegisterDead(SrcInt.reg, tri_);
      SrcLR->end = SrcLR->start + 1;
    }
  }

  // CopyMI may have implicit operands, transfer them over to the newly
  // rematerialized instruction. And update implicit def interval valnos.
  for (unsigned i = CopyMI->getDesc().getNumOperands(),
         e = CopyMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = CopyMI->getOperand(i);
    if (MO.isReg() && MO.isImplicit())
      NewMI->addOperand(MO);
    if (MO.isDef() && li_->hasInterval(MO.getReg())) {
      unsigned Reg = MO.getReg();
      DLR = li_->getInterval(Reg).getLiveRangeContaining(DefIdx);
      if (DLR && DLR->valno->getCopy() == CopyMI)
        DLR->valno->setCopy(0);
    }
  }

  li_->ReplaceMachineInstrInMaps(CopyMI, NewMI);
  CopyMI->eraseFromParent();
  ReMatCopies.insert(CopyMI);
  ReMatDefs.insert(DefMI);
  ++NumReMats;
  return true;
}

/// isBackEdgeCopy - Returns true if CopyMI is a back edge copy.
///
bool SimpleRegisterCoalescing::isBackEdgeCopy(MachineInstr *CopyMI,
                                              unsigned DstReg) const {
  MachineBasicBlock *MBB = CopyMI->getParent();
  const MachineLoop *L = loopInfo->getLoopFor(MBB);
  if (!L)
    return false;
  if (MBB != L->getLoopLatch())
    return false;

  LiveInterval &LI = li_->getInterval(DstReg);
  unsigned DefIdx = li_->getInstructionIndex(CopyMI);
  LiveInterval::const_iterator DstLR =
    LI.FindLiveRangeContaining(li_->getDefIndex(DefIdx));
  if (DstLR == LI.end())
    return false;
  if (DstLR->valno->kills.size() == 1 && DstLR->valno->kills[0].isPHIKill)
    return true;
  return false;
}

/// UpdateRegDefsUses - Replace all defs and uses of SrcReg to DstReg and
/// update the subregister number if it is not zero. If DstReg is a
/// physical register and the existing subregister number of the def / use
/// being updated is not zero, make sure to set it to the correct physical
/// subregister.
void
SimpleRegisterCoalescing::UpdateRegDefsUses(unsigned SrcReg, unsigned DstReg,
                                            unsigned SubIdx) {
  bool DstIsPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);
  if (DstIsPhys && SubIdx) {
    // Figure out the real physical register we are updating with.
    DstReg = tri_->getSubReg(DstReg, SubIdx);
    SubIdx = 0;
  }

  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(SrcReg),
         E = mri_->reg_end(); I != E; ) {
    MachineOperand &O = I.getOperand();
    MachineInstr *UseMI = &*I;
    ++I;
    unsigned OldSubIdx = O.getSubReg();
    if (DstIsPhys) {
      unsigned UseDstReg = DstReg;
      if (OldSubIdx)
          UseDstReg = tri_->getSubReg(DstReg, OldSubIdx);

      unsigned CopySrcReg, CopyDstReg, CopySrcSubIdx, CopyDstSubIdx;
      if (tii_->isMoveInstr(*UseMI, CopySrcReg, CopyDstReg,
                            CopySrcSubIdx, CopyDstSubIdx) &&
          CopySrcReg != CopyDstReg &&
          CopySrcReg == SrcReg && CopyDstReg != UseDstReg) {
        // If the use is a copy and it won't be coalesced away, and its source
        // is defined by a trivial computation, try to rematerialize it instead.
        if (ReMaterializeTrivialDef(li_->getInterval(SrcReg), CopyDstReg,
                                    CopyDstSubIdx, UseMI))
          continue;
      }

      O.setReg(UseDstReg);
      O.setSubReg(0);
      continue;
    }

    // Sub-register indexes goes from small to large. e.g.
    // RAX: 1 -> AL, 2 -> AX, 3 -> EAX
    // EAX: 1 -> AL, 2 -> AX
    // So RAX's sub-register 2 is AX, RAX's sub-regsiter 3 is EAX, whose
    // sub-register 2 is also AX.
    if (SubIdx && OldSubIdx && SubIdx != OldSubIdx)
      assert(OldSubIdx < SubIdx && "Conflicting sub-register index!");
    else if (SubIdx)
      O.setSubReg(SubIdx);
    // Remove would-be duplicated kill marker.
    if (O.isKill() && UseMI->killsRegister(DstReg))
      O.setIsKill(false);
    O.setReg(DstReg);

    // After updating the operand, check if the machine instruction has
    // become a copy. If so, update its val# information.
    if (JoinedCopies.count(UseMI))
      continue;

    const TargetInstrDesc &TID = UseMI->getDesc();
    unsigned CopySrcReg, CopyDstReg, CopySrcSubIdx, CopyDstSubIdx;
    if (TID.getNumDefs() == 1 && TID.getNumOperands() > 2 &&
        tii_->isMoveInstr(*UseMI, CopySrcReg, CopyDstReg,
                          CopySrcSubIdx, CopyDstSubIdx) &&
        CopySrcReg != CopyDstReg &&
        (TargetRegisterInfo::isVirtualRegister(CopyDstReg) ||
         allocatableRegs_[CopyDstReg])) {
      LiveInterval &LI = li_->getInterval(CopyDstReg);
      unsigned DefIdx = li_->getDefIndex(li_->getInstructionIndex(UseMI));
      if (const LiveRange *DLR = LI.getLiveRangeContaining(DefIdx)) {
        if (DLR->valno->def == DefIdx)
          DLR->valno->setCopy(UseMI);
      }
    }
  }
}

/// RemoveUnnecessaryKills - Remove kill markers that are no longer accurate
/// due to live range lengthening as the result of coalescing.
void SimpleRegisterCoalescing::RemoveUnnecessaryKills(unsigned Reg,
                                                      LiveInterval &LI) {
  for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(Reg),
         UE = mri_->use_end(); UI != UE; ++UI) {
    MachineOperand &UseMO = UI.getOperand();
    if (!UseMO.isKill())
      continue;
    MachineInstr *UseMI = UseMO.getParent();
    unsigned UseIdx = li_->getUseIndex(li_->getInstructionIndex(UseMI));
    const LiveRange *LR = LI.getLiveRangeContaining(UseIdx);
    if (!LR || !LI.isKill(LR->valno, UseIdx+1)) {
      if (LR->valno->def != UseIdx+1) {
        // Interesting problem. After coalescing reg1027's def and kill are both
        // at the same point:  %reg1027,0.000000e+00 = [56,814:0)  0@70-(814)
        //
        // bb5:
        // 60	%reg1027<def> = t2MOVr %reg1027, 14, %reg0, %reg0
        // 68	%reg1027<def> = t2LDRi12 %reg1027<kill>, 8, 14, %reg0
        // 76	t2CMPzri %reg1038<kill,undef>, 0, 14, %reg0, %CPSR<imp-def>
        // 84	%reg1027<def> = t2MOVr %reg1027, 14, %reg0, %reg0
        // 96	t2Bcc mbb<bb5,0x2030910>, 1, %CPSR<kill>
        //
        // Do not remove the kill marker on t2LDRi12.
        UseMO.setIsKill(false);
      }
    }
  }
}

/// removeIntervalIfEmpty - Check if the live interval of a physical register
/// is empty, if so remove it and also remove the empty intervals of its
/// sub-registers. Return true if live interval is removed.
static bool removeIntervalIfEmpty(LiveInterval &li, LiveIntervals *li_,
                                  const TargetRegisterInfo *tri_) {
  if (li.empty()) {
    if (TargetRegisterInfo::isPhysicalRegister(li.reg))
      for (const unsigned* SR = tri_->getSubRegisters(li.reg); *SR; ++SR) {
        if (!li_->hasInterval(*SR))
          continue;
        LiveInterval &sli = li_->getInterval(*SR);
        if (sli.empty())
          li_->removeInterval(*SR);
      }
    li_->removeInterval(li.reg);
    return true;
  }
  return false;
}

/// ShortenDeadCopyLiveRange - Shorten a live range defined by a dead copy.
/// Return true if live interval is removed.
bool SimpleRegisterCoalescing::ShortenDeadCopyLiveRange(LiveInterval &li,
                                                        MachineInstr *CopyMI) {
  unsigned CopyIdx = li_->getInstructionIndex(CopyMI);
  LiveInterval::iterator MLR =
    li.FindLiveRangeContaining(li_->getDefIndex(CopyIdx));
  if (MLR == li.end())
    return false;  // Already removed by ShortenDeadCopySrcLiveRange.
  unsigned RemoveStart = MLR->start;
  unsigned RemoveEnd = MLR->end;
  unsigned DefIdx = li_->getDefIndex(CopyIdx);
  // Remove the liverange that's defined by this.
  if (RemoveStart == DefIdx && RemoveEnd == DefIdx+1) {
    removeRange(li, RemoveStart, RemoveEnd, li_, tri_);
    return removeIntervalIfEmpty(li, li_, tri_);
  }
  return false;
}

/// RemoveDeadDef - If a def of a live interval is now determined dead, remove
/// the val# it defines. If the live interval becomes empty, remove it as well.
bool SimpleRegisterCoalescing::RemoveDeadDef(LiveInterval &li,
                                             MachineInstr *DefMI) {
  unsigned DefIdx = li_->getDefIndex(li_->getInstructionIndex(DefMI));
  LiveInterval::iterator MLR = li.FindLiveRangeContaining(DefIdx);
  if (DefIdx != MLR->valno->def)
    return false;
  li.removeValNo(MLR->valno);
  return removeIntervalIfEmpty(li, li_, tri_);
}

/// PropagateDeadness - Propagate the dead marker to the instruction which
/// defines the val#.
static void PropagateDeadness(LiveInterval &li, MachineInstr *CopyMI,
                              unsigned &LRStart, LiveIntervals *li_,
                              const TargetRegisterInfo* tri_) {
  MachineInstr *DefMI =
    li_->getInstructionFromIndex(li_->getDefIndex(LRStart));
  if (DefMI && DefMI != CopyMI) {
    int DeadIdx = DefMI->findRegisterDefOperandIdx(li.reg, false);
    if (DeadIdx != -1)
      DefMI->getOperand(DeadIdx).setIsDead();
    else
      DefMI->addOperand(MachineOperand::CreateReg(li.reg,
                                                  true, true, false, true));
    ++LRStart;
  }
}

/// ShortenDeadCopySrcLiveRange - Shorten a live range as it's artificially
/// extended by a dead copy. Mark the last use (if any) of the val# as kill as
/// ends the live range there. If there isn't another use, then this live range
/// is dead. Return true if live interval is removed.
bool
SimpleRegisterCoalescing::ShortenDeadCopySrcLiveRange(LiveInterval &li,
                                                      MachineInstr *CopyMI) {
  unsigned CopyIdx = li_->getInstructionIndex(CopyMI);
  if (CopyIdx == 0) {
    // FIXME: special case: function live in. It can be a general case if the
    // first instruction index starts at > 0 value.
    assert(TargetRegisterInfo::isPhysicalRegister(li.reg));
    // Live-in to the function but dead. Remove it from entry live-in set.
    if (mf_->begin()->isLiveIn(li.reg))
      mf_->begin()->removeLiveIn(li.reg);
    const LiveRange *LR = li.getLiveRangeContaining(CopyIdx);
    removeRange(li, LR->start, LR->end, li_, tri_);
    return removeIntervalIfEmpty(li, li_, tri_);
  }

  LiveInterval::iterator LR = li.FindLiveRangeContaining(CopyIdx-1);
  if (LR == li.end())
    // Livein but defined by a phi.
    return false;

  unsigned RemoveStart = LR->start;
  unsigned RemoveEnd = li_->getDefIndex(CopyIdx)+1;
  if (LR->end > RemoveEnd)
    // More uses past this copy? Nothing to do.
    return false;

  // If there is a last use in the same bb, we can't remove the live range.
  // Shorten the live interval and return.
  MachineBasicBlock *CopyMBB = CopyMI->getParent();
  if (TrimLiveIntervalToLastUse(CopyIdx, CopyMBB, li, LR))
    return false;

  // There are other kills of the val#. Nothing to do.
  if (!li.isOnlyLROfValNo(LR))
    return false;

  MachineBasicBlock *StartMBB = li_->getMBBFromIndex(RemoveStart);
  if (!isSameOrFallThroughBB(StartMBB, CopyMBB, tii_))
    // If the live range starts in another mbb and the copy mbb is not a fall
    // through mbb, then we can only cut the range from the beginning of the
    // copy mbb.
    RemoveStart = li_->getMBBStartIdx(CopyMBB) + 1;

  if (LR->valno->def == RemoveStart) {
    // If the def MI defines the val# and this copy is the only kill of the
    // val#, then propagate the dead marker.
    PropagateDeadness(li, CopyMI, RemoveStart, li_, tri_);
    ++numDeadValNo;

    if (li.isKill(LR->valno, RemoveEnd))
      li.removeKill(LR->valno, RemoveEnd);
  }

  removeRange(li, RemoveStart, RemoveEnd, li_, tri_);
  return removeIntervalIfEmpty(li, li_, tri_);
}

/// CanCoalesceWithImpDef - Returns true if the specified copy instruction
/// from an implicit def to another register can be coalesced away.
bool SimpleRegisterCoalescing::CanCoalesceWithImpDef(MachineInstr *CopyMI,
                                                     LiveInterval &li,
                                                     LiveInterval &ImpLi) const{
  if (!CopyMI->killsRegister(ImpLi.reg))
    return false;
  // Make sure this is the only use.
  for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(ImpLi.reg),
         UE = mri_->use_end(); UI != UE;) {
    MachineInstr *UseMI = &*UI;
    ++UI;
    if (CopyMI == UseMI || JoinedCopies.count(UseMI))
      continue;
    return false;
  }
  return true;
}


/// isWinToJoinVRWithSrcPhysReg - Return true if it's worth while to join a
/// a virtual destination register with physical source register.
bool
SimpleRegisterCoalescing::isWinToJoinVRWithSrcPhysReg(MachineInstr *CopyMI,
                                                     MachineBasicBlock *CopyMBB,
                                                     LiveInterval &DstInt,
                                                     LiveInterval &SrcInt) {
  // If the virtual register live interval is long but it has low use desity,
  // do not join them, instead mark the physical register as its allocation
  // preference.
  const TargetRegisterClass *RC = mri_->getRegClass(DstInt.reg);
  unsigned Threshold = allocatableRCRegs_[RC].count() * 2;
  unsigned Length = li_->getApproximateInstructionCount(DstInt);
  if (Length > Threshold &&
      (((float)std::distance(mri_->use_begin(DstInt.reg),
                             mri_->use_end()) / Length) < (1.0 / Threshold)))
    return false;

  // If the virtual register live interval extends into a loop, turn down
  // aggressiveness.
  unsigned CopyIdx = li_->getDefIndex(li_->getInstructionIndex(CopyMI));
  const MachineLoop *L = loopInfo->getLoopFor(CopyMBB);
  if (!L) {
    // Let's see if the virtual register live interval extends into the loop.
    LiveInterval::iterator DLR = DstInt.FindLiveRangeContaining(CopyIdx);
    assert(DLR != DstInt.end() && "Live range not found!");
    DLR = DstInt.FindLiveRangeContaining(DLR->end+1);
    if (DLR != DstInt.end()) {
      CopyMBB = li_->getMBBFromIndex(DLR->start);
      L = loopInfo->getLoopFor(CopyMBB);
    }
  }

  if (!L || Length <= Threshold)
    return true;

  unsigned UseIdx = li_->getUseIndex(CopyIdx);
  LiveInterval::iterator SLR = SrcInt.FindLiveRangeContaining(UseIdx);
  MachineBasicBlock *SMBB = li_->getMBBFromIndex(SLR->start);
  if (loopInfo->getLoopFor(SMBB) != L) {
    if (!loopInfo->isLoopHeader(CopyMBB))
      return false;
    // If vr's live interval extends pass the loop header, do not join.
    for (MachineBasicBlock::succ_iterator SI = CopyMBB->succ_begin(),
           SE = CopyMBB->succ_end(); SI != SE; ++SI) {
      MachineBasicBlock *SuccMBB = *SI;
      if (SuccMBB == CopyMBB)
        continue;
      if (DstInt.overlaps(li_->getMBBStartIdx(SuccMBB),
                          li_->getMBBEndIdx(SuccMBB)+1))
        return false;
    }
  }
  return true;
}

/// isWinToJoinVRWithDstPhysReg - Return true if it's worth while to join a
/// copy from a virtual source register to a physical destination register.
bool
SimpleRegisterCoalescing::isWinToJoinVRWithDstPhysReg(MachineInstr *CopyMI,
                                                     MachineBasicBlock *CopyMBB,
                                                     LiveInterval &DstInt,
                                                     LiveInterval &SrcInt) {
  // If the virtual register live interval is long but it has low use desity,
  // do not join them, instead mark the physical register as its allocation
  // preference.
  const TargetRegisterClass *RC = mri_->getRegClass(SrcInt.reg);
  unsigned Threshold = allocatableRCRegs_[RC].count() * 2;
  unsigned Length = li_->getApproximateInstructionCount(SrcInt);
  if (Length > Threshold &&
      (((float)std::distance(mri_->use_begin(SrcInt.reg),
                             mri_->use_end()) / Length) < (1.0 / Threshold)))
    return false;

  if (SrcInt.empty())
    // Must be implicit_def.
    return false;

  // If the virtual register live interval is defined or cross a loop, turn
  // down aggressiveness.
  unsigned CopyIdx = li_->getDefIndex(li_->getInstructionIndex(CopyMI));
  unsigned UseIdx = li_->getUseIndex(CopyIdx);
  LiveInterval::iterator SLR = SrcInt.FindLiveRangeContaining(UseIdx);
  assert(SLR != SrcInt.end() && "Live range not found!");
  SLR = SrcInt.FindLiveRangeContaining(SLR->start-1);
  if (SLR == SrcInt.end())
    return true;
  MachineBasicBlock *SMBB = li_->getMBBFromIndex(SLR->start);
  const MachineLoop *L = loopInfo->getLoopFor(SMBB);

  if (!L || Length <= Threshold)
    return true;

  if (loopInfo->getLoopFor(CopyMBB) != L) {
    if (SMBB != L->getLoopLatch())
      return false;
    // If vr's live interval is extended from before the loop latch, do not
    // join.
    for (MachineBasicBlock::pred_iterator PI = SMBB->pred_begin(),
           PE = SMBB->pred_end(); PI != PE; ++PI) {
      MachineBasicBlock *PredMBB = *PI;
      if (PredMBB == SMBB)
        continue;
      if (SrcInt.overlaps(li_->getMBBStartIdx(PredMBB),
                          li_->getMBBEndIdx(PredMBB)+1))
        return false;
    }
  }
  return true;
}

/// isWinToJoinCrossClass - Return true if it's profitable to coalesce
/// two virtual registers from different register classes.
bool
SimpleRegisterCoalescing::isWinToJoinCrossClass(unsigned LargeReg,
                                                unsigned SmallReg,
                                                unsigned Threshold) {
  // Then make sure the intervals are *short*.
  LiveInterval &LargeInt = li_->getInterval(LargeReg);
  LiveInterval &SmallInt = li_->getInterval(SmallReg);
  unsigned LargeSize = li_->getApproximateInstructionCount(LargeInt);
  unsigned SmallSize = li_->getApproximateInstructionCount(SmallInt);
  if (SmallSize > Threshold || LargeSize > Threshold)
    if ((float)std::distance(mri_->use_begin(SmallReg),
                             mri_->use_end()) / SmallSize <
        (float)std::distance(mri_->use_begin(LargeReg),
                             mri_->use_end()) / LargeSize)
      return false;
  return true;
}

/// HasIncompatibleSubRegDefUse - If we are trying to coalesce a virtual
/// register with a physical register, check if any of the virtual register
/// operand is a sub-register use or def. If so, make sure it won't result
/// in an illegal extract_subreg or insert_subreg instruction. e.g.
/// vr1024 = extract_subreg vr1025, 1
/// ...
/// vr1024 = mov8rr AH
/// If vr1024 is coalesced with AH, the extract_subreg is now illegal since
/// AH does not have a super-reg whose sub-register 1 is AH.
bool
SimpleRegisterCoalescing::HasIncompatibleSubRegDefUse(MachineInstr *CopyMI,
                                                      unsigned VirtReg,
                                                      unsigned PhysReg) {
  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(VirtReg),
         E = mri_->reg_end(); I != E; ++I) {
    MachineOperand &O = I.getOperand();
    MachineInstr *MI = &*I;
    if (MI == CopyMI || JoinedCopies.count(MI))
      continue;
    unsigned SubIdx = O.getSubReg();
    if (SubIdx && !tri_->getSubReg(PhysReg, SubIdx))
      return true;
    if (MI->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG) {
      SubIdx = MI->getOperand(2).getImm();
      if (O.isUse() && !tri_->getSubReg(PhysReg, SubIdx))
        return true;
      if (O.isDef()) {
        unsigned SrcReg = MI->getOperand(1).getReg();
        const TargetRegisterClass *RC =
          TargetRegisterInfo::isPhysicalRegister(SrcReg)
          ? tri_->getPhysicalRegisterRegClass(SrcReg)
          : mri_->getRegClass(SrcReg);
        if (!tri_->getMatchingSuperReg(PhysReg, SubIdx, RC))
          return true;
      }
    }
    if (MI->getOpcode() == TargetInstrInfo::INSERT_SUBREG ||
        MI->getOpcode() == TargetInstrInfo::SUBREG_TO_REG) {
      SubIdx = MI->getOperand(3).getImm();
      if (VirtReg == MI->getOperand(0).getReg()) {
        if (!tri_->getSubReg(PhysReg, SubIdx))
          return true;
      } else {
        unsigned DstReg = MI->getOperand(0).getReg();
        const TargetRegisterClass *RC =
          TargetRegisterInfo::isPhysicalRegister(DstReg)
          ? tri_->getPhysicalRegisterRegClass(DstReg)
          : mri_->getRegClass(DstReg);
        if (!tri_->getMatchingSuperReg(PhysReg, SubIdx, RC))
          return true;
      }
    }
  }
  return false;
}


/// CanJoinExtractSubRegToPhysReg - Return true if it's possible to coalesce
/// an extract_subreg where dst is a physical register, e.g.
/// cl = EXTRACT_SUBREG reg1024, 1
bool
SimpleRegisterCoalescing::CanJoinExtractSubRegToPhysReg(unsigned DstReg,
                                               unsigned SrcReg, unsigned SubIdx,
                                               unsigned &RealDstReg) {
  const TargetRegisterClass *RC = mri_->getRegClass(SrcReg);
  RealDstReg = tri_->getMatchingSuperReg(DstReg, SubIdx, RC);
  assert(RealDstReg && "Invalid extract_subreg instruction!");

  // For this type of EXTRACT_SUBREG, conservatively
  // check if the live interval of the source register interfere with the
  // actual super physical register we are trying to coalesce with.
  LiveInterval &RHS = li_->getInterval(SrcReg);
  if (li_->hasInterval(RealDstReg) &&
      RHS.overlaps(li_->getInterval(RealDstReg))) {
    DOUT << "Interfere with register ";
    DEBUG(li_->getInterval(RealDstReg).print(DOUT, tri_));
    return false; // Not coalescable
  }
  for (const unsigned* SR = tri_->getSubRegisters(RealDstReg); *SR; ++SR)
    if (li_->hasInterval(*SR) && RHS.overlaps(li_->getInterval(*SR))) {
      DOUT << "Interfere with sub-register ";
      DEBUG(li_->getInterval(*SR).print(DOUT, tri_));
      return false; // Not coalescable
    }
  return true;
}

/// CanJoinInsertSubRegToPhysReg - Return true if it's possible to coalesce
/// an insert_subreg where src is a physical register, e.g.
/// reg1024 = INSERT_SUBREG reg1024, c1, 0
bool
SimpleRegisterCoalescing::CanJoinInsertSubRegToPhysReg(unsigned DstReg,
                                               unsigned SrcReg, unsigned SubIdx,
                                               unsigned &RealSrcReg) {
  const TargetRegisterClass *RC = mri_->getRegClass(DstReg);
  RealSrcReg = tri_->getMatchingSuperReg(SrcReg, SubIdx, RC);
  assert(RealSrcReg && "Invalid extract_subreg instruction!");

  LiveInterval &RHS = li_->getInterval(DstReg);
  if (li_->hasInterval(RealSrcReg) &&
      RHS.overlaps(li_->getInterval(RealSrcReg))) {
    DOUT << "Interfere with register ";
    DEBUG(li_->getInterval(RealSrcReg).print(DOUT, tri_));
    return false; // Not coalescable
  }
  for (const unsigned* SR = tri_->getSubRegisters(RealSrcReg); *SR; ++SR)
    if (li_->hasInterval(*SR) && RHS.overlaps(li_->getInterval(*SR))) {
      DOUT << "Interfere with sub-register ";
      DEBUG(li_->getInterval(*SR).print(DOUT, tri_));
      return false; // Not coalescable
    }
  return true;
}

/// getRegAllocPreference - Return register allocation preference register.
///
static unsigned getRegAllocPreference(unsigned Reg, MachineFunction &MF,
                                      MachineRegisterInfo *MRI,
                                      const TargetRegisterInfo *TRI) {
  if (TargetRegisterInfo::isPhysicalRegister(Reg))
    return 0;
  std::pair<unsigned, unsigned> Hint = MRI->getRegAllocationHint(Reg);
  return TRI->ResolveRegAllocHint(Hint.first, Hint.second, MF);
}

/// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
/// which are the src/dst of the copy instruction CopyMI.  This returns true
/// if the copy was successfully coalesced away. If it is not currently
/// possible to coalesce this interval, but it may be possible if other
/// things get coalesced, then it returns true by reference in 'Again'.
bool SimpleRegisterCoalescing::JoinCopy(CopyRec &TheCopy, bool &Again) {
  MachineInstr *CopyMI = TheCopy.MI;

  Again = false;
  if (JoinedCopies.count(CopyMI) || ReMatCopies.count(CopyMI))
    return false; // Already done.

  DOUT << li_->getInstructionIndex(CopyMI) << '\t' << *CopyMI;

  unsigned SrcReg, DstReg, SrcSubIdx = 0, DstSubIdx = 0;
  bool isExtSubReg = CopyMI->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG;
  bool isInsSubReg = CopyMI->getOpcode() == TargetInstrInfo::INSERT_SUBREG;
  bool isSubRegToReg = CopyMI->getOpcode() == TargetInstrInfo::SUBREG_TO_REG;
  unsigned SubIdx = 0;
  if (isExtSubReg) {
    DstReg    = CopyMI->getOperand(0).getReg();
    DstSubIdx = CopyMI->getOperand(0).getSubReg();
    SrcReg    = CopyMI->getOperand(1).getReg();
    SrcSubIdx = CopyMI->getOperand(2).getImm();
  } else if (isInsSubReg || isSubRegToReg) {
    DstReg    = CopyMI->getOperand(0).getReg();
    DstSubIdx = CopyMI->getOperand(3).getImm();
    SrcReg    = CopyMI->getOperand(2).getReg();
    SrcSubIdx = CopyMI->getOperand(2).getSubReg();
    if (SrcSubIdx && SrcSubIdx != DstSubIdx) {
      // r1025 = INSERT_SUBREG r1025, r1024<2>, 2 Then r1024 has already been
      // coalesced to a larger register so the subreg indices cancel out.
      DOUT << "\tSource of insert_subreg is already coalesced "
           << "to another register.\n";
      return false;  // Not coalescable.
    }
  } else if (!tii_->isMoveInstr(*CopyMI, SrcReg, DstReg, SrcSubIdx, DstSubIdx)){
    llvm_unreachable("Unrecognized copy instruction!");
  }

  // If they are already joined we continue.
  if (SrcReg == DstReg) {
    DOUT << "\tCopy already coalesced.\n";
    return false;  // Not coalescable.
  }
  
  bool SrcIsPhys = TargetRegisterInfo::isPhysicalRegister(SrcReg);
  bool DstIsPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);

  // If they are both physical registers, we cannot join them.
  if (SrcIsPhys && DstIsPhys) {
    DOUT << "\tCan not coalesce physregs.\n";
    return false;  // Not coalescable.
  }
  
  // We only join virtual registers with allocatable physical registers.
  if (SrcIsPhys && !allocatableRegs_[SrcReg]) {
    DOUT << "\tSrc reg is unallocatable physreg.\n";
    return false;  // Not coalescable.
  }
  if (DstIsPhys && !allocatableRegs_[DstReg]) {
    DOUT << "\tDst reg is unallocatable physreg.\n";
    return false;  // Not coalescable.
  }

  // Check that a physical source register is compatible with dst regclass
  if (SrcIsPhys) {
    unsigned SrcSubReg = SrcSubIdx ?
      tri_->getSubReg(SrcReg, SrcSubIdx) : SrcReg;
    const TargetRegisterClass *DstRC = mri_->getRegClass(DstReg);
    const TargetRegisterClass *DstSubRC = DstRC;
    if (DstSubIdx)
      DstSubRC = DstRC->getSubRegisterRegClass(DstSubIdx);
    assert(DstSubRC && "Illegal subregister index");
    if (!DstSubRC->contains(SrcSubReg)) {
      DEBUG(errs() << "\tIncompatible destination regclass: "
            << tri_->getName(SrcSubReg) << " not in " << DstSubRC->getName()
            << ".\n");
      return false;             // Not coalescable.
    }
  }

  // Check that a physical dst register is compatible with source regclass
  if (DstIsPhys) {
    unsigned DstSubReg = DstSubIdx ?
      tri_->getSubReg(DstReg, DstSubIdx) : DstReg;
    const TargetRegisterClass *SrcRC = mri_->getRegClass(SrcReg);
    const TargetRegisterClass *SrcSubRC = SrcRC;
    if (SrcSubIdx)
      SrcSubRC = SrcRC->getSubRegisterRegClass(SrcSubIdx);
    assert(SrcSubRC && "Illegal subregister index");
    if (!SrcSubRC->contains(DstReg)) {
      DEBUG(errs() << "\tIncompatible source regclass: "
            << tri_->getName(DstSubReg) << " not in " << SrcSubRC->getName()
            << ".\n");
      (void)DstSubReg;
      return false;             // Not coalescable.
    }
  }

  // Should be non-null only when coalescing to a sub-register class.
  bool CrossRC = false;
  const TargetRegisterClass *SrcRC= SrcIsPhys ? 0 : mri_->getRegClass(SrcReg);
  const TargetRegisterClass *DstRC= DstIsPhys ? 0 : mri_->getRegClass(DstReg);
  const TargetRegisterClass *NewRC = NULL;
  MachineBasicBlock *CopyMBB = CopyMI->getParent();
  unsigned RealDstReg = 0;
  unsigned RealSrcReg = 0;
  if (isExtSubReg || isInsSubReg || isSubRegToReg) {
    SubIdx = CopyMI->getOperand(isExtSubReg ? 2 : 3).getImm();
    if (SrcIsPhys && isExtSubReg) {
      // r1024 = EXTRACT_SUBREG EAX, 0 then r1024 is really going to be
      // coalesced with AX.
      unsigned DstSubIdx = CopyMI->getOperand(0).getSubReg();
      if (DstSubIdx) {
        // r1024<2> = EXTRACT_SUBREG EAX, 2. Then r1024 has already been
        // coalesced to a larger register so the subreg indices cancel out.
        if (DstSubIdx != SubIdx) {
          DOUT << "\t Sub-register indices mismatch.\n";
          return false; // Not coalescable.
        }
      } else
        SrcReg = tri_->getSubReg(SrcReg, SubIdx);
      SubIdx = 0;
    } else if (DstIsPhys && (isInsSubReg || isSubRegToReg)) {
      // EAX = INSERT_SUBREG EAX, r1024, 0
      unsigned SrcSubIdx = CopyMI->getOperand(2).getSubReg();
      if (SrcSubIdx) {
        // EAX = INSERT_SUBREG EAX, r1024<2>, 2 Then r1024 has already been
        // coalesced to a larger register so the subreg indices cancel out.
        if (SrcSubIdx != SubIdx) {
          DOUT << "\t Sub-register indices mismatch.\n";
          return false; // Not coalescable.
        }
      } else
        DstReg = tri_->getSubReg(DstReg, SubIdx);
      SubIdx = 0;
    } else if ((DstIsPhys && isExtSubReg) ||
               (SrcIsPhys && (isInsSubReg || isSubRegToReg))) {
      if (!isSubRegToReg && CopyMI->getOperand(1).getSubReg()) {
        DOUT << "\tSrc of extract_subreg already coalesced with reg"
             << " of a super-class.\n";
        return false; // Not coalescable.
      }

      if (isExtSubReg) {
        if (!CanJoinExtractSubRegToPhysReg(DstReg, SrcReg, SubIdx, RealDstReg))
          return false; // Not coalescable
      } else {
        if (!CanJoinInsertSubRegToPhysReg(DstReg, SrcReg, SubIdx, RealSrcReg))
          return false; // Not coalescable
      }
      SubIdx = 0;
    } else {
      unsigned OldSubIdx = isExtSubReg ? CopyMI->getOperand(0).getSubReg()
        : CopyMI->getOperand(2).getSubReg();
      if (OldSubIdx) {
        if (OldSubIdx == SubIdx && !differingRegisterClasses(SrcReg, DstReg))
          // r1024<2> = EXTRACT_SUBREG r1025, 2. Then r1024 has already been
          // coalesced to a larger register so the subreg indices cancel out.
          // Also check if the other larger register is of the same register
          // class as the would be resulting register.
          SubIdx = 0;
        else {
          DOUT << "\t Sub-register indices mismatch.\n";
          return false; // Not coalescable.
        }
      }
      if (SubIdx) {
        if (!DstIsPhys && !SrcIsPhys) {
          if (isInsSubReg || isSubRegToReg) {
            NewRC = tri_->getMatchingSuperRegClass(DstRC, SrcRC, SubIdx);
          } else // extract_subreg {
            NewRC = tri_->getMatchingSuperRegClass(SrcRC, DstRC, SubIdx);
          }
        if (!NewRC) {
          DOUT << "\t Conflicting sub-register indices.\n";
          return false;  // Not coalescable
        }

        unsigned LargeReg = isExtSubReg ? SrcReg : DstReg;
        unsigned SmallReg = isExtSubReg ? DstReg : SrcReg;
        unsigned Limit= allocatableRCRegs_[mri_->getRegClass(SmallReg)].count();
        if (!isWinToJoinCrossClass(LargeReg, SmallReg, Limit)) {
          Again = true;  // May be possible to coalesce later.
          return false;
        }
      }
    }
  } else if (differingRegisterClasses(SrcReg, DstReg)) {
    if (DisableCrossClassJoin)
      return false;
    CrossRC = true;

    // FIXME: What if the result of a EXTRACT_SUBREG is then coalesced
    // with another? If it's the resulting destination register, then
    // the subidx must be propagated to uses (but only those defined
    // by the EXTRACT_SUBREG). If it's being coalesced into another
    // register, it should be safe because register is assumed to have
    // the register class of the super-register.

    // Process moves where one of the registers have a sub-register index.
    MachineOperand *DstMO = CopyMI->findRegisterDefOperand(DstReg);
    MachineOperand *SrcMO = CopyMI->findRegisterUseOperand(SrcReg);
    SubIdx = DstMO->getSubReg();
    if (SubIdx) {
      if (SrcMO->getSubReg())
        // FIXME: can we handle this?
        return false;
      // This is not an insert_subreg but it looks like one.
      // e.g. %reg1024:4 = MOV32rr %EAX
      isInsSubReg = true;
      if (SrcIsPhys) {
        if (!CanJoinInsertSubRegToPhysReg(DstReg, SrcReg, SubIdx, RealSrcReg))
          return false; // Not coalescable
        SubIdx = 0;
      }
    } else {
      SubIdx = SrcMO->getSubReg();
      if (SubIdx) {
        // This is not a extract_subreg but it looks like one.
        // e.g. %cl = MOV16rr %reg1024:1
        isExtSubReg = true;
        if (DstIsPhys) {
          if (!CanJoinExtractSubRegToPhysReg(DstReg, SrcReg, SubIdx,RealDstReg))
            return false; // Not coalescable
          SubIdx = 0;
        }
      }
    }

    unsigned LargeReg = SrcReg;
    unsigned SmallReg = DstReg;

    // Now determine the register class of the joined register.
    if (isExtSubReg) {
      if (SubIdx && DstRC && DstRC->isASubClass()) {
        // This is a move to a sub-register class. However, the source is a
        // sub-register of a larger register class. We don't know what should
        // the register class be. FIXME.
        Again = true;
        return false;
      }
      if (!DstIsPhys && !SrcIsPhys)
        NewRC = SrcRC;
    } else if (!SrcIsPhys && !DstIsPhys) {
      NewRC = getCommonSubClass(SrcRC, DstRC);
      if (!NewRC) {
        DEBUG(errs() << "\tDisjoint regclasses: "
              << SrcRC->getName() << ", "
              << DstRC->getName() << ".\n");
        return false;           // Not coalescable.
      }
      if (DstRC->getSize() > SrcRC->getSize())
        std::swap(LargeReg, SmallReg);
    }

    // If we are joining two virtual registers and the resulting register
    // class is more restrictive (fewer register, smaller size). Check if it's
    // worth doing the merge.
    if (!SrcIsPhys && !DstIsPhys &&
        (isExtSubReg || DstRC->isASubClass()) &&
        !isWinToJoinCrossClass(LargeReg, SmallReg,
                               allocatableRCRegs_[NewRC].count())) {
      DOUT << "\tSrc/Dest are different register classes.\n";
      // Allow the coalescer to try again in case either side gets coalesced to
      // a physical register that's compatible with the other side. e.g.
      // r1024 = MOV32to32_ r1025
      // But later r1024 is assigned EAX then r1025 may be coalesced with EAX.
      Again = true;  // May be possible to coalesce later.
      return false;
    }
  }

  // Will it create illegal extract_subreg / insert_subreg?
  if (SrcIsPhys && HasIncompatibleSubRegDefUse(CopyMI, DstReg, SrcReg))
    return false;
  if (DstIsPhys && HasIncompatibleSubRegDefUse(CopyMI, SrcReg, DstReg))
    return false;
  
  LiveInterval &SrcInt = li_->getInterval(SrcReg);
  LiveInterval &DstInt = li_->getInterval(DstReg);
  assert(SrcInt.reg == SrcReg && DstInt.reg == DstReg &&
         "Register mapping is horribly broken!");

  DOUT << "\t\tInspecting "; SrcInt.print(DOUT, tri_);
  DOUT << " and "; DstInt.print(DOUT, tri_);
  DOUT << ": ";

  // Save a copy of the virtual register live interval. We'll manually
  // merge this into the "real" physical register live interval this is
  // coalesced with.
  LiveInterval *SavedLI = 0;
  if (RealDstReg)
    SavedLI = li_->dupInterval(&SrcInt);
  else if (RealSrcReg)
    SavedLI = li_->dupInterval(&DstInt);

  // Check if it is necessary to propagate "isDead" property.
  if (!isExtSubReg && !isInsSubReg && !isSubRegToReg) {
    MachineOperand *mopd = CopyMI->findRegisterDefOperand(DstReg, false);
    bool isDead = mopd->isDead();

    // We need to be careful about coalescing a source physical register with a
    // virtual register. Once the coalescing is done, it cannot be broken and
    // these are not spillable! If the destination interval uses are far away,
    // think twice about coalescing them!
    if (!isDead && (SrcIsPhys || DstIsPhys)) {
      // If the copy is in a loop, take care not to coalesce aggressively if the
      // src is coming in from outside the loop (or the dst is out of the loop).
      // If it's not in a loop, then determine whether to join them base purely
      // by the length of the interval.
      if (PhysJoinTweak) {
        if (SrcIsPhys) {
          if (!isWinToJoinVRWithSrcPhysReg(CopyMI, CopyMBB, DstInt, SrcInt)) {
            mri_->setRegAllocationHint(DstInt.reg, 0, SrcReg);
            ++numAborts;
            DOUT << "\tMay tie down a physical register, abort!\n";
            Again = true;  // May be possible to coalesce later.
            return false;
          }
        } else {
          if (!isWinToJoinVRWithDstPhysReg(CopyMI, CopyMBB, DstInt, SrcInt)) {
            mri_->setRegAllocationHint(SrcInt.reg, 0, DstReg);
            ++numAborts;
            DOUT << "\tMay tie down a physical register, abort!\n";
            Again = true;  // May be possible to coalesce later.
            return false;
          }
        }
      } else {
        // If the virtual register live interval is long but it has low use desity,
        // do not join them, instead mark the physical register as its allocation
        // preference.
        LiveInterval &JoinVInt = SrcIsPhys ? DstInt : SrcInt;
        unsigned JoinVReg = SrcIsPhys ? DstReg : SrcReg;
        unsigned JoinPReg = SrcIsPhys ? SrcReg : DstReg;
        const TargetRegisterClass *RC = mri_->getRegClass(JoinVReg);
        unsigned Threshold = allocatableRCRegs_[RC].count() * 2;
        if (TheCopy.isBackEdge)
          Threshold *= 2; // Favors back edge copies.

        unsigned Length = li_->getApproximateInstructionCount(JoinVInt);
        float Ratio = 1.0 / Threshold;
        if (Length > Threshold &&
            (((float)std::distance(mri_->use_begin(JoinVReg),
                                   mri_->use_end()) / Length) < Ratio)) {
          mri_->setRegAllocationHint(JoinVInt.reg, 0, JoinPReg);
          ++numAborts;
          DOUT << "\tMay tie down a physical register, abort!\n";
          Again = true;  // May be possible to coalesce later.
          return false;
        }
      }
    }
  }

  // Okay, attempt to join these two intervals.  On failure, this returns false.
  // Otherwise, if one of the intervals being joined is a physreg, this method
  // always canonicalizes DstInt to be it.  The output "SrcInt" will not have
  // been modified, so we can use this information below to update aliases.
  bool Swapped = false;
  // If SrcInt is implicitly defined, it's safe to coalesce.
  bool isEmpty = SrcInt.empty();
  if (isEmpty && !CanCoalesceWithImpDef(CopyMI, DstInt, SrcInt)) {
    // Only coalesce an empty interval (defined by implicit_def) with
    // another interval which has a valno defined by the CopyMI and the CopyMI
    // is a kill of the implicit def.
    DOUT << "Not profitable!\n";
    return false;
  }

  if (!isEmpty && !JoinIntervals(DstInt, SrcInt, Swapped)) {
    // Coalescing failed.

    // If definition of source is defined by trivial computation, try
    // rematerializing it.
    if (!isExtSubReg && !isInsSubReg && !isSubRegToReg &&
        ReMaterializeTrivialDef(SrcInt, DstReg, DstSubIdx, CopyMI))
      return true;
    
    // If we can eliminate the copy without merging the live ranges, do so now.
    if (!isExtSubReg && !isInsSubReg && !isSubRegToReg &&
        (AdjustCopiesBackFrom(SrcInt, DstInt, CopyMI) ||
         RemoveCopyByCommutingDef(SrcInt, DstInt, CopyMI))) {
      JoinedCopies.insert(CopyMI);
      return true;
    }
    
    // Otherwise, we are unable to join the intervals.
    DOUT << "Interference!\n";
    Again = true;  // May be possible to coalesce later.
    return false;
  }

  LiveInterval *ResSrcInt = &SrcInt;
  LiveInterval *ResDstInt = &DstInt;
  if (Swapped) {
    std::swap(SrcReg, DstReg);
    std::swap(ResSrcInt, ResDstInt);
  }
  assert(TargetRegisterInfo::isVirtualRegister(SrcReg) &&
         "LiveInterval::join didn't work right!");
                               
  // If we're about to merge live ranges into a physical register live interval,
  // we have to update any aliased register's live ranges to indicate that they
  // have clobbered values for this range.
  if (TargetRegisterInfo::isPhysicalRegister(DstReg)) {
    // If this is a extract_subreg where dst is a physical register, e.g.
    // cl = EXTRACT_SUBREG reg1024, 1
    // then create and update the actual physical register allocated to RHS.
    if (RealDstReg || RealSrcReg) {
      LiveInterval &RealInt =
        li_->getOrCreateInterval(RealDstReg ? RealDstReg : RealSrcReg);
      for (LiveInterval::const_vni_iterator I = SavedLI->vni_begin(),
             E = SavedLI->vni_end(); I != E; ++I) {
        const VNInfo *ValNo = *I;
        VNInfo *NewValNo = RealInt.getNextValue(ValNo->def, ValNo->getCopy(),
                                                false, // updated at *
                                                li_->getVNInfoAllocator());
        NewValNo->setFlags(ValNo->getFlags()); // * updated here.
        RealInt.addKills(NewValNo, ValNo->kills);
        RealInt.MergeValueInAsValue(*SavedLI, ValNo, NewValNo);
      }
      RealInt.weight += SavedLI->weight;      
      DstReg = RealDstReg ? RealDstReg : RealSrcReg;
    }

    // Update the liveintervals of sub-registers.
    for (const unsigned *AS = tri_->getSubRegisters(DstReg); *AS; ++AS)
      li_->getOrCreateInterval(*AS).MergeInClobberRanges(*ResSrcInt,
                                                 li_->getVNInfoAllocator());
  }

  // If this is a EXTRACT_SUBREG, make sure the result of coalescing is the
  // larger super-register.
  if ((isExtSubReg || isInsSubReg || isSubRegToReg) &&
      !SrcIsPhys && !DstIsPhys) {
    if ((isExtSubReg && !Swapped) ||
        ((isInsSubReg || isSubRegToReg) && Swapped)) {
      ResSrcInt->Copy(*ResDstInt, mri_, li_->getVNInfoAllocator());
      std::swap(SrcReg, DstReg);
      std::swap(ResSrcInt, ResDstInt);
    }
  }

  // Coalescing to a virtual register that is of a sub-register class of the
  // other. Make sure the resulting register is set to the right register class.
  if (CrossRC)
    ++numCrossRCs;

  // This may happen even if it's cross-rc coalescing. e.g.
  // %reg1026<def> = SUBREG_TO_REG 0, %reg1037<kill>, 4
  // reg1026 -> GR64, reg1037 -> GR32_ABCD. The resulting register will have to
  // be allocate a register from GR64_ABCD.
  if (NewRC)
    mri_->setRegClass(DstReg, NewRC);

  if (NewHeuristic) {
    // Add all copies that define val# in the source interval into the queue.
    for (LiveInterval::const_vni_iterator i = ResSrcInt->vni_begin(),
           e = ResSrcInt->vni_end(); i != e; ++i) {
      const VNInfo *vni = *i;
      // FIXME: Do isPHIDef and isDefAccurate both need to be tested?
      if (!vni->def || vni->isUnused() || vni->isPHIDef() || !vni->isDefAccurate())
        continue;
      MachineInstr *CopyMI = li_->getInstructionFromIndex(vni->def);
      unsigned NewSrcReg, NewDstReg, NewSrcSubIdx, NewDstSubIdx;
      if (CopyMI &&
          JoinedCopies.count(CopyMI) == 0 &&
          tii_->isMoveInstr(*CopyMI, NewSrcReg, NewDstReg,
                            NewSrcSubIdx, NewDstSubIdx)) {
        unsigned LoopDepth = loopInfo->getLoopDepth(CopyMBB);
        JoinQueue->push(CopyRec(CopyMI, LoopDepth,
                                isBackEdgeCopy(CopyMI, DstReg)));
      }
    }
  }

  // Remember to delete the copy instruction.
  JoinedCopies.insert(CopyMI);

  // Some live range has been lengthened due to colaescing, eliminate the
  // unnecessary kills.
  RemoveUnnecessaryKills(SrcReg, *ResDstInt);
  if (TargetRegisterInfo::isVirtualRegister(DstReg))
    RemoveUnnecessaryKills(DstReg, *ResDstInt);

  UpdateRegDefsUses(SrcReg, DstReg, SubIdx);

  // SrcReg is guarateed to be the register whose live interval that is
  // being merged.
  li_->removeInterval(SrcReg);

  // Update regalloc hint.
  tri_->UpdateRegAllocHint(SrcReg, DstReg, *mf_);

  // Manually deleted the live interval copy.
  if (SavedLI) {
    SavedLI->clear();
    delete SavedLI;
  }

  // If resulting interval has a preference that no longer fits because of subreg
  // coalescing, just clear the preference.
  unsigned Preference = getRegAllocPreference(ResDstInt->reg, *mf_, mri_, tri_);
  if (Preference && (isExtSubReg || isInsSubReg || isSubRegToReg) &&
      TargetRegisterInfo::isVirtualRegister(ResDstInt->reg)) {
    const TargetRegisterClass *RC = mri_->getRegClass(ResDstInt->reg);
    if (!RC->contains(Preference))
      mri_->setRegAllocationHint(ResDstInt->reg, 0, 0);
  }

  DOUT << "\n\t\tJoined.  Result = "; ResDstInt->print(DOUT, tri_);
  DOUT << "\n";

  ++numJoins;
  return true;
}

/// ComputeUltimateVN - Assuming we are going to join two live intervals,
/// compute what the resultant value numbers for each value in the input two
/// ranges will be.  This is complicated by copies between the two which can
/// and will commonly cause multiple value numbers to be merged into one.
///
/// VN is the value number that we're trying to resolve.  InstDefiningValue
/// keeps track of the new InstDefiningValue assignment for the result
/// LiveInterval.  ThisFromOther/OtherFromThis are sets that keep track of
/// whether a value in this or other is a copy from the opposite set.
/// ThisValNoAssignments/OtherValNoAssignments keep track of value #'s that have
/// already been assigned.
///
/// ThisFromOther[x] - If x is defined as a copy from the other interval, this
/// contains the value number the copy is from.
///
static unsigned ComputeUltimateVN(VNInfo *VNI,
                                  SmallVector<VNInfo*, 16> &NewVNInfo,
                                  DenseMap<VNInfo*, VNInfo*> &ThisFromOther,
                                  DenseMap<VNInfo*, VNInfo*> &OtherFromThis,
                                  SmallVector<int, 16> &ThisValNoAssignments,
                                  SmallVector<int, 16> &OtherValNoAssignments) {
  unsigned VN = VNI->id;

  // If the VN has already been computed, just return it.
  if (ThisValNoAssignments[VN] >= 0)
    return ThisValNoAssignments[VN];
//  assert(ThisValNoAssignments[VN] != -2 && "Cyclic case?");

  // If this val is not a copy from the other val, then it must be a new value
  // number in the destination.
  DenseMap<VNInfo*, VNInfo*>::iterator I = ThisFromOther.find(VNI);
  if (I == ThisFromOther.end()) {
    NewVNInfo.push_back(VNI);
    return ThisValNoAssignments[VN] = NewVNInfo.size()-1;
  }
  VNInfo *OtherValNo = I->second;

  // Otherwise, this *is* a copy from the RHS.  If the other side has already
  // been computed, return it.
  if (OtherValNoAssignments[OtherValNo->id] >= 0)
    return ThisValNoAssignments[VN] = OtherValNoAssignments[OtherValNo->id];
  
  // Mark this value number as currently being computed, then ask what the
  // ultimate value # of the other value is.
  ThisValNoAssignments[VN] = -2;
  unsigned UltimateVN =
    ComputeUltimateVN(OtherValNo, NewVNInfo, OtherFromThis, ThisFromOther,
                      OtherValNoAssignments, ThisValNoAssignments);
  return ThisValNoAssignments[VN] = UltimateVN;
}

static bool InVector(VNInfo *Val, const SmallVector<VNInfo*, 8> &V) {
  return std::find(V.begin(), V.end(), Val) != V.end();
}

/// RangeIsDefinedByCopyFromReg - Return true if the specified live range of
/// the specified live interval is defined by a copy from the specified
/// register.
bool SimpleRegisterCoalescing::RangeIsDefinedByCopyFromReg(LiveInterval &li,
                                                           LiveRange *LR,
                                                           unsigned Reg) {
  unsigned SrcReg = li_->getVNInfoSourceReg(LR->valno);
  if (SrcReg == Reg)
    return true;
  // FIXME: Do isPHIDef and isDefAccurate both need to be tested?
  if ((LR->valno->isPHIDef() || !LR->valno->isDefAccurate()) &&
      TargetRegisterInfo::isPhysicalRegister(li.reg) &&
      *tri_->getSuperRegisters(li.reg)) {
    // It's a sub-register live interval, we may not have precise information.
    // Re-compute it.
    MachineInstr *DefMI = li_->getInstructionFromIndex(LR->start);
    unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
    if (DefMI &&
        tii_->isMoveInstr(*DefMI, SrcReg, DstReg, SrcSubIdx, DstSubIdx) &&
        DstReg == li.reg && SrcReg == Reg) {
      // Cache computed info.
      LR->valno->def  = LR->start;
      LR->valno->setCopy(DefMI);
      return true;
    }
  }
  return false;
}

/// SimpleJoin - Attempt to joint the specified interval into this one. The
/// caller of this method must guarantee that the RHS only contains a single
/// value number and that the RHS is not defined by a copy from this
/// interval.  This returns false if the intervals are not joinable, or it
/// joins them and returns true.
bool SimpleRegisterCoalescing::SimpleJoin(LiveInterval &LHS, LiveInterval &RHS){
  assert(RHS.containsOneValue());
  
  // Some number (potentially more than one) value numbers in the current
  // interval may be defined as copies from the RHS.  Scan the overlapping
  // portions of the LHS and RHS, keeping track of this and looking for
  // overlapping live ranges that are NOT defined as copies.  If these exist, we
  // cannot coalesce.
  
  LiveInterval::iterator LHSIt = LHS.begin(), LHSEnd = LHS.end();
  LiveInterval::iterator RHSIt = RHS.begin(), RHSEnd = RHS.end();
  
  if (LHSIt->start < RHSIt->start) {
    LHSIt = std::upper_bound(LHSIt, LHSEnd, RHSIt->start);
    if (LHSIt != LHS.begin()) --LHSIt;
  } else if (RHSIt->start < LHSIt->start) {
    RHSIt = std::upper_bound(RHSIt, RHSEnd, LHSIt->start);
    if (RHSIt != RHS.begin()) --RHSIt;
  }
  
  SmallVector<VNInfo*, 8> EliminatedLHSVals;
  
  while (1) {
    // Determine if these live intervals overlap.
    bool Overlaps = false;
    if (LHSIt->start <= RHSIt->start)
      Overlaps = LHSIt->end > RHSIt->start;
    else
      Overlaps = RHSIt->end > LHSIt->start;
    
    // If the live intervals overlap, there are two interesting cases: if the
    // LHS interval is defined by a copy from the RHS, it's ok and we record
    // that the LHS value # is the same as the RHS.  If it's not, then we cannot
    // coalesce these live ranges and we bail out.
    if (Overlaps) {
      // If we haven't already recorded that this value # is safe, check it.
      if (!InVector(LHSIt->valno, EliminatedLHSVals)) {
        // Copy from the RHS?
        if (!RangeIsDefinedByCopyFromReg(LHS, LHSIt, RHS.reg))
          return false;    // Nope, bail out.

        if (LHSIt->contains(RHSIt->valno->def))
          // Here is an interesting situation:
          // BB1:
          //   vr1025 = copy vr1024
          //   ..
          // BB2:
          //   vr1024 = op 
          //          = vr1025
          // Even though vr1025 is copied from vr1024, it's not safe to
          // coalesce them since the live range of vr1025 intersects the
          // def of vr1024. This happens because vr1025 is assigned the
          // value of the previous iteration of vr1024.
          return false;
        EliminatedLHSVals.push_back(LHSIt->valno);
      }
      
      // We know this entire LHS live range is okay, so skip it now.
      if (++LHSIt == LHSEnd) break;
      continue;
    }
    
    if (LHSIt->end < RHSIt->end) {
      if (++LHSIt == LHSEnd) break;
    } else {
      // One interesting case to check here.  It's possible that we have
      // something like "X3 = Y" which defines a new value number in the LHS,
      // and is the last use of this liverange of the RHS.  In this case, we
      // want to notice this copy (so that it gets coalesced away) even though
      // the live ranges don't actually overlap.
      if (LHSIt->start == RHSIt->end) {
        if (InVector(LHSIt->valno, EliminatedLHSVals)) {
          // We already know that this value number is going to be merged in
          // if coalescing succeeds.  Just skip the liverange.
          if (++LHSIt == LHSEnd) break;
        } else {
          // Otherwise, if this is a copy from the RHS, mark it as being merged
          // in.
          if (RangeIsDefinedByCopyFromReg(LHS, LHSIt, RHS.reg)) {
            if (LHSIt->contains(RHSIt->valno->def))
              // Here is an interesting situation:
              // BB1:
              //   vr1025 = copy vr1024
              //   ..
              // BB2:
              //   vr1024 = op 
              //          = vr1025
              // Even though vr1025 is copied from vr1024, it's not safe to
              // coalesced them since live range of vr1025 intersects the
              // def of vr1024. This happens because vr1025 is assigned the
              // value of the previous iteration of vr1024.
              return false;
            EliminatedLHSVals.push_back(LHSIt->valno);

            // We know this entire LHS live range is okay, so skip it now.
            if (++LHSIt == LHSEnd) break;
          }
        }
      }
      
      if (++RHSIt == RHSEnd) break;
    }
  }
  
  // If we got here, we know that the coalescing will be successful and that
  // the value numbers in EliminatedLHSVals will all be merged together.  Since
  // the most common case is that EliminatedLHSVals has a single number, we
  // optimize for it: if there is more than one value, we merge them all into
  // the lowest numbered one, then handle the interval as if we were merging
  // with one value number.
  VNInfo *LHSValNo = NULL;
  if (EliminatedLHSVals.size() > 1) {
    // Loop through all the equal value numbers merging them into the smallest
    // one.
    VNInfo *Smallest = EliminatedLHSVals[0];
    for (unsigned i = 1, e = EliminatedLHSVals.size(); i != e; ++i) {
      if (EliminatedLHSVals[i]->id < Smallest->id) {
        // Merge the current notion of the smallest into the smaller one.
        LHS.MergeValueNumberInto(Smallest, EliminatedLHSVals[i]);
        Smallest = EliminatedLHSVals[i];
      } else {
        // Merge into the smallest.
        LHS.MergeValueNumberInto(EliminatedLHSVals[i], Smallest);
      }
    }
    LHSValNo = Smallest;
  } else if (EliminatedLHSVals.empty()) {
    if (TargetRegisterInfo::isPhysicalRegister(LHS.reg) &&
        *tri_->getSuperRegisters(LHS.reg))
      // Imprecise sub-register information. Can't handle it.
      return false;
    llvm_unreachable("No copies from the RHS?");
  } else {
    LHSValNo = EliminatedLHSVals[0];
  }
  
  // Okay, now that there is a single LHS value number that we're merging the
  // RHS into, update the value number info for the LHS to indicate that the
  // value number is defined where the RHS value number was.
  const VNInfo *VNI = RHS.getValNumInfo(0);
  LHSValNo->def  = VNI->def;
  LHSValNo->setCopy(VNI->getCopy());
  
  // Okay, the final step is to loop over the RHS live intervals, adding them to
  // the LHS.
  if (VNI->hasPHIKill())
    LHSValNo->setHasPHIKill(true);
  LHS.addKills(LHSValNo, VNI->kills);
  LHS.MergeRangesInAsValue(RHS, LHSValNo);

  LHS.ComputeJoinedWeight(RHS);

  // Update regalloc hint if both are virtual registers.
  if (TargetRegisterInfo::isVirtualRegister(LHS.reg) && 
      TargetRegisterInfo::isVirtualRegister(RHS.reg)) {
    std::pair<unsigned, unsigned> RHSPref = mri_->getRegAllocationHint(RHS.reg);
    std::pair<unsigned, unsigned> LHSPref = mri_->getRegAllocationHint(LHS.reg);
    if (RHSPref != LHSPref)
      mri_->setRegAllocationHint(LHS.reg, RHSPref.first, RHSPref.second);
  }

  // Update the liveintervals of sub-registers.
  if (TargetRegisterInfo::isPhysicalRegister(LHS.reg))
    for (const unsigned *AS = tri_->getSubRegisters(LHS.reg); *AS; ++AS)
      li_->getOrCreateInterval(*AS).MergeInClobberRanges(LHS,
                                                    li_->getVNInfoAllocator());

  return true;
}

/// JoinIntervals - Attempt to join these two intervals.  On failure, this
/// returns false.  Otherwise, if one of the intervals being joined is a
/// physreg, this method always canonicalizes LHS to be it.  The output
/// "RHS" will not have been modified, so we can use this information
/// below to update aliases.
bool
SimpleRegisterCoalescing::JoinIntervals(LiveInterval &LHS, LiveInterval &RHS,
                                        bool &Swapped) {
  // Compute the final value assignment, assuming that the live ranges can be
  // coalesced.
  SmallVector<int, 16> LHSValNoAssignments;
  SmallVector<int, 16> RHSValNoAssignments;
  DenseMap<VNInfo*, VNInfo*> LHSValsDefinedFromRHS;
  DenseMap<VNInfo*, VNInfo*> RHSValsDefinedFromLHS;
  SmallVector<VNInfo*, 16> NewVNInfo;

  // If a live interval is a physical register, conservatively check if any
  // of its sub-registers is overlapping the live interval of the virtual
  // register. If so, do not coalesce.
  if (TargetRegisterInfo::isPhysicalRegister(LHS.reg) &&
      *tri_->getSubRegisters(LHS.reg)) {
    // If it's coalescing a virtual register to a physical register, estimate
    // its live interval length. This is the *cost* of scanning an entire live
    // interval. If the cost is low, we'll do an exhaustive check instead.

    // If this is something like this:
    // BB1:
    // v1024 = op
    // ...
    // BB2:
    // ...
    // RAX   = v1024
    //
    // That is, the live interval of v1024 crosses a bb. Then we can't rely on
    // less conservative check. It's possible a sub-register is defined before
    // v1024 (or live in) and live out of BB1.
    if (RHS.containsOneValue() &&
        li_->intervalIsInOneMBB(RHS) &&
        li_->getApproximateInstructionCount(RHS) <= 10) {
      // Perform a more exhaustive check for some common cases.
      if (li_->conflictsWithPhysRegRef(RHS, LHS.reg, true, JoinedCopies))
        return false;
    } else {
      for (const unsigned* SR = tri_->getSubRegisters(LHS.reg); *SR; ++SR)
        if (li_->hasInterval(*SR) && RHS.overlaps(li_->getInterval(*SR))) {
          DOUT << "Interfere with sub-register ";
          DEBUG(li_->getInterval(*SR).print(DOUT, tri_));
          return false;
        }
    }
  } else if (TargetRegisterInfo::isPhysicalRegister(RHS.reg) &&
             *tri_->getSubRegisters(RHS.reg)) {
    if (LHS.containsOneValue() &&
        li_->getApproximateInstructionCount(LHS) <= 10) {
      // Perform a more exhaustive check for some common cases.
      if (li_->conflictsWithPhysRegRef(LHS, RHS.reg, false, JoinedCopies))
        return false;
    } else {
      for (const unsigned* SR = tri_->getSubRegisters(RHS.reg); *SR; ++SR)
        if (li_->hasInterval(*SR) && LHS.overlaps(li_->getInterval(*SR))) {
          DOUT << "Interfere with sub-register ";
          DEBUG(li_->getInterval(*SR).print(DOUT, tri_));
          return false;
        }
    }
  }
                          
  // Compute ultimate value numbers for the LHS and RHS values.
  if (RHS.containsOneValue()) {
    // Copies from a liveinterval with a single value are simple to handle and
    // very common, handle the special case here.  This is important, because
    // often RHS is small and LHS is large (e.g. a physreg).
    
    // Find out if the RHS is defined as a copy from some value in the LHS.
    int RHSVal0DefinedFromLHS = -1;
    int RHSValID = -1;
    VNInfo *RHSValNoInfo = NULL;
    VNInfo *RHSValNoInfo0 = RHS.getValNumInfo(0);
    unsigned RHSSrcReg = li_->getVNInfoSourceReg(RHSValNoInfo0);
    if (RHSSrcReg == 0 || RHSSrcReg != LHS.reg) {
      // If RHS is not defined as a copy from the LHS, we can use simpler and
      // faster checks to see if the live ranges are coalescable.  This joiner
      // can't swap the LHS/RHS intervals though.
      if (!TargetRegisterInfo::isPhysicalRegister(RHS.reg)) {
        return SimpleJoin(LHS, RHS);
      } else {
        RHSValNoInfo = RHSValNoInfo0;
      }
    } else {
      // It was defined as a copy from the LHS, find out what value # it is.
      RHSValNoInfo = LHS.getLiveRangeContaining(RHSValNoInfo0->def-1)->valno;
      RHSValID = RHSValNoInfo->id;
      RHSVal0DefinedFromLHS = RHSValID;
    }
    
    LHSValNoAssignments.resize(LHS.getNumValNums(), -1);
    RHSValNoAssignments.resize(RHS.getNumValNums(), -1);
    NewVNInfo.resize(LHS.getNumValNums(), NULL);
    
    // Okay, *all* of the values in LHS that are defined as a copy from RHS
    // should now get updated.
    for (LiveInterval::vni_iterator i = LHS.vni_begin(), e = LHS.vni_end();
         i != e; ++i) {
      VNInfo *VNI = *i;
      unsigned VN = VNI->id;
      if (unsigned LHSSrcReg = li_->getVNInfoSourceReg(VNI)) {
        if (LHSSrcReg != RHS.reg) {
          // If this is not a copy from the RHS, its value number will be
          // unmodified by the coalescing.
          NewVNInfo[VN] = VNI;
          LHSValNoAssignments[VN] = VN;
        } else if (RHSValID == -1) {
          // Otherwise, it is a copy from the RHS, and we don't already have a
          // value# for it.  Keep the current value number, but remember it.
          LHSValNoAssignments[VN] = RHSValID = VN;
          NewVNInfo[VN] = RHSValNoInfo;
          LHSValsDefinedFromRHS[VNI] = RHSValNoInfo0;
        } else {
          // Otherwise, use the specified value #.
          LHSValNoAssignments[VN] = RHSValID;
          if (VN == (unsigned)RHSValID) {  // Else this val# is dead.
            NewVNInfo[VN] = RHSValNoInfo;
            LHSValsDefinedFromRHS[VNI] = RHSValNoInfo0;
          }
        }
      } else {
        NewVNInfo[VN] = VNI;
        LHSValNoAssignments[VN] = VN;
      }
    }
    
    assert(RHSValID != -1 && "Didn't find value #?");
    RHSValNoAssignments[0] = RHSValID;
    if (RHSVal0DefinedFromLHS != -1) {
      // This path doesn't go through ComputeUltimateVN so just set
      // it to anything.
      RHSValsDefinedFromLHS[RHSValNoInfo0] = (VNInfo*)1;
    }
  } else {
    // Loop over the value numbers of the LHS, seeing if any are defined from
    // the RHS.
    for (LiveInterval::vni_iterator i = LHS.vni_begin(), e = LHS.vni_end();
         i != e; ++i) {
      VNInfo *VNI = *i;
      if (VNI->isUnused() || VNI->getCopy() == 0)  // Src not defined by a copy?
        continue;
      
      // DstReg is known to be a register in the LHS interval.  If the src is
      // from the RHS interval, we can use its value #.
      if (li_->getVNInfoSourceReg(VNI) != RHS.reg)
        continue;
      
      // Figure out the value # from the RHS.
      LHSValsDefinedFromRHS[VNI]=RHS.getLiveRangeContaining(VNI->def-1)->valno;
    }
    
    // Loop over the value numbers of the RHS, seeing if any are defined from
    // the LHS.
    for (LiveInterval::vni_iterator i = RHS.vni_begin(), e = RHS.vni_end();
         i != e; ++i) {
      VNInfo *VNI = *i;
      if (VNI->isUnused() || VNI->getCopy() == 0)  // Src not defined by a copy?
        continue;
      
      // DstReg is known to be a register in the RHS interval.  If the src is
      // from the LHS interval, we can use its value #.
      if (li_->getVNInfoSourceReg(VNI) != LHS.reg)
        continue;
      
      // Figure out the value # from the LHS.
      RHSValsDefinedFromLHS[VNI]=LHS.getLiveRangeContaining(VNI->def-1)->valno;
    }
    
    LHSValNoAssignments.resize(LHS.getNumValNums(), -1);
    RHSValNoAssignments.resize(RHS.getNumValNums(), -1);
    NewVNInfo.reserve(LHS.getNumValNums() + RHS.getNumValNums());
    
    for (LiveInterval::vni_iterator i = LHS.vni_begin(), e = LHS.vni_end();
         i != e; ++i) {
      VNInfo *VNI = *i;
      unsigned VN = VNI->id;
      if (LHSValNoAssignments[VN] >= 0 || VNI->isUnused()) 
        continue;
      ComputeUltimateVN(VNI, NewVNInfo,
                        LHSValsDefinedFromRHS, RHSValsDefinedFromLHS,
                        LHSValNoAssignments, RHSValNoAssignments);
    }
    for (LiveInterval::vni_iterator i = RHS.vni_begin(), e = RHS.vni_end();
         i != e; ++i) {
      VNInfo *VNI = *i;
      unsigned VN = VNI->id;
      if (RHSValNoAssignments[VN] >= 0 || VNI->isUnused())
        continue;
      // If this value number isn't a copy from the LHS, it's a new number.
      if (RHSValsDefinedFromLHS.find(VNI) == RHSValsDefinedFromLHS.end()) {
        NewVNInfo.push_back(VNI);
        RHSValNoAssignments[VN] = NewVNInfo.size()-1;
        continue;
      }
      
      ComputeUltimateVN(VNI, NewVNInfo,
                        RHSValsDefinedFromLHS, LHSValsDefinedFromRHS,
                        RHSValNoAssignments, LHSValNoAssignments);
    }
  }
  
  // Armed with the mappings of LHS/RHS values to ultimate values, walk the
  // interval lists to see if these intervals are coalescable.
  LiveInterval::const_iterator I = LHS.begin();
  LiveInterval::const_iterator IE = LHS.end();
  LiveInterval::const_iterator J = RHS.begin();
  LiveInterval::const_iterator JE = RHS.end();
  
  // Skip ahead until the first place of potential sharing.
  if (I->start < J->start) {
    I = std::upper_bound(I, IE, J->start);
    if (I != LHS.begin()) --I;
  } else if (J->start < I->start) {
    J = std::upper_bound(J, JE, I->start);
    if (J != RHS.begin()) --J;
  }
  
  while (1) {
    // Determine if these two live ranges overlap.
    bool Overlaps;
    if (I->start < J->start) {
      Overlaps = I->end > J->start;
    } else {
      Overlaps = J->end > I->start;
    }

    // If so, check value # info to determine if they are really different.
    if (Overlaps) {
      // If the live range overlap will map to the same value number in the
      // result liverange, we can still coalesce them.  If not, we can't.
      if (LHSValNoAssignments[I->valno->id] !=
          RHSValNoAssignments[J->valno->id])
        return false;
    }
    
    if (I->end < J->end) {
      ++I;
      if (I == IE) break;
    } else {
      ++J;
      if (J == JE) break;
    }
  }

  // Update kill info. Some live ranges are extended due to copy coalescing.
  for (DenseMap<VNInfo*, VNInfo*>::iterator I = LHSValsDefinedFromRHS.begin(),
         E = LHSValsDefinedFromRHS.end(); I != E; ++I) {
    VNInfo *VNI = I->first;
    unsigned LHSValID = LHSValNoAssignments[VNI->id];
    LiveInterval::removeKill(NewVNInfo[LHSValID], VNI->def);
    if (VNI->hasPHIKill())
      NewVNInfo[LHSValID]->setHasPHIKill(true);
    RHS.addKills(NewVNInfo[LHSValID], VNI->kills);
  }

  // Update kill info. Some live ranges are extended due to copy coalescing.
  for (DenseMap<VNInfo*, VNInfo*>::iterator I = RHSValsDefinedFromLHS.begin(),
         E = RHSValsDefinedFromLHS.end(); I != E; ++I) {
    VNInfo *VNI = I->first;
    unsigned RHSValID = RHSValNoAssignments[VNI->id];
    LiveInterval::removeKill(NewVNInfo[RHSValID], VNI->def);
    if (VNI->hasPHIKill())
      NewVNInfo[RHSValID]->setHasPHIKill(true);
    LHS.addKills(NewVNInfo[RHSValID], VNI->kills);
  }

  // If we get here, we know that we can coalesce the live ranges.  Ask the
  // intervals to coalesce themselves now.
  if ((RHS.ranges.size() > LHS.ranges.size() &&
      TargetRegisterInfo::isVirtualRegister(LHS.reg)) ||
      TargetRegisterInfo::isPhysicalRegister(RHS.reg)) {
    RHS.join(LHS, &RHSValNoAssignments[0], &LHSValNoAssignments[0], NewVNInfo,
             mri_);
    Swapped = true;
  } else {
    LHS.join(RHS, &LHSValNoAssignments[0], &RHSValNoAssignments[0], NewVNInfo,
             mri_);
    Swapped = false;
  }
  return true;
}

namespace {
  // DepthMBBCompare - Comparison predicate that sort first based on the loop
  // depth of the basic block (the unsigned), and then on the MBB number.
  struct DepthMBBCompare {
    typedef std::pair<unsigned, MachineBasicBlock*> DepthMBBPair;
    bool operator()(const DepthMBBPair &LHS, const DepthMBBPair &RHS) const {
      if (LHS.first > RHS.first) return true;   // Deeper loops first
      return LHS.first == RHS.first &&
        LHS.second->getNumber() < RHS.second->getNumber();
    }
  };
}

/// getRepIntervalSize - Returns the size of the interval that represents the
/// specified register.
template<class SF>
unsigned JoinPriorityQueue<SF>::getRepIntervalSize(unsigned Reg) {
  return Rc->getRepIntervalSize(Reg);
}

/// CopyRecSort::operator - Join priority queue sorting function.
///
bool CopyRecSort::operator()(CopyRec left, CopyRec right) const {
  // Inner loops first.
  if (left.LoopDepth > right.LoopDepth)
    return false;
  else if (left.LoopDepth == right.LoopDepth)
    if (left.isBackEdge && !right.isBackEdge)
      return false;
  return true;
}

void SimpleRegisterCoalescing::CopyCoalesceInMBB(MachineBasicBlock *MBB,
                                               std::vector<CopyRec> &TryAgain) {
  DEBUG(errs() << ((Value*)MBB->getBasicBlock())->getName() << ":\n");

  std::vector<CopyRec> VirtCopies;
  std::vector<CopyRec> PhysCopies;
  std::vector<CopyRec> ImpDefCopies;
  unsigned LoopDepth = loopInfo->getLoopDepth(MBB);
  for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
       MII != E;) {
    MachineInstr *Inst = MII++;
    
    // If this isn't a copy nor a extract_subreg, we can't join intervals.
    unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
    if (Inst->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG) {
      DstReg = Inst->getOperand(0).getReg();
      SrcReg = Inst->getOperand(1).getReg();
    } else if (Inst->getOpcode() == TargetInstrInfo::INSERT_SUBREG ||
               Inst->getOpcode() == TargetInstrInfo::SUBREG_TO_REG) {
      DstReg = Inst->getOperand(0).getReg();
      SrcReg = Inst->getOperand(2).getReg();
    } else if (!tii_->isMoveInstr(*Inst, SrcReg, DstReg, SrcSubIdx, DstSubIdx))
      continue;

    bool SrcIsPhys = TargetRegisterInfo::isPhysicalRegister(SrcReg);
    bool DstIsPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);
    if (NewHeuristic) {
      JoinQueue->push(CopyRec(Inst, LoopDepth, isBackEdgeCopy(Inst, DstReg)));
    } else {
      if (li_->hasInterval(SrcReg) && li_->getInterval(SrcReg).empty())
        ImpDefCopies.push_back(CopyRec(Inst, 0, false));
      else if (SrcIsPhys || DstIsPhys)
        PhysCopies.push_back(CopyRec(Inst, 0, false));
      else
        VirtCopies.push_back(CopyRec(Inst, 0, false));
    }
  }

  if (NewHeuristic)
    return;

  // Try coalescing implicit copies first, followed by copies to / from
  // physical registers, then finally copies from virtual registers to
  // virtual registers.
  for (unsigned i = 0, e = ImpDefCopies.size(); i != e; ++i) {
    CopyRec &TheCopy = ImpDefCopies[i];
    bool Again = false;
    if (!JoinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
  for (unsigned i = 0, e = PhysCopies.size(); i != e; ++i) {
    CopyRec &TheCopy = PhysCopies[i];
    bool Again = false;
    if (!JoinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
  for (unsigned i = 0, e = VirtCopies.size(); i != e; ++i) {
    CopyRec &TheCopy = VirtCopies[i];
    bool Again = false;
    if (!JoinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
}

void SimpleRegisterCoalescing::joinIntervals() {
  DOUT << "********** JOINING INTERVALS ***********\n";

  if (NewHeuristic)
    JoinQueue = new JoinPriorityQueue<CopyRecSort>(this);

  std::vector<CopyRec> TryAgainList;
  if (loopInfo->empty()) {
    // If there are no loops in the function, join intervals in function order.
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();
         I != E; ++I)
      CopyCoalesceInMBB(I, TryAgainList);
  } else {
    // Otherwise, join intervals in inner loops before other intervals.
    // Unfortunately we can't just iterate over loop hierarchy here because
    // there may be more MBB's than BB's.  Collect MBB's for sorting.

    // Join intervals in the function prolog first. We want to join physical
    // registers with virtual registers before the intervals got too long.
    std::vector<std::pair<unsigned, MachineBasicBlock*> > MBBs;
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();I != E;++I){
      MachineBasicBlock *MBB = I;
      MBBs.push_back(std::make_pair(loopInfo->getLoopDepth(MBB), I));
    }

    // Sort by loop depth.
    std::sort(MBBs.begin(), MBBs.end(), DepthMBBCompare());

    // Finally, join intervals in loop nest order.
    for (unsigned i = 0, e = MBBs.size(); i != e; ++i)
      CopyCoalesceInMBB(MBBs[i].second, TryAgainList);
  }
  
  // Joining intervals can allow other intervals to be joined.  Iteratively join
  // until we make no progress.
  if (NewHeuristic) {
    SmallVector<CopyRec, 16> TryAgain;
    bool ProgressMade = true;
    while (ProgressMade) {
      ProgressMade = false;
      while (!JoinQueue->empty()) {
        CopyRec R = JoinQueue->pop();
        bool Again = false;
        bool Success = JoinCopy(R, Again);
        if (Success)
          ProgressMade = true;
        else if (Again)
          TryAgain.push_back(R);
      }

      if (ProgressMade) {
        while (!TryAgain.empty()) {
          JoinQueue->push(TryAgain.back());
          TryAgain.pop_back();
        }
      }
    }
  } else {
    bool ProgressMade = true;
    while (ProgressMade) {
      ProgressMade = false;

      for (unsigned i = 0, e = TryAgainList.size(); i != e; ++i) {
        CopyRec &TheCopy = TryAgainList[i];
        if (TheCopy.MI) {
          bool Again = false;
          bool Success = JoinCopy(TheCopy, Again);
          if (Success || !Again) {
            TheCopy.MI = 0;   // Mark this one as done.
            ProgressMade = true;
          }
        }
      }
    }
  }

  if (NewHeuristic)
    delete JoinQueue;  
}

/// Return true if the two specified registers belong to different register
/// classes.  The registers may be either phys or virt regs.
bool
SimpleRegisterCoalescing::differingRegisterClasses(unsigned RegA,
                                                   unsigned RegB) const {
  // Get the register classes for the first reg.
  if (TargetRegisterInfo::isPhysicalRegister(RegA)) {
    assert(TargetRegisterInfo::isVirtualRegister(RegB) &&
           "Shouldn't consider two physregs!");
    return !mri_->getRegClass(RegB)->contains(RegA);
  }

  // Compare against the regclass for the second reg.
  const TargetRegisterClass *RegClassA = mri_->getRegClass(RegA);
  if (TargetRegisterInfo::isVirtualRegister(RegB)) {
    const TargetRegisterClass *RegClassB = mri_->getRegClass(RegB);
    return RegClassA != RegClassB;
  }
  return !RegClassA->contains(RegB);
}

/// lastRegisterUse - Returns the last use of the specific register between
/// cycles Start and End or NULL if there are no uses.
MachineOperand *
SimpleRegisterCoalescing::lastRegisterUse(unsigned Start, unsigned End,
                                          unsigned Reg, unsigned &UseIdx) const{
  UseIdx = 0;
  if (TargetRegisterInfo::isVirtualRegister(Reg)) {
    MachineOperand *LastUse = NULL;
    for (MachineRegisterInfo::use_iterator I = mri_->use_begin(Reg),
           E = mri_->use_end(); I != E; ++I) {
      MachineOperand &Use = I.getOperand();
      MachineInstr *UseMI = Use.getParent();
      unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
      if (tii_->isMoveInstr(*UseMI, SrcReg, DstReg, SrcSubIdx, DstSubIdx) &&
          SrcReg == DstReg)
        // Ignore identity copies.
        continue;
      unsigned Idx = li_->getInstructionIndex(UseMI);
      if (Idx >= Start && Idx < End && Idx >= UseIdx) {
        LastUse = &Use;
        UseIdx = li_->getUseIndex(Idx);
      }
    }
    return LastUse;
  }

  int e = (End-1) / InstrSlots::NUM * InstrSlots::NUM;
  int s = Start;
  while (e >= s) {
    // Skip deleted instructions
    MachineInstr *MI = li_->getInstructionFromIndex(e);
    while ((e - InstrSlots::NUM) >= s && !MI) {
      e -= InstrSlots::NUM;
      MI = li_->getInstructionFromIndex(e);
    }
    if (e < s || MI == NULL)
      return NULL;

    // Ignore identity copies.
    unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
    if (!(tii_->isMoveInstr(*MI, SrcReg, DstReg, SrcSubIdx, DstSubIdx) &&
          SrcReg == DstReg))
      for (unsigned i = 0, NumOps = MI->getNumOperands(); i != NumOps; ++i) {
        MachineOperand &Use = MI->getOperand(i);
        if (Use.isReg() && Use.isUse() && Use.getReg() &&
            tri_->regsOverlap(Use.getReg(), Reg)) {
          UseIdx = li_->getUseIndex(e);
          return &Use;
        }
      }

    e -= InstrSlots::NUM;
  }

  return NULL;
}


void SimpleRegisterCoalescing::printRegName(unsigned reg) const {
  if (TargetRegisterInfo::isPhysicalRegister(reg))
    cerr << tri_->getName(reg);
  else
    cerr << "%reg" << reg;
}

void SimpleRegisterCoalescing::releaseMemory() {
  JoinedCopies.clear();
  ReMatCopies.clear();
  ReMatDefs.clear();
}

static bool isZeroLengthInterval(LiveInterval *li) {
  for (LiveInterval::Ranges::const_iterator
         i = li->ranges.begin(), e = li->ranges.end(); i != e; ++i)
    if (i->end - i->start > LiveInterval::InstrSlots::NUM)
      return false;
  return true;
}


bool SimpleRegisterCoalescing::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  mri_ = &fn.getRegInfo();
  tm_ = &fn.getTarget();
  tri_ = tm_->getRegisterInfo();
  tii_ = tm_->getInstrInfo();
  li_ = &getAnalysis<LiveIntervals>();
  loopInfo = &getAnalysis<MachineLoopInfo>();

  DEBUG(errs() << "********** SIMPLE REGISTER COALESCING **********\n"
        << "********** Function: "
        << ((Value*)mf_->getFunction())->getName() << '\n');

  allocatableRegs_ = tri_->getAllocatableSet(fn);
  for (TargetRegisterInfo::regclass_iterator I = tri_->regclass_begin(),
         E = tri_->regclass_end(); I != E; ++I)
    allocatableRCRegs_.insert(std::make_pair(*I,
                                             tri_->getAllocatableSet(fn, *I)));

  // Join (coalesce) intervals if requested.
  if (EnableJoining) {
    joinIntervals();
    DEBUG({
        DOUT << "********** INTERVALS POST JOINING **********\n";
        for (LiveIntervals::iterator I = li_->begin(), E = li_->end(); I != E; ++I){
          I->second->print(DOUT, tri_);
          DOUT << "\n";
        }
      });
  }

  // Perform a final pass over the instructions and compute spill weights
  // and remove identity moves.
  SmallVector<unsigned, 4> DeadDefs;
  for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
       mbbi != mbbe; ++mbbi) {
    MachineBasicBlock* mbb = mbbi;
    unsigned loopDepth = loopInfo->getLoopDepth(mbb);

    for (MachineBasicBlock::iterator mii = mbb->begin(), mie = mbb->end();
         mii != mie; ) {
      MachineInstr *MI = mii;
      unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
      if (JoinedCopies.count(MI)) {
        // Delete all coalesced copies.
        if (!tii_->isMoveInstr(*MI, SrcReg, DstReg, SrcSubIdx, DstSubIdx)) {
          assert((MI->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG ||
                  MI->getOpcode() == TargetInstrInfo::INSERT_SUBREG ||
                  MI->getOpcode() == TargetInstrInfo::SUBREG_TO_REG) &&
                 "Unrecognized copy instruction");
          DstReg = MI->getOperand(0).getReg();
        }
        if (MI->registerDefIsDead(DstReg)) {
          LiveInterval &li = li_->getInterval(DstReg);
          if (!ShortenDeadCopySrcLiveRange(li, MI))
            ShortenDeadCopyLiveRange(li, MI);
        }
        li_->RemoveMachineInstrFromMaps(MI);
        mii = mbbi->erase(mii);
        ++numPeep;
        continue;
      }

      // Now check if this is a remat'ed def instruction which is now dead.
      if (ReMatDefs.count(MI)) {
        bool isDead = true;
        for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
          const MachineOperand &MO = MI->getOperand(i);
          if (!MO.isReg())
            continue;
          unsigned Reg = MO.getReg();
          if (!Reg)
            continue;
          if (TargetRegisterInfo::isVirtualRegister(Reg))
            DeadDefs.push_back(Reg);
          if (MO.isDead())
            continue;
          if (TargetRegisterInfo::isPhysicalRegister(Reg) ||
              !mri_->use_empty(Reg)) {
            isDead = false;
            break;
          }
        }
        if (isDead) {
          while (!DeadDefs.empty()) {
            unsigned DeadDef = DeadDefs.back();
            DeadDefs.pop_back();
            RemoveDeadDef(li_->getInterval(DeadDef), MI);
          }
          li_->RemoveMachineInstrFromMaps(mii);
          mii = mbbi->erase(mii);
          continue;
        } else
          DeadDefs.clear();
      }

      // If the move will be an identity move delete it
      bool isMove= tii_->isMoveInstr(*MI, SrcReg, DstReg, SrcSubIdx, DstSubIdx);
      if (isMove && SrcReg == DstReg) {
        if (li_->hasInterval(SrcReg)) {
          LiveInterval &RegInt = li_->getInterval(SrcReg);
          // If def of this move instruction is dead, remove its live range
          // from the dstination register's live interval.
          if (MI->registerDefIsDead(DstReg)) {
            if (!ShortenDeadCopySrcLiveRange(RegInt, MI))
              ShortenDeadCopyLiveRange(RegInt, MI);
          }
        }
        li_->RemoveMachineInstrFromMaps(MI);
        mii = mbbi->erase(mii);
        ++numPeep;
      } else {
        SmallSet<unsigned, 4> UniqueUses;
        for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
          const MachineOperand &mop = MI->getOperand(i);
          if (mop.isReg() && mop.getReg() &&
              TargetRegisterInfo::isVirtualRegister(mop.getReg())) {
            unsigned reg = mop.getReg();
            // Multiple uses of reg by the same instruction. It should not
            // contribute to spill weight again.
            if (UniqueUses.count(reg) != 0)
              continue;
            LiveInterval &RegInt = li_->getInterval(reg);
            RegInt.weight +=
              li_->getSpillWeight(mop.isDef(), mop.isUse(), loopDepth);
            UniqueUses.insert(reg);
          }
        }
        ++mii;
      }
    }
  }

  for (LiveIntervals::iterator I = li_->begin(), E = li_->end(); I != E; ++I) {
    LiveInterval &LI = *I->second;
    if (TargetRegisterInfo::isVirtualRegister(LI.reg)) {
      // If the live interval length is essentially zero, i.e. in every live
      // range the use follows def immediately, it doesn't make sense to spill
      // it and hope it will be easier to allocate for this li.
      if (isZeroLengthInterval(&LI))
        LI.weight = HUGE_VALF;
      else {
        bool isLoad = false;
        SmallVector<LiveInterval*, 4> SpillIs;
        if (li_->isReMaterializable(LI, SpillIs, isLoad)) {
          // If all of the definitions of the interval are re-materializable,
          // it is a preferred candidate for spilling. If non of the defs are
          // loads, then it's potentially very cheap to re-materialize.
          // FIXME: this gets much more complicated once we support non-trivial
          // re-materialization.
          if (isLoad)
            LI.weight *= 0.9F;
          else
            LI.weight *= 0.5F;
        }
      }

      // Slightly prefer live interval that has been assigned a preferred reg.
      std::pair<unsigned, unsigned> Hint = mri_->getRegAllocationHint(LI.reg);
      if (Hint.first || Hint.second)
        LI.weight *= 1.01F;

      // Divide the weight of the interval by its size.  This encourages 
      // spilling of intervals that are large and have few uses, and
      // discourages spilling of small intervals with many uses.
      LI.weight /= li_->getApproximateInstructionCount(LI) * InstrSlots::NUM;
    }
  }

  DEBUG(dump());
  return true;
}

/// print - Implement the dump method.
void SimpleRegisterCoalescing::print(std::ostream &O, const Module* m) const {
   li_->print(O, m);
}

RegisterCoalescer* llvm::createSimpleRegisterCoalescer() {
  return new SimpleRegisterCoalescing();
}

// Make sure that anything that uses RegisterCoalescer pulls in this file...
DEFINING_FILE_FOR(SimpleRegisterCoalescing)
