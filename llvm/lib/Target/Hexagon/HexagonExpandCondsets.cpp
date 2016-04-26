//===--- HexagonExpandCondsets.cpp ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Replace mux instructions with the corresponding legal instructions.
// It is meant to work post-SSA, but still on virtual registers. It was
// originally placed between register coalescing and machine instruction
// scheduler.
// In this place in the optimization sequence, live interval analysis had
// been performed, and the live intervals should be preserved. A large part
// of the code deals with preserving the liveness information.
//
// Liveness tracking aside, the main functionality of this pass is divided
// into two steps. The first step is to replace an instruction
//   vreg0 = C2_mux vreg0, vreg1, vreg2
// with a pair of conditional transfers
//   vreg0 = A2_tfrt vreg0, vreg1
//   vreg0 = A2_tfrf vreg0, vreg2
// It is the intention that the execution of this pass could be terminated
// after this step, and the code generated would be functionally correct.
//
// If the uses of the source values vreg1 and vreg2 are kills, and their
// definitions are predicable, then in the second step, the conditional
// transfers will then be rewritten as predicated instructions. E.g.
//   vreg0 = A2_or vreg1, vreg2
//   vreg3 = A2_tfrt vreg99, vreg0<kill>
// will be rewritten as
//   vreg3 = A2_port vreg99, vreg1, vreg2
//
// This replacement has two variants: "up" and "down". Consider this case:
//   vreg0 = A2_or vreg1, vreg2
//   ... [intervening instructions] ...
//   vreg3 = A2_tfrt vreg99, vreg0<kill>
// variant "up":
//   vreg3 = A2_port vreg99, vreg1, vreg2
//   ... [intervening instructions, vreg0->vreg3] ...
//   [deleted]
// variant "down":
//   [deleted]
//   ... [intervening instructions] ...
//   vreg3 = A2_port vreg99, vreg1, vreg2
//
// Both, one or none of these variants may be valid, and checks are made
// to rule out inapplicable variants.
//
// As an additional optimization, before either of the two steps above is
// executed, the pass attempts to coalesce the target register with one of
// the source registers, e.g. given an instruction
//   vreg3 = C2_mux vreg0, vreg1, vreg2
// vreg3 will be coalesced with either vreg1 or vreg2. If this succeeds,
// the instruction would then be (for example)
//   vreg3 = C2_mux vreg0, vreg3, vreg2
// and, under certain circumstances, this could result in only one predicated
// instruction:
//   vreg3 = A2_tfrf vreg0, vreg2
//

#define DEBUG_TYPE "expand-condsets"
#include "HexagonTargetMachine.h"

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<unsigned> OptTfrLimit("expand-condsets-tfr-limit",
  cl::init(~0U), cl::Hidden, cl::desc("Max number of mux expansions"));
static cl::opt<unsigned> OptCoaLimit("expand-condsets-coa-limit",
  cl::init(~0U), cl::Hidden, cl::desc("Max number of segment coalescings"));

namespace llvm {
  void initializeHexagonExpandCondsetsPass(PassRegistry&);
  FunctionPass *createHexagonExpandCondsets();
}

namespace {
  class HexagonExpandCondsets : public MachineFunctionPass {
  public:
    static char ID;
    HexagonExpandCondsets() :
        MachineFunctionPass(ID), HII(0), TRI(0), MRI(0),
        LIS(0), CoaLimitActive(false),
        TfrLimitActive(false), CoaCounter(0), TfrCounter(0) {
      if (OptCoaLimit.getPosition())
        CoaLimitActive = true, CoaLimit = OptCoaLimit;
      if (OptTfrLimit.getPosition())
        TfrLimitActive = true, TfrLimit = OptTfrLimit;
      initializeHexagonExpandCondsetsPass(*PassRegistry::getPassRegistry());
    }

    virtual const char *getPassName() const {
      return "Hexagon Expand Condsets";
    }
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LiveIntervals>();
      AU.addPreserved<LiveIntervals>();
      AU.addPreserved<SlotIndexes>();
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }
    virtual bool runOnMachineFunction(MachineFunction &MF);

  private:
    const HexagonInstrInfo *HII;
    const TargetRegisterInfo *TRI;
    MachineRegisterInfo *MRI;
    LiveIntervals *LIS;

    bool CoaLimitActive, TfrLimitActive;
    unsigned CoaLimit, TfrLimit, CoaCounter, TfrCounter;

    struct RegisterRef {
      RegisterRef(const MachineOperand &Op) : Reg(Op.getReg()),
          Sub(Op.getSubReg()) {}
      RegisterRef(unsigned R = 0, unsigned S = 0) : Reg(R), Sub(S) {}
      bool operator== (RegisterRef RR) const {
        return Reg == RR.Reg && Sub == RR.Sub;
      }
      bool operator!= (RegisterRef RR) const { return !operator==(RR); }
      unsigned Reg, Sub;
    };

    typedef DenseMap<unsigned,unsigned> ReferenceMap;
    enum { Sub_Low = 0x1, Sub_High = 0x2, Sub_None = (Sub_Low | Sub_High) };
    enum { Exec_Then = 0x10, Exec_Else = 0x20 };
    unsigned getMaskForSub(unsigned Sub);
    bool isCondset(const MachineInstr *MI);

    void addRefToMap(RegisterRef RR, ReferenceMap &Map, unsigned Exec);
    bool isRefInMap(RegisterRef, ReferenceMap &Map, unsigned Exec);

    LiveInterval::iterator nextSegment(LiveInterval &LI, SlotIndex S);
    LiveInterval::iterator prevSegment(LiveInterval &LI, SlotIndex S);
    void makeDefined(unsigned Reg, SlotIndex S, bool SetDef);
    void makeUndead(unsigned Reg, SlotIndex S);
    void shrinkToUses(unsigned Reg, LiveInterval &LI);
    void updateKillFlags(unsigned Reg, LiveInterval &LI);
    void terminateSegment(LiveInterval::iterator LT, SlotIndex S,
        LiveInterval &LI);
    void addInstrToLiveness(MachineInstr *MI);
    void removeInstrFromLiveness(MachineInstr *MI);

    unsigned getCondTfrOpcode(const MachineOperand &SO, bool Cond);
    MachineInstr *genTfrFor(MachineOperand &SrcOp, unsigned DstR,
        unsigned DstSR, const MachineOperand &PredOp, bool Cond);
    bool split(MachineInstr *MI);
    bool splitInBlock(MachineBasicBlock &B);

    bool isPredicable(MachineInstr *MI);
    MachineInstr *getReachingDefForPred(RegisterRef RD,
        MachineBasicBlock::iterator UseIt, unsigned PredR, bool Cond);
    bool canMoveOver(MachineInstr *MI, ReferenceMap &Defs, ReferenceMap &Uses);
    bool canMoveMemTo(MachineInstr *MI, MachineInstr *ToI, bool IsDown);
    void predicateAt(RegisterRef RD, MachineInstr *MI,
        MachineBasicBlock::iterator Where, unsigned PredR, bool Cond,
        bool PredUndef);
    void renameInRange(RegisterRef RO, RegisterRef RN, unsigned PredR,
        bool Cond, MachineBasicBlock::iterator First,
        MachineBasicBlock::iterator Last);
    bool predicate(MachineInstr *TfrI, bool Cond);
    bool predicateInBlock(MachineBasicBlock &B);

    void postprocessUndefImplicitUses(MachineBasicBlock &B);
    void removeImplicitUses(MachineInstr *MI);
    void removeImplicitUses(MachineBasicBlock &B);

    bool isIntReg(RegisterRef RR, unsigned &BW);
    bool isIntraBlocks(LiveInterval &LI);
    bool coalesceRegisters(RegisterRef R1, RegisterRef R2);
    bool coalesceSegments(MachineFunction &MF);
  };
}

char HexagonExpandCondsets::ID = 0;


unsigned HexagonExpandCondsets::getMaskForSub(unsigned Sub) {
  switch (Sub) {
    case Hexagon::subreg_loreg:
      return Sub_Low;
    case Hexagon::subreg_hireg:
      return Sub_High;
    case Hexagon::NoSubRegister:
      return Sub_None;
  }
  llvm_unreachable("Invalid subregister");
}


bool HexagonExpandCondsets::isCondset(const MachineInstr *MI) {
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
    case Hexagon::C2_mux:
    case Hexagon::C2_muxii:
    case Hexagon::C2_muxir:
    case Hexagon::C2_muxri:
    case Hexagon::MUX64_rr:
        return true;
      break;
  }
  return false;
}


void HexagonExpandCondsets::addRefToMap(RegisterRef RR, ReferenceMap &Map,
      unsigned Exec) {
  unsigned Mask = getMaskForSub(RR.Sub) | Exec;
  ReferenceMap::iterator F = Map.find(RR.Reg);
  if (F == Map.end())
    Map.insert(std::make_pair(RR.Reg, Mask));
  else
    F->second |= Mask;
}


bool HexagonExpandCondsets::isRefInMap(RegisterRef RR, ReferenceMap &Map,
      unsigned Exec) {
  ReferenceMap::iterator F = Map.find(RR.Reg);
  if (F == Map.end())
    return false;
  unsigned Mask = getMaskForSub(RR.Sub) | Exec;
  if (Mask & F->second)
    return true;
  return false;
}


LiveInterval::iterator HexagonExpandCondsets::nextSegment(LiveInterval &LI,
      SlotIndex S) {
  for (LiveInterval::iterator I = LI.begin(), E = LI.end(); I != E; ++I) {
    if (I->start >= S)
      return I;
  }
  return LI.end();
}


LiveInterval::iterator HexagonExpandCondsets::prevSegment(LiveInterval &LI,
      SlotIndex S) {
  LiveInterval::iterator P = LI.end();
  for (LiveInterval::iterator I = LI.begin(), E = LI.end(); I != E; ++I) {
    if (I->end > S)
      return P;
    P = I;
  }
  return P;
}


/// Find the implicit use of register Reg in slot index S, and make sure
/// that the "defined" flag is set to SetDef. While the mux expansion is
/// going on, predicated instructions will have implicit uses of the
/// registers that are being defined. This is to keep any preceding
/// definitions live. If there is no preceding definition, the implicit
/// use will be marked as "undef", otherwise it will be "defined". This
/// function is used to update the flag.
void HexagonExpandCondsets::makeDefined(unsigned Reg, SlotIndex S,
      bool SetDef) {
  if (!S.isRegister())
    return;
  MachineInstr *MI = LIS->getInstructionFromIndex(S);
  assert(MI && "Expecting instruction");
  for (auto &Op : MI->operands()) {
    if (!Op.isReg() || !Op.isUse() || Op.getReg() != Reg)
      continue;
    bool IsDef = !Op.isUndef();
    if (Op.isImplicit() && IsDef != SetDef)
      Op.setIsUndef(!SetDef);
  }
}


void HexagonExpandCondsets::makeUndead(unsigned Reg, SlotIndex S) {
  // If S is a block boundary, then there can still be a dead def reaching
  // this point. Instead of traversing the CFG, queue start points of all
  // live segments that begin with a register, and end at a block boundary.
  // This may "resurrect" some truly dead definitions, but doing so is
  // harmless.
  SmallVector<MachineInstr*,8> Defs;
  if (S.isBlock()) {
    LiveInterval &LI = LIS->getInterval(Reg);
    for (LiveInterval::iterator I = LI.begin(), E = LI.end(); I != E; ++I) {
      if (!I->start.isRegister() || !I->end.isBlock())
        continue;
      MachineInstr *MI = LIS->getInstructionFromIndex(I->start);
      Defs.push_back(MI);
    }
  } else if (S.isRegister()) {
    MachineInstr *MI = LIS->getInstructionFromIndex(S);
    Defs.push_back(MI);
  }

  for (unsigned i = 0, n = Defs.size(); i < n; ++i) {
    MachineInstr *MI = Defs[i];
    for (auto &Op : MI->operands()) {
      if (!Op.isReg() || !Op.isDef() || Op.getReg() != Reg)
        continue;
      Op.setIsDead(false);
    }
  }
}


/// Shrink the segments in the live interval for a given register to the last
/// use before each subsequent def. Unlike LiveIntervals::shrinkToUses, this
/// function will not mark any definitions of Reg as dead. The reason for this
/// is that this function is used while a MUX instruction is being expanded,
/// or while a conditional copy is undergoing predication. During these
/// processes, there may be defs present in the instruction sequence that have
/// not yet been removed, or there may be missing uses that have not yet been
/// added. We want to utilize LiveIntervals::shrinkToUses as much as possible,
/// but since it does not extend any intervals that are too short, we need to
/// pre-emptively extend them here in anticipation of further changes.
void HexagonExpandCondsets::shrinkToUses(unsigned Reg, LiveInterval &LI) {
  SmallVector<MachineInstr*,4> Deads;
  LIS->shrinkToUses(&LI, &Deads);
  // Need to undo the deadification made by "shrinkToUses". It's easier to
  // do it here, since we have a list of all instructions that were just
  // marked as dead.
  for (unsigned i = 0, n = Deads.size(); i < n; ++i) {
    MachineInstr *MI = Deads[i];
    // Clear the "dead" flag.
    for (auto &Op : MI->operands()) {
      if (!Op.isReg() || !Op.isDef() || Op.getReg() != Reg)
        continue;
      Op.setIsDead(false);
    }
    // Extend the live segment to the beginning of the next one.
    LiveInterval::iterator End = LI.end();
    SlotIndex S = LIS->getInstructionIndex(*MI).getRegSlot();
    LiveInterval::iterator T = LI.FindSegmentContaining(S);
    assert(T != End);
    LiveInterval::iterator N = std::next(T);
    if (N != End)
      T->end = N->start;
    else
      T->end = LIS->getMBBEndIdx(MI->getParent());
  }
  updateKillFlags(Reg, LI);
}


/// Given an updated live interval LI for register Reg, update the kill flags
/// in instructions using Reg to reflect the liveness changes.
void HexagonExpandCondsets::updateKillFlags(unsigned Reg, LiveInterval &LI) {
  MRI->clearKillFlags(Reg);
  for (LiveInterval::iterator I = LI.begin(), E = LI.end(); I != E; ++I) {
    SlotIndex EX = I->end;
    if (!EX.isRegister())
      continue;
    MachineInstr *MI = LIS->getInstructionFromIndex(EX);
    for (auto &Op : MI->operands()) {
      if (!Op.isReg() || !Op.isUse() || Op.getReg() != Reg)
        continue;
      // Only set the kill flag on the first encountered use of Reg in this
      // instruction.
      Op.setIsKill(true);
      break;
    }
  }
}


/// When adding a new instruction to liveness, the newly added definition
/// will start a new live segment. This may happen at a position that falls
/// within an existing live segment. In such case that live segment needs to
/// be truncated to make room for the new segment. Ultimately, the truncation
/// will occur at the last use, but for now the segment can be terminated
/// right at the place where the new segment will start. The segments will be
/// shrunk-to-uses later.
void HexagonExpandCondsets::terminateSegment(LiveInterval::iterator LT,
      SlotIndex S, LiveInterval &LI) {
  // Terminate the live segment pointed to by LT within a live interval LI.
  if (LT == LI.end())
    return;

  VNInfo *OldVN = LT->valno;
  SlotIndex EX = LT->end;
  LT->end = S;
  // If LT does not end at a block boundary, the termination is done.
  if (!EX.isBlock())
    return;

  // If LT ended at a block boundary, it's possible that its value number
  // is picked up at the beginning other blocks. Create a new value number
  // and change such blocks to use it instead.
  VNInfo *NewVN = 0;
  for (LiveInterval::iterator I = LI.begin(), E = LI.end(); I != E; ++I) {
    if (!I->start.isBlock() || I->valno != OldVN)
      continue;
    // Generate on-demand a new value number that is defined by the
    // block beginning (i.e. -phi).
    if (!NewVN)
      NewVN = LI.getNextValue(I->start, LIS->getVNInfoAllocator());
    I->valno = NewVN;
  }
}


/// Add the specified instruction to live intervals. This function is used
/// to update the live intervals while the program code is being changed.
/// Neither the expansion of a MUX, nor the predication are atomic, and this
/// function is used to update the live intervals while these transformations
/// are being done.
void HexagonExpandCondsets::addInstrToLiveness(MachineInstr *MI) {
  SlotIndex MX = LIS->isNotInMIMap(*MI) ? LIS->InsertMachineInstrInMaps(*MI)
                                        : LIS->getInstructionIndex(*MI);
  DEBUG(dbgs() << "adding liveness info for instr\n  " << MX << "  " << *MI);

  MX = MX.getRegSlot();
  bool Predicated = HII->isPredicated(*MI);
  MachineBasicBlock *MB = MI->getParent();

  // Strip all implicit uses from predicated instructions. They will be
  // added again, according to the updated information.
  if (Predicated)
    removeImplicitUses(MI);

  // For each def in MI we need to insert a new live segment starting at MX
  // into the interval. If there already exists a live segment in the interval
  // that contains MX, we need to terminate it at MX.
  SmallVector<RegisterRef,2> Defs;
  for (auto &Op : MI->operands())
    if (Op.isReg() && Op.isDef())
      Defs.push_back(RegisterRef(Op));

  for (unsigned i = 0, n = Defs.size(); i < n; ++i) {
    unsigned DefR = Defs[i].Reg;
    LiveInterval &LID = LIS->getInterval(DefR);
    DEBUG(dbgs() << "adding def " << PrintReg(DefR, TRI)
                 << " with interval\n  " << LID << "\n");
    // If MX falls inside of an existing live segment, terminate it.
    LiveInterval::iterator LT = LID.FindSegmentContaining(MX);
    if (LT != LID.end())
      terminateSegment(LT, MX, LID);
    DEBUG(dbgs() << "after terminating segment\n  " << LID << "\n");

    // Create a new segment starting from MX.
    LiveInterval::iterator P = prevSegment(LID, MX), N = nextSegment(LID, MX);
    SlotIndex EX;
    VNInfo *VN = LID.getNextValue(MX, LIS->getVNInfoAllocator());
    if (N == LID.end()) {
      // There is no live segment after MX. End this segment at the end of
      // the block.
      EX = LIS->getMBBEndIdx(MB);
    } else {
      // If the next segment starts at the block boundary, end the new segment
      // at the boundary of the preceding block (i.e. the previous index).
      // Otherwise, end the segment at the beginning of the next segment. In
      // either case it will be "shrunk-to-uses" later.
      EX = N->start.isBlock() ? N->start.getPrevIndex() : N->start;
    }
    if (Predicated) {
      // Predicated instruction will have an implicit use of the defined
      // register. This is necessary so that this definition will not make
      // any previous definitions dead. If there are no previous live
      // segments, still add the implicit use, but make it "undef".
      // Because of the implicit use, the preceding definition is not
      // dead. Mark is as such (if necessary).
      MachineOperand ImpUse = MachineOperand::CreateReg(DefR, false, true);
      ImpUse.setSubReg(Defs[i].Sub);
      bool Undef = false;
      if (P == LID.end())
        Undef = true;
      else {
        // If the previous segment extends to the end of the previous block,
        // the end index may actually be the beginning of this block. If
        // the previous segment ends at a block boundary, move it back by one,
        // to get the proper block for it.
        SlotIndex PE = P->end.isBlock() ? P->end.getPrevIndex() : P->end;
        MachineBasicBlock *PB = LIS->getMBBFromIndex(PE);
        if (PB != MB && !LIS->isLiveInToMBB(LID, MB))
          Undef = true;
      }
      if (!Undef) {
        makeUndead(DefR, P->valno->def);
        // We are adding a live use, so extend the previous segment to
        // include it.
        P->end = MX;
      } else {
        ImpUse.setIsUndef(true);
      }

      if (!MI->readsRegister(DefR))
        MI->addOperand(ImpUse);
      if (N != LID.end())
        makeDefined(DefR, N->start, true);
    }
    LiveRange::Segment NR = LiveRange::Segment(MX, EX, VN);
    LID.addSegment(NR);
    DEBUG(dbgs() << "added a new segment " << NR << "\n  " << LID << "\n");
    shrinkToUses(DefR, LID);
    DEBUG(dbgs() << "updated imp-uses: " << *MI);
    LID.verify();
  }

  // For each use in MI:
  // - If there is no live segment that contains MX for the used register,
  //   extend the previous one. Ignore implicit uses.
  for (auto &Op : MI->operands()) {
    if (!Op.isReg() || !Op.isUse() || Op.isImplicit() || Op.isUndef())
      continue;
    unsigned UseR = Op.getReg();
    LiveInterval &LIU = LIS->getInterval(UseR);
    // Find the last segment P that starts before MX.
    LiveInterval::iterator P = LIU.FindSegmentContaining(MX);
    if (P == LIU.end())
      P = prevSegment(LIU, MX);

    assert(P != LIU.end() && "MI uses undefined register?");
    SlotIndex EX = P->end;
    // If P contains MX, there is not much to do.
    if (EX > MX) {
      Op.setIsKill(false);
      continue;
    }
    // Otherwise, extend P to "next(MX)".
    P->end = MX.getNextIndex();
    Op.setIsKill(true);
    // Get the old "kill" instruction, and remove the kill flag.
    if (MachineInstr *KI = LIS->getInstructionFromIndex(MX))
      KI->clearRegisterKills(UseR, nullptr);
    shrinkToUses(UseR, LIU);
    LIU.verify();
  }
}


/// Update the live interval information to reflect the removal of the given
/// instruction from the program. As with "addInstrToLiveness", this function
/// is called while the program code is being changed.
void HexagonExpandCondsets::removeInstrFromLiveness(MachineInstr *MI) {
  SlotIndex MX = LIS->getInstructionIndex(*MI).getRegSlot();
  DEBUG(dbgs() << "removing instr\n  " << MX << "  " << *MI);

  // For each def in MI:
  // If MI starts a live segment, merge this segment with the previous segment.
  //
  for (auto &Op : MI->operands()) {
    if (!Op.isReg() || !Op.isDef())
      continue;
    unsigned DefR = Op.getReg();
    LiveInterval &LID = LIS->getInterval(DefR);
    LiveInterval::iterator LT = LID.FindSegmentContaining(MX);
    assert(LT != LID.end() && "Expecting live segments");
    DEBUG(dbgs() << "removing def at " << MX << " of " << PrintReg(DefR, TRI)
                 << " with interval\n  " << LID << "\n");
    if (LT->start != MX)
      continue;

    VNInfo *MVN = LT->valno;
    if (LT != LID.begin()) {
      // If the current live segment is not the first, the task is easy. If
      // the previous segment continues into the current block, extend it to
      // the end of the current one, and merge the value numbers.
      // Otherwise, remove the current segment, and make the end of it "undef".
      LiveInterval::iterator P = std::prev(LT);
      SlotIndex PE = P->end.isBlock() ? P->end.getPrevIndex() : P->end;
      MachineBasicBlock *MB = MI->getParent();
      MachineBasicBlock *PB = LIS->getMBBFromIndex(PE);
      if (PB != MB && !LIS->isLiveInToMBB(LID, MB)) {
        makeDefined(DefR, LT->end, false);
        LID.removeSegment(*LT);
      } else {
        // Make the segments adjacent, so that merge-vn can also merge the
        // segments.
        P->end = LT->start;
        makeUndead(DefR, P->valno->def);
        LID.MergeValueNumberInto(MVN, P->valno);
      }
    } else {
      LiveInterval::iterator N = std::next(LT);
      LiveInterval::iterator RmB = LT, RmE = N;
      while (N != LID.end()) {
        // Iterate until the first register-based definition is found
        // (i.e. skip all block-boundary entries).
        LiveInterval::iterator Next = std::next(N);
        if (N->start.isRegister()) {
          makeDefined(DefR, N->start, false);
          break;
        }
        if (N->end.isRegister()) {
          makeDefined(DefR, N->end, false);
          RmE = Next;
          break;
        }
        RmE = Next;
        N = Next;
      }
      // Erase the segments in one shot to avoid invalidating iterators.
      LID.segments.erase(RmB, RmE);
    }

    bool VNUsed = false;
    for (LiveInterval::iterator I = LID.begin(), E = LID.end(); I != E; ++I) {
      if (I->valno != MVN)
        continue;
      VNUsed = true;
      break;
    }
    if (!VNUsed)
      MVN->markUnused();

    DEBUG(dbgs() << "new interval: ");
    if (!LID.empty()) {
      DEBUG(dbgs() << LID << "\n");
      LID.verify();
    } else {
      DEBUG(dbgs() << "<empty>\n");
      LIS->removeInterval(DefR);
    }
  }

  // For uses there is nothing to do. The intervals will be updated via
  // shrinkToUses.
  SmallVector<unsigned,4> Uses;
  for (auto &Op : MI->operands()) {
    if (!Op.isReg() || !Op.isUse())
      continue;
    unsigned R = Op.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(R))
      continue;
    Uses.push_back(R);
  }
  LIS->RemoveMachineInstrFromMaps(*MI);
  MI->eraseFromParent();
  for (unsigned i = 0, n = Uses.size(); i < n; ++i) {
    LiveInterval &LI = LIS->getInterval(Uses[i]);
    shrinkToUses(Uses[i], LI);
  }
}


/// Get the opcode for a conditional transfer of the value in SO (source
/// operand). The condition (true/false) is given in Cond.
unsigned HexagonExpandCondsets::getCondTfrOpcode(const MachineOperand &SO,
      bool Cond) {
  using namespace Hexagon;
  if (SO.isReg()) {
    unsigned PhysR;
    RegisterRef RS = SO;
    if (TargetRegisterInfo::isVirtualRegister(RS.Reg)) {
      const TargetRegisterClass *VC = MRI->getRegClass(RS.Reg);
      assert(VC->begin() != VC->end() && "Empty register class");
      PhysR = *VC->begin();
    } else {
      assert(TargetRegisterInfo::isPhysicalRegister(RS.Reg));
      PhysR = RS.Reg;
    }
    unsigned PhysS = (RS.Sub == 0) ? PhysR : TRI->getSubReg(PhysR, RS.Sub);
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(PhysS);
    switch (RC->getSize()) {
      case 4:
        return Cond ? A2_tfrt : A2_tfrf;
      case 8:
        return Cond ? A2_tfrpt : A2_tfrpf;
    }
    llvm_unreachable("Invalid register operand");
  }
  if (SO.isImm() || SO.isFPImm())
    return Cond ? C2_cmoveit : C2_cmoveif;
  llvm_unreachable("Unexpected source operand");
}


/// Generate a conditional transfer, copying the value SrcOp to the
/// destination register DstR:DstSR, and using the predicate register from
/// PredOp. The Cond argument specifies whether the predicate is to be
/// if(PredOp), or if(!PredOp).
MachineInstr *HexagonExpandCondsets::genTfrFor(MachineOperand &SrcOp,
      unsigned DstR, unsigned DstSR, const MachineOperand &PredOp, bool Cond) {
  MachineInstr *MI = SrcOp.getParent();
  MachineBasicBlock &B = *MI->getParent();
  MachineBasicBlock::iterator At = MI;
  DebugLoc DL = MI->getDebugLoc();

  // Don't avoid identity copies here (i.e. if the source and the destination
  // are the same registers). It is actually better to generate them here,
  // since this would cause the copy to potentially be predicated in the next
  // step. The predication will remove such a copy if it is unable to
  /// predicate.

  unsigned Opc = getCondTfrOpcode(SrcOp, Cond);
  MachineInstr *TfrI = BuildMI(B, At, DL, HII->get(Opc))
        .addReg(DstR, RegState::Define, DstSR)
        .addOperand(PredOp)
        .addOperand(SrcOp);
  // We don't want any kills yet.
  TfrI->clearKillInfo();
  DEBUG(dbgs() << "created an initial copy: " << *TfrI);
  return TfrI;
}


/// Replace a MUX instruction MI with a pair A2_tfrt/A2_tfrf. This function
/// performs all necessary changes to complete the replacement.
bool HexagonExpandCondsets::split(MachineInstr *MI) {
  if (TfrLimitActive) {
    if (TfrCounter >= TfrLimit)
      return false;
    TfrCounter++;
  }
  DEBUG(dbgs() << "\nsplitting BB#" << MI->getParent()->getNumber()
               << ": " << *MI);
  MachineOperand &MD = MI->getOperand(0); // Definition
  MachineOperand &MP = MI->getOperand(1); // Predicate register
  assert(MD.isDef());
  unsigned DR = MD.getReg(), DSR = MD.getSubReg();

  // First, create the two invididual conditional transfers, and add each
  // of them to the live intervals information. Do that first and then remove
  // the old instruction from live intervals.
  if (MachineInstr *TfrT = genTfrFor(MI->getOperand(2), DR, DSR, MP, true))
    addInstrToLiveness(TfrT);
  if (MachineInstr *TfrF = genTfrFor(MI->getOperand(3), DR, DSR, MP, false))
    addInstrToLiveness(TfrF);
  removeInstrFromLiveness(MI);

  return true;
}


/// Split all MUX instructions in the given block into pairs of contitional
/// transfers.
bool HexagonExpandCondsets::splitInBlock(MachineBasicBlock &B) {
  bool Changed = false;
  MachineBasicBlock::iterator I, E, NextI;
  for (I = B.begin(), E = B.end(); I != E; I = NextI) {
    NextI = std::next(I);
    if (isCondset(I))
      Changed |= split(I);
  }
  return Changed;
}


bool HexagonExpandCondsets::isPredicable(MachineInstr *MI) {
  if (HII->isPredicated(*MI) || !HII->isPredicable(*MI))
    return false;
  if (MI->hasUnmodeledSideEffects() || MI->mayStore())
    return false;
  // Reject instructions with multiple defs (e.g. post-increment loads).
  bool HasDef = false;
  for (auto &Op : MI->operands()) {
    if (!Op.isReg() || !Op.isDef())
      continue;
    if (HasDef)
      return false;
    HasDef = true;
  }
  for (auto &Mo : MI->memoperands())
    if (Mo->isVolatile())
      return false;
  return true;
}


/// Find the reaching definition for a predicated use of RD. The RD is used
/// under the conditions given by PredR and Cond, and this function will ignore
/// definitions that set RD under the opposite conditions.
MachineInstr *HexagonExpandCondsets::getReachingDefForPred(RegisterRef RD,
      MachineBasicBlock::iterator UseIt, unsigned PredR, bool Cond) {
  MachineBasicBlock &B = *UseIt->getParent();
  MachineBasicBlock::iterator I = UseIt, S = B.begin();
  if (I == S)
    return 0;

  bool PredValid = true;
  do {
    --I;
    MachineInstr *MI = &*I;
    // Check if this instruction can be ignored, i.e. if it is predicated
    // on the complementary condition.
    if (PredValid && HII->isPredicated(*MI)) {
      if (MI->readsRegister(PredR) && (Cond != HII->isPredicatedTrue(*MI)))
        continue;
    }

    // Check the defs. If the PredR is defined, invalidate it. If RD is
    // defined, return the instruction or 0, depending on the circumstances.
    for (auto &Op : MI->operands()) {
      if (!Op.isReg() || !Op.isDef())
        continue;
      RegisterRef RR = Op;
      if (RR.Reg == PredR) {
        PredValid = false;
        continue;
      }
      if (RR.Reg != RD.Reg)
        continue;
      // If the "Reg" part agrees, there is still the subregister to check.
      // If we are looking for vreg1:loreg, we can skip vreg1:hireg, but
      // not vreg1 (w/o subregisters).
      if (RR.Sub == RD.Sub)
        return MI;
      if (RR.Sub == 0 || RD.Sub == 0)
        return 0;
      // We have different subregisters, so we can continue looking.
    }
  } while (I != S);

  return 0;
}


/// Check if the instruction MI can be safely moved over a set of instructions
/// whose side-effects (in terms of register defs and uses) are expressed in
/// the maps Defs and Uses. These maps reflect the conditional defs and uses
/// that depend on the same predicate register to allow moving instructions
/// over instructions predicated on the opposite condition.
bool HexagonExpandCondsets::canMoveOver(MachineInstr *MI, ReferenceMap &Defs,
      ReferenceMap &Uses) {
  // In order to be able to safely move MI over instructions that define
  // "Defs" and use "Uses", no def operand from MI can be defined or used
  // and no use operand can be defined.
  for (auto &Op : MI->operands()) {
    if (!Op.isReg())
      continue;
    RegisterRef RR = Op;
    // For physical register we would need to check register aliases, etc.
    // and we don't want to bother with that. It would be of little value
    // before the actual register rewriting (from virtual to physical).
    if (!TargetRegisterInfo::isVirtualRegister(RR.Reg))
      return false;
    // No redefs for any operand.
    if (isRefInMap(RR, Defs, Exec_Then))
      return false;
    // For defs, there cannot be uses.
    if (Op.isDef() && isRefInMap(RR, Uses, Exec_Then))
      return false;
  }
  return true;
}


/// Check if the instruction accessing memory (TheI) can be moved to the
/// location ToI.
bool HexagonExpandCondsets::canMoveMemTo(MachineInstr *TheI, MachineInstr *ToI,
      bool IsDown) {
  bool IsLoad = TheI->mayLoad(), IsStore = TheI->mayStore();
  if (!IsLoad && !IsStore)
    return true;
  if (HII->areMemAccessesTriviallyDisjoint(TheI, ToI))
    return true;
  if (TheI->hasUnmodeledSideEffects())
    return false;

  MachineBasicBlock::iterator StartI = IsDown ? TheI : ToI;
  MachineBasicBlock::iterator EndI = IsDown ? ToI : TheI;
  bool Ordered = TheI->hasOrderedMemoryRef();

  // Search for aliased memory reference in (StartI, EndI).
  for (MachineBasicBlock::iterator I = std::next(StartI); I != EndI; ++I) {
    MachineInstr *MI = &*I;
    if (MI->hasUnmodeledSideEffects())
      return false;
    bool L = MI->mayLoad(), S = MI->mayStore();
    if (!L && !S)
      continue;
    if (Ordered && MI->hasOrderedMemoryRef())
      return false;

    bool Conflict = (L && IsStore) || S;
    if (Conflict)
      return false;
  }
  return true;
}


/// Generate a predicated version of MI (where the condition is given via
/// PredR and Cond) at the point indicated by Where.
void HexagonExpandCondsets::predicateAt(RegisterRef RD, MachineInstr *MI,
      MachineBasicBlock::iterator Where, unsigned PredR, bool Cond,
      bool PredUndef) {
  // The problem with updating live intervals is that we can move one def
  // past another def. In particular, this can happen when moving an A2_tfrt
  // over an A2_tfrf defining the same register. From the point of view of
  // live intervals, these two instructions are two separate definitions,
  // and each one starts another live segment. LiveIntervals's "handleMove"
  // does not allow such moves, so we need to handle it ourselves. To avoid
  // invalidating liveness data while we are using it, the move will be
  // implemented in 4 steps: (1) add a clone of the instruction MI at the
  // target location, (2) update liveness, (3) delete the old instruction,
  // and (4) update liveness again.

  MachineBasicBlock &B = *MI->getParent();
  DebugLoc DL = Where->getDebugLoc();  // "Where" points to an instruction.
  unsigned Opc = MI->getOpcode();
  unsigned PredOpc = HII->getCondOpcode(Opc, !Cond);
  MachineInstrBuilder MB = BuildMI(B, Where, DL, HII->get(PredOpc));
  unsigned Ox = 0, NP = MI->getNumOperands();
  // Skip all defs from MI first.
  while (Ox < NP) {
    MachineOperand &MO = MI->getOperand(Ox);
    if (!MO.isReg() || !MO.isDef())
      break;
    Ox++;
  }
  // Add the new def, then the predicate register, then the rest of the
  // operands.
  MB.addReg(RD.Reg, RegState::Define, RD.Sub);
  MB.addReg(PredR, PredUndef ? RegState::Undef : 0);
  while (Ox < NP) {
    MachineOperand &MO = MI->getOperand(Ox);
    if (!MO.isReg() || !MO.isImplicit())
      MB.addOperand(MO);
    Ox++;
  }

  MachineFunction &MF = *B.getParent();
  MachineInstr::mmo_iterator I = MI->memoperands_begin();
  unsigned NR = std::distance(I, MI->memoperands_end());
  MachineInstr::mmo_iterator MemRefs = MF.allocateMemRefsArray(NR);
  for (unsigned i = 0; i < NR; ++i)
    MemRefs[i] = *I++;
  MB.setMemRefs(MemRefs, MemRefs+NR);

  MachineInstr *NewI = MB;
  NewI->clearKillInfo();
  addInstrToLiveness(NewI);
}


/// In the range [First, Last], rename all references to the "old" register RO
/// to the "new" register RN, but only in instructions predicated on the given
/// condition.
void HexagonExpandCondsets::renameInRange(RegisterRef RO, RegisterRef RN,
      unsigned PredR, bool Cond, MachineBasicBlock::iterator First,
      MachineBasicBlock::iterator Last) {
  MachineBasicBlock::iterator End = std::next(Last);
  for (MachineBasicBlock::iterator I = First; I != End; ++I) {
    MachineInstr *MI = &*I;
    // Do not touch instructions that are not predicated, or are predicated
    // on the opposite condition.
    if (!HII->isPredicated(*MI))
      continue;
    if (!MI->readsRegister(PredR) || (Cond != HII->isPredicatedTrue(*MI)))
      continue;

    for (auto &Op : MI->operands()) {
      if (!Op.isReg() || RO != RegisterRef(Op))
        continue;
      Op.setReg(RN.Reg);
      Op.setSubReg(RN.Sub);
      // In practice, this isn't supposed to see any defs.
      assert(!Op.isDef() && "Not expecting a def");
    }
  }
}


/// For a given conditional copy, predicate the definition of the source of
/// the copy under the given condition (using the same predicate register as
/// the copy).
bool HexagonExpandCondsets::predicate(MachineInstr *TfrI, bool Cond) {
  // TfrI - A2_tfr[tf] Instruction (not A2_tfrsi).
  unsigned Opc = TfrI->getOpcode();
  (void)Opc;
  assert(Opc == Hexagon::A2_tfrt || Opc == Hexagon::A2_tfrf);
  DEBUG(dbgs() << "\nattempt to predicate if-" << (Cond ? "true" : "false")
               << ": " << *TfrI);

  MachineOperand &MD = TfrI->getOperand(0);
  MachineOperand &MP = TfrI->getOperand(1);
  MachineOperand &MS = TfrI->getOperand(2);
  // The source operand should be a <kill>. This is not strictly necessary,
  // but it makes things a lot simpler. Otherwise, we would need to rename
  // some registers, which would complicate the transformation considerably.
  if (!MS.isKill())
    return false;

  RegisterRef RT(MS);
  unsigned PredR = MP.getReg();
  MachineInstr *DefI = getReachingDefForPred(RT, TfrI, PredR, Cond);
  if (!DefI || !isPredicable(DefI))
    return false;

  DEBUG(dbgs() << "Source def: " << *DefI);

  // Collect the information about registers defined and used between the
  // DefI and the TfrI.
  // Map: reg -> bitmask of subregs
  ReferenceMap Uses, Defs;
  MachineBasicBlock::iterator DefIt = DefI, TfrIt = TfrI;

  // Check if the predicate register is valid between DefI and TfrI.
  // If it is, we can then ignore instructions predicated on the negated
  // conditions when collecting def and use information.
  bool PredValid = true;
  for (MachineBasicBlock::iterator I = std::next(DefIt); I != TfrIt; ++I) {
    if (!I->modifiesRegister(PredR, 0))
      continue;
    PredValid = false;
    break;
  }

  for (MachineBasicBlock::iterator I = std::next(DefIt); I != TfrIt; ++I) {
    MachineInstr *MI = &*I;
    // If this instruction is predicated on the same register, it could
    // potentially be ignored.
    // By default assume that the instruction executes on the same condition
    // as TfrI (Exec_Then), and also on the opposite one (Exec_Else).
    unsigned Exec = Exec_Then | Exec_Else;
    if (PredValid && HII->isPredicated(*MI) && MI->readsRegister(PredR))
      Exec = (Cond == HII->isPredicatedTrue(*MI)) ? Exec_Then : Exec_Else;

    for (auto &Op : MI->operands()) {
      if (!Op.isReg())
        continue;
      // We don't want to deal with physical registers. The reason is that
      // they can be aliased with other physical registers. Aliased virtual
      // registers must share the same register number, and can only differ
      // in the subregisters, which we are keeping track of. Physical
      // registers ters no longer have subregisters---their super- and
      // subregisters are other physical registers, and we are not checking
      // that.
      RegisterRef RR = Op;
      if (!TargetRegisterInfo::isVirtualRegister(RR.Reg))
        return false;

      ReferenceMap &Map = Op.isDef() ? Defs : Uses;
      addRefToMap(RR, Map, Exec);
    }
  }

  // The situation:
  //   RT = DefI
  //   ...
  //   RD = TfrI ..., RT

  // If the register-in-the-middle (RT) is used or redefined between
  // DefI and TfrI, we may not be able proceed with this transformation.
  // We can ignore a def that will not execute together with TfrI, and a
  // use that will. If there is such a use (that does execute together with
  // TfrI), we will not be able to move DefI down. If there is a use that
  // executed if TfrI's condition is false, then RT must be available
  // unconditionally (cannot be predicated).
  // Essentially, we need to be able to rename RT to RD in this segment.
  if (isRefInMap(RT, Defs, Exec_Then) || isRefInMap(RT, Uses, Exec_Else))
    return false;
  RegisterRef RD = MD;
  // If the predicate register is defined between DefI and TfrI, the only
  // potential thing to do would be to move the DefI down to TfrI, and then
  // predicate. The reaching def (DefI) must be movable down to the location
  // of the TfrI.
  // If the target register of the TfrI (RD) is not used or defined between
  // DefI and TfrI, consider moving TfrI up to DefI.
  bool CanUp =   canMoveOver(TfrI, Defs, Uses);
  bool CanDown = canMoveOver(DefI, Defs, Uses);
  // The TfrI does not access memory, but DefI could. Check if it's safe
  // to move DefI down to TfrI.
  if (DefI->mayLoad() || DefI->mayStore())
    if (!canMoveMemTo(DefI, TfrI, true))
      CanDown = false;

  DEBUG(dbgs() << "Can move up: " << (CanUp ? "yes" : "no")
               << ", can move down: " << (CanDown ? "yes\n" : "no\n"));
  MachineBasicBlock::iterator PastDefIt = std::next(DefIt);
  if (CanUp)
    predicateAt(RD, DefI, PastDefIt, PredR, Cond, MP.isUndef());
  else if (CanDown)
    predicateAt(RD, DefI, TfrIt, PredR, Cond, MP.isUndef());
  else
    return false;

  if (RT != RD)
    renameInRange(RT, RD, PredR, Cond, PastDefIt, TfrIt);

  // Delete the user of RT first (it should work either way, but this order
  // of deleting is more natural).
  removeInstrFromLiveness(TfrI);
  removeInstrFromLiveness(DefI);
  return true;
}


/// Predicate all cases of conditional copies in the specified block.
bool HexagonExpandCondsets::predicateInBlock(MachineBasicBlock &B) {
  bool Changed = false;
  MachineBasicBlock::iterator I, E, NextI;
  for (I = B.begin(), E = B.end(); I != E; I = NextI) {
    NextI = std::next(I);
    unsigned Opc = I->getOpcode();
    if (Opc == Hexagon::A2_tfrt || Opc == Hexagon::A2_tfrf) {
      bool Done = predicate(I, (Opc == Hexagon::A2_tfrt));
      if (!Done) {
        // If we didn't predicate I, we may need to remove it in case it is
        // an "identity" copy, e.g.  vreg1 = A2_tfrt vreg2, vreg1.
        if (RegisterRef(I->getOperand(0)) == RegisterRef(I->getOperand(2)))
          removeInstrFromLiveness(I);
      }
      Changed |= Done;
    }
  }
  return Changed;
}


void HexagonExpandCondsets::removeImplicitUses(MachineInstr *MI) {
  for (unsigned i = MI->getNumOperands(); i > 0; --i) {
    MachineOperand &MO = MI->getOperand(i-1);
    if (MO.isReg() && MO.isUse() && MO.isImplicit())
      MI->RemoveOperand(i-1);
  }
}


void HexagonExpandCondsets::removeImplicitUses(MachineBasicBlock &B) {
  for (MachineInstr &MI : B)
    if (HII->isPredicated(MI))
      removeImplicitUses(&MI);
}


void HexagonExpandCondsets::postprocessUndefImplicitUses(MachineBasicBlock &B) {
  // Implicit uses that are "undef" are only meaningful (outside of the
  // internals of this pass) when the instruction defines a subregister,
  // and the implicit-undef use applies to the defined register. In such
  // cases, the proper way to record the information in the IR is to mark
  // the definition as "undef", which will be interpreted as "read-undef".
  typedef SmallSet<unsigned,2> RegisterSet;
  for (MachineBasicBlock::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    MachineInstr *MI = &*I;
    RegisterSet Undefs;
    for (unsigned i = MI->getNumOperands(); i > 0; --i) {
      MachineOperand &MO = MI->getOperand(i-1);
      if (MO.isReg() && MO.isUse() && MO.isImplicit() && MO.isUndef()) {
        MI->RemoveOperand(i-1);
        Undefs.insert(MO.getReg());
      }
    }
    for (auto &Op : MI->operands()) {
      if (!Op.isReg() || !Op.isDef() || !Op.getSubReg())
        continue;
      if (Undefs.count(Op.getReg()))
        Op.setIsUndef(true);
    }
  }
}


bool HexagonExpandCondsets::isIntReg(RegisterRef RR, unsigned &BW) {
  if (!TargetRegisterInfo::isVirtualRegister(RR.Reg))
    return false;
  const TargetRegisterClass *RC = MRI->getRegClass(RR.Reg);
  if (RC == &Hexagon::IntRegsRegClass) {
    BW = 32;
    return true;
  }
  if (RC == &Hexagon::DoubleRegsRegClass) {
    BW = (RR.Sub != 0) ? 32 : 64;
    return true;
  }
  return false;
}


bool HexagonExpandCondsets::isIntraBlocks(LiveInterval &LI) {
  for (LiveInterval::iterator I = LI.begin(), E = LI.end(); I != E; ++I) {
    LiveRange::Segment &LR = *I;
    // Range must start at a register...
    if (!LR.start.isRegister())
      return false;
    // ...and end in a register or in a dead slot.
    if (!LR.end.isRegister() && !LR.end.isDead())
      return false;
  }
  return true;
}


bool HexagonExpandCondsets::coalesceRegisters(RegisterRef R1, RegisterRef R2) {
  if (CoaLimitActive) {
    if (CoaCounter >= CoaLimit)
      return false;
    CoaCounter++;
  }
  unsigned BW1, BW2;
  if (!isIntReg(R1, BW1) || !isIntReg(R2, BW2) || BW1 != BW2)
    return false;
  if (MRI->isLiveIn(R1.Reg))
    return false;
  if (MRI->isLiveIn(R2.Reg))
    return false;

  LiveInterval &L1 = LIS->getInterval(R1.Reg);
  LiveInterval &L2 = LIS->getInterval(R2.Reg);
  bool Overlap = L1.overlaps(L2);

  DEBUG(dbgs() << "compatible registers: ("
               << (Overlap ? "overlap" : "disjoint") << ")\n  "
               << PrintReg(R1.Reg, TRI, R1.Sub) << "  " << L1 << "\n  "
               << PrintReg(R2.Reg, TRI, R2.Sub) << "  " << L2 << "\n");
  if (R1.Sub || R2.Sub)
    return false;
  if (Overlap)
    return false;

  // Coalescing could have a negative impact on scheduling, so try to limit
  // to some reasonable extent. Only consider coalescing segments, when one
  // of them does not cross basic block boundaries.
  if (!isIntraBlocks(L1) && !isIntraBlocks(L2))
    return false;

  MRI->replaceRegWith(R2.Reg, R1.Reg);

  // Move all live segments from L2 to L1.
  typedef DenseMap<VNInfo*,VNInfo*> ValueInfoMap;
  ValueInfoMap VM;
  for (LiveInterval::iterator I = L2.begin(), E = L2.end(); I != E; ++I) {
    VNInfo *NewVN, *OldVN = I->valno;
    ValueInfoMap::iterator F = VM.find(OldVN);
    if (F == VM.end()) {
      NewVN = L1.getNextValue(I->valno->def, LIS->getVNInfoAllocator());
      VM.insert(std::make_pair(OldVN, NewVN));
    } else {
      NewVN = F->second;
    }
    L1.addSegment(LiveRange::Segment(I->start, I->end, NewVN));
  }
  while (L2.begin() != L2.end())
    L2.removeSegment(*L2.begin());

  updateKillFlags(R1.Reg, L1);
  DEBUG(dbgs() << "coalesced: " << L1 << "\n");
  L1.verify();

  return true;
}


/// Attempt to coalesce one of the source registers to a MUX intruction with
/// the destination register. This could lead to having only one predicated
/// instruction in the end instead of two.
bool HexagonExpandCondsets::coalesceSegments(MachineFunction &MF) {
  SmallVector<MachineInstr*,16> Condsets;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock &B = *I;
    for (MachineBasicBlock::iterator J = B.begin(), F = B.end(); J != F; ++J) {
      MachineInstr *MI = &*J;
      if (!isCondset(MI))
        continue;
      MachineOperand &S1 = MI->getOperand(2), &S2 = MI->getOperand(3);
      if (!S1.isReg() && !S2.isReg())
        continue;
      Condsets.push_back(MI);
    }
  }

  bool Changed = false;
  for (unsigned i = 0, n = Condsets.size(); i < n; ++i) {
    MachineInstr *CI = Condsets[i];
    RegisterRef RD = CI->getOperand(0);
    RegisterRef RP = CI->getOperand(1);
    MachineOperand &S1 = CI->getOperand(2), &S2 = CI->getOperand(3);
    bool Done = false;
    // Consider this case:
    //   vreg1 = instr1 ...
    //   vreg2 = instr2 ...
    //   vreg0 = C2_mux ..., vreg1, vreg2
    // If vreg0 was coalesced with vreg1, we could end up with the following
    // code:
    //   vreg0 = instr1 ...
    //   vreg2 = instr2 ...
    //   vreg0 = A2_tfrf ..., vreg2
    // which will later become:
    //   vreg0 = instr1 ...
    //   vreg0 = instr2_cNotPt ...
    // i.e. there will be an unconditional definition (instr1) of vreg0
    // followed by a conditional one. The output dependency was there before
    // and it unavoidable, but if instr1 is predicable, we will no longer be
    // able to predicate it here.
    // To avoid this scenario, don't coalesce the destination register with
    // a source register that is defined by a predicable instruction.
    if (S1.isReg()) {
      RegisterRef RS = S1;
      MachineInstr *RDef = getReachingDefForPred(RS, CI, RP.Reg, true);
      if (!RDef || !HII->isPredicable(*RDef))
        Done = coalesceRegisters(RD, RegisterRef(S1));
    }
    if (!Done && S2.isReg()) {
      RegisterRef RS = S2;
      MachineInstr *RDef = getReachingDefForPred(RS, CI, RP.Reg, false);
      if (!RDef || !HII->isPredicable(*RDef))
        Done = coalesceRegisters(RD, RegisterRef(S2));
    }
    Changed |= Done;
  }
  return Changed;
}


bool HexagonExpandCondsets::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  HII = static_cast<const HexagonInstrInfo*>(MF.getSubtarget().getInstrInfo());
  TRI = MF.getSubtarget().getRegisterInfo();
  LIS = &getAnalysis<LiveIntervals>();
  MRI = &MF.getRegInfo();
  DEBUG(MF.print(dbgs() << "Before expand-condsets\n", LIS->getSlotIndexes()));

  bool Changed = false;

  // Try to coalesce the target of a mux with one of its sources.
  // This could eliminate a register copy in some circumstances.
  Changed |= coalesceSegments(MF);

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    // First, simply split all muxes into a pair of conditional transfers
    // and update the live intervals to reflect the new arrangement.
    // This is done mainly to make the live interval update simpler, than it
    // would be while trying to predicate instructions at the same time.
    Changed |= splitInBlock(*I);
    // Traverse all blocks and collapse predicable instructions feeding
    // conditional transfers into predicated instructions.
    // Walk over all the instructions again, so we may catch pre-existing
    // cases that were not created in the previous step.
    Changed |= predicateInBlock(*I);
  }

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    postprocessUndefImplicitUses(*I);

  if (Changed)
    DEBUG(MF.print(dbgs() << "After expand-condsets\n", LIS->getSlotIndexes()));

  return Changed;
}


//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

static void initializePassOnce(PassRegistry &Registry) {
  const char *Name = "Hexagon Expand Condsets";
  PassInfo *PI = new PassInfo(Name, "expand-condsets",
        &HexagonExpandCondsets::ID, 0, false, false);
  Registry.registerPass(*PI, true);
}

void llvm::initializeHexagonExpandCondsetsPass(PassRegistry &Registry) {
  CALL_ONCE_INITIALIZATION(initializePassOnce)
}


FunctionPass *llvm::createHexagonExpandCondsets() {
  return new HexagonExpandCondsets();
}
