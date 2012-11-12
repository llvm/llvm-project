//===-- TargetInstrInfoImpl.cpp - Target Instruction Information ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetInstrInfoImpl class, it just provides default
// implementations of various methods.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScoreboardHazardRecognizer.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static cl::opt<bool> DisableHazardRecognizer(
  "disable-sched-hazard", cl::Hidden, cl::init(false),
  cl::desc("Disable hazard detection during preRA scheduling"));

/// ReplaceTailWithBranchTo - Delete the instruction OldInst and everything
/// after it, replacing it with an unconditional branch to NewDest.
void
TargetInstrInfoImpl::ReplaceTailWithBranchTo(MachineBasicBlock::iterator Tail,
                                             MachineBasicBlock *NewDest) const {
  MachineBasicBlock *MBB = Tail->getParent();

  // Remove all the old successors of MBB from the CFG.
  while (!MBB->succ_empty())
    MBB->removeSuccessor(MBB->succ_begin());

  // Remove all the dead instructions from the end of MBB.
  MBB->erase(Tail, MBB->end());

  // If MBB isn't immediately before MBB, insert a branch to it.
  if (++MachineFunction::iterator(MBB) != MachineFunction::iterator(NewDest))
    InsertBranch(*MBB, NewDest, 0, SmallVector<MachineOperand, 0>(),
                 Tail->getDebugLoc());
  MBB->addSuccessor(NewDest);
}

// commuteInstruction - The default implementation of this method just exchanges
// the two operands returned by findCommutedOpIndices.
MachineInstr *TargetInstrInfoImpl::commuteInstruction(MachineInstr *MI,
                                                      bool NewMI) const {
  const MCInstrDesc &MCID = MI->getDesc();
  bool HasDef = MCID.getNumDefs();
  if (HasDef && !MI->getOperand(0).isReg())
    // No idea how to commute this instruction. Target should implement its own.
    return 0;
  unsigned Idx1, Idx2;
  if (!findCommutedOpIndices(MI, Idx1, Idx2)) {
    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "Don't know how to commute: " << *MI;
    report_fatal_error(Msg.str());
  }

  assert(MI->getOperand(Idx1).isReg() && MI->getOperand(Idx2).isReg() &&
         "This only knows how to commute register operands so far");
  unsigned Reg0 = HasDef ? MI->getOperand(0).getReg() : 0;
  unsigned Reg1 = MI->getOperand(Idx1).getReg();
  unsigned Reg2 = MI->getOperand(Idx2).getReg();
  unsigned SubReg0 = HasDef ? MI->getOperand(0).getSubReg() : 0;
  unsigned SubReg1 = MI->getOperand(Idx1).getSubReg();
  unsigned SubReg2 = MI->getOperand(Idx2).getSubReg();
  bool Reg1IsKill = MI->getOperand(Idx1).isKill();
  bool Reg2IsKill = MI->getOperand(Idx2).isKill();
  // If destination is tied to either of the commuted source register, then
  // it must be updated.
  if (HasDef && Reg0 == Reg1 &&
      MI->getDesc().getOperandConstraint(Idx1, MCOI::TIED_TO) == 0) {
    Reg2IsKill = false;
    Reg0 = Reg2;
    SubReg0 = SubReg2;
  } else if (HasDef && Reg0 == Reg2 &&
             MI->getDesc().getOperandConstraint(Idx2, MCOI::TIED_TO) == 0) {
    Reg1IsKill = false;
    Reg0 = Reg1;
    SubReg0 = SubReg1;
  }

  if (NewMI) {
    // Create a new instruction.
    MachineFunction &MF = *MI->getParent()->getParent();
    MI = MF.CloneMachineInstr(MI);
  }

  if (HasDef) {
    MI->getOperand(0).setReg(Reg0);
    MI->getOperand(0).setSubReg(SubReg0);
  }
  MI->getOperand(Idx2).setReg(Reg1);
  MI->getOperand(Idx1).setReg(Reg2);
  MI->getOperand(Idx2).setSubReg(SubReg1);
  MI->getOperand(Idx1).setSubReg(SubReg2);
  MI->getOperand(Idx2).setIsKill(Reg1IsKill);
  MI->getOperand(Idx1).setIsKill(Reg2IsKill);
  return MI;
}

/// findCommutedOpIndices - If specified MI is commutable, return the two
/// operand indices that would swap value. Return true if the instruction
/// is not in a form which this routine understands.
bool TargetInstrInfoImpl::findCommutedOpIndices(MachineInstr *MI,
                                                unsigned &SrcOpIdx1,
                                                unsigned &SrcOpIdx2) const {
  assert(!MI->isBundle() &&
         "TargetInstrInfoImpl::findCommutedOpIndices() can't handle bundles");

  const MCInstrDesc &MCID = MI->getDesc();
  if (!MCID.isCommutable())
    return false;
  // This assumes v0 = op v1, v2 and commuting would swap v1 and v2. If this
  // is not true, then the target must implement this.
  SrcOpIdx1 = MCID.getNumDefs();
  SrcOpIdx2 = SrcOpIdx1 + 1;
  if (!MI->getOperand(SrcOpIdx1).isReg() ||
      !MI->getOperand(SrcOpIdx2).isReg())
    // No idea.
    return false;
  return true;
}


bool
TargetInstrInfoImpl::isUnpredicatedTerminator(const MachineInstr *MI) const {
  if (!MI->isTerminator()) return false;

  // Conditional branch is a special case.
  if (MI->isBranch() && !MI->isBarrier())
    return true;
  if (!MI->isPredicable())
    return true;
  return !isPredicated(MI);
}


bool TargetInstrInfoImpl::PredicateInstruction(MachineInstr *MI,
                            const SmallVectorImpl<MachineOperand> &Pred) const {
  bool MadeChange = false;

  assert(!MI->isBundle() &&
         "TargetInstrInfoImpl::PredicateInstruction() can't handle bundles");

  const MCInstrDesc &MCID = MI->getDesc();
  if (!MI->isPredicable())
    return false;

  for (unsigned j = 0, i = 0, e = MI->getNumOperands(); i != e; ++i) {
    if (MCID.OpInfo[i].isPredicate()) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg()) {
        MO.setReg(Pred[j].getReg());
        MadeChange = true;
      } else if (MO.isImm()) {
        MO.setImm(Pred[j].getImm());
        MadeChange = true;
      } else if (MO.isMBB()) {
        MO.setMBB(Pred[j].getMBB());
        MadeChange = true;
      }
      ++j;
    }
  }
  return MadeChange;
}

bool TargetInstrInfoImpl::hasLoadFromStackSlot(const MachineInstr *MI,
                                        const MachineMemOperand *&MMO,
                                        int &FrameIndex) const {
  for (MachineInstr::mmo_iterator o = MI->memoperands_begin(),
         oe = MI->memoperands_end();
       o != oe;
       ++o) {
    if ((*o)->isLoad() && (*o)->getValue())
      if (const FixedStackPseudoSourceValue *Value =
          dyn_cast<const FixedStackPseudoSourceValue>((*o)->getValue())) {
        FrameIndex = Value->getFrameIndex();
        MMO = *o;
        return true;
      }
  }
  return false;
}

bool TargetInstrInfoImpl::hasStoreToStackSlot(const MachineInstr *MI,
                                       const MachineMemOperand *&MMO,
                                       int &FrameIndex) const {
  for (MachineInstr::mmo_iterator o = MI->memoperands_begin(),
         oe = MI->memoperands_end();
       o != oe;
       ++o) {
    if ((*o)->isStore() && (*o)->getValue())
      if (const FixedStackPseudoSourceValue *Value =
          dyn_cast<const FixedStackPseudoSourceValue>((*o)->getValue())) {
        FrameIndex = Value->getFrameIndex();
        MMO = *o;
        return true;
      }
  }
  return false;
}

void TargetInstrInfoImpl::reMaterialize(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator I,
                                        unsigned DestReg,
                                        unsigned SubIdx,
                                        const MachineInstr *Orig,
                                        const TargetRegisterInfo &TRI) const {
  MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
  MI->substituteRegister(MI->getOperand(0).getReg(), DestReg, SubIdx, TRI);
  MBB.insert(I, MI);
}

bool
TargetInstrInfoImpl::produceSameValue(const MachineInstr *MI0,
                                      const MachineInstr *MI1,
                                      const MachineRegisterInfo *MRI) const {
  return MI0->isIdenticalTo(MI1, MachineInstr::IgnoreVRegDefs);
}

MachineInstr *TargetInstrInfoImpl::duplicate(MachineInstr *Orig,
                                             MachineFunction &MF) const {
  assert(!Orig->isNotDuplicable() &&
         "Instruction cannot be duplicated");
  return MF.CloneMachineInstr(Orig);
}

// If the COPY instruction in MI can be folded to a stack operation, return
// the register class to use.
static const TargetRegisterClass *canFoldCopy(const MachineInstr *MI,
                                              unsigned FoldIdx) {
  assert(MI->isCopy() && "MI must be a COPY instruction");
  if (MI->getNumOperands() != 2)
    return 0;
  assert(FoldIdx<2 && "FoldIdx refers no nonexistent operand");

  const MachineOperand &FoldOp = MI->getOperand(FoldIdx);
  const MachineOperand &LiveOp = MI->getOperand(1-FoldIdx);

  if (FoldOp.getSubReg() || LiveOp.getSubReg())
    return 0;

  unsigned FoldReg = FoldOp.getReg();
  unsigned LiveReg = LiveOp.getReg();

  assert(TargetRegisterInfo::isVirtualRegister(FoldReg) &&
         "Cannot fold physregs");

  const MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  const TargetRegisterClass *RC = MRI.getRegClass(FoldReg);

  if (TargetRegisterInfo::isPhysicalRegister(LiveOp.getReg()))
    return RC->contains(LiveOp.getReg()) ? RC : 0;

  if (RC->hasSubClassEq(MRI.getRegClass(LiveReg)))
    return RC;

  // FIXME: Allow folding when register classes are memory compatible.
  return 0;
}

bool TargetInstrInfoImpl::
canFoldMemoryOperand(const MachineInstr *MI,
                     const SmallVectorImpl<unsigned> &Ops) const {
  return MI->isCopy() && Ops.size() == 1 && canFoldCopy(MI, Ops[0]);
}

/// foldMemoryOperand - Attempt to fold a load or store of the specified stack
/// slot into the specified machine instruction for the specified operand(s).
/// If this is possible, a new instruction is returned with the specified
/// operand folded, otherwise NULL is returned. The client is responsible for
/// removing the old instruction and adding the new one in the instruction
/// stream.
MachineInstr*
TargetInstrInfo::foldMemoryOperand(MachineBasicBlock::iterator MI,
                                   const SmallVectorImpl<unsigned> &Ops,
                                   int FI) const {
  unsigned Flags = 0;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    if (MI->getOperand(Ops[i]).isDef())
      Flags |= MachineMemOperand::MOStore;
    else
      Flags |= MachineMemOperand::MOLoad;

  MachineBasicBlock *MBB = MI->getParent();
  assert(MBB && "foldMemoryOperand needs an inserted instruction");
  MachineFunction &MF = *MBB->getParent();

  // Ask the target to do the actual folding.
  if (MachineInstr *NewMI = foldMemoryOperandImpl(MF, MI, Ops, FI)) {
    // Add a memory operand, foldMemoryOperandImpl doesn't do that.
    assert((!(Flags & MachineMemOperand::MOStore) ||
            NewMI->mayStore()) &&
           "Folded a def to a non-store!");
    assert((!(Flags & MachineMemOperand::MOLoad) ||
            NewMI->mayLoad()) &&
           "Folded a use to a non-load!");
    const MachineFrameInfo &MFI = *MF.getFrameInfo();
    assert(MFI.getObjectOffset(FI) != -1);
    MachineMemOperand *MMO =
      MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(FI),
                              Flags, MFI.getObjectSize(FI),
                              MFI.getObjectAlignment(FI));
    NewMI->addMemOperand(MF, MMO);

    // FIXME: change foldMemoryOperandImpl semantics to also insert NewMI.
    return MBB->insert(MI, NewMI);
  }

  // Straight COPY may fold as load/store.
  if (!MI->isCopy() || Ops.size() != 1)
    return 0;

  const TargetRegisterClass *RC = canFoldCopy(MI, Ops[0]);
  if (!RC)
    return 0;

  const MachineOperand &MO = MI->getOperand(1-Ops[0]);
  MachineBasicBlock::iterator Pos = MI;
  const TargetRegisterInfo *TRI = MF.getTarget().getRegisterInfo();

  if (Flags == MachineMemOperand::MOStore)
    storeRegToStackSlot(*MBB, Pos, MO.getReg(), MO.isKill(), FI, RC, TRI);
  else
    loadRegFromStackSlot(*MBB, Pos, MO.getReg(), FI, RC, TRI);
  return --Pos;
}

/// foldMemoryOperand - Same as the previous version except it allows folding
/// of any load and store from / to any address, not just from a specific
/// stack slot.
MachineInstr*
TargetInstrInfo::foldMemoryOperand(MachineBasicBlock::iterator MI,
                                   const SmallVectorImpl<unsigned> &Ops,
                                   MachineInstr* LoadMI) const {
  assert(LoadMI->canFoldAsLoad() && "LoadMI isn't foldable!");
#ifndef NDEBUG
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    assert(MI->getOperand(Ops[i]).isUse() && "Folding load into def!");
#endif
  MachineBasicBlock &MBB = *MI->getParent();
  MachineFunction &MF = *MBB.getParent();

  // Ask the target to do the actual folding.
  MachineInstr *NewMI = foldMemoryOperandImpl(MF, MI, Ops, LoadMI);
  if (!NewMI) return 0;

  NewMI = MBB.insert(MI, NewMI);

  // Copy the memoperands from the load to the folded instruction.
  NewMI->setMemRefs(LoadMI->memoperands_begin(),
                    LoadMI->memoperands_end());

  return NewMI;
}

bool TargetInstrInfo::
isReallyTriviallyReMaterializableGeneric(const MachineInstr *MI,
                                         AliasAnalysis *AA) const {
  const MachineFunction &MF = *MI->getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetMachine &TM = MF.getTarget();
  const TargetInstrInfo &TII = *TM.getInstrInfo();

  // Remat clients assume operand 0 is the defined register.
  if (!MI->getNumOperands() || !MI->getOperand(0).isReg())
    return false;
  unsigned DefReg = MI->getOperand(0).getReg();

  // A sub-register definition can only be rematerialized if the instruction
  // doesn't read the other parts of the register.  Otherwise it is really a
  // read-modify-write operation on the full virtual register which cannot be
  // moved safely.
  if (TargetRegisterInfo::isVirtualRegister(DefReg) &&
      MI->getOperand(0).getSubReg() && MI->readsVirtualRegister(DefReg))
    return false;

  // A load from a fixed stack slot can be rematerialized. This may be
  // redundant with subsequent checks, but it's target-independent,
  // simple, and a common case.
  int FrameIdx = 0;
  if (TII.isLoadFromStackSlot(MI, FrameIdx) &&
      MF.getFrameInfo()->isImmutableObjectIndex(FrameIdx))
    return true;

  // Avoid instructions obviously unsafe for remat.
  if (MI->isNotDuplicable() || MI->mayStore() ||
      MI->hasUnmodeledSideEffects())
    return false;

  // Don't remat inline asm. We have no idea how expensive it is
  // even if it's side effect free.
  if (MI->isInlineAsm())
    return false;

  // Avoid instructions which load from potentially varying memory.
  if (MI->mayLoad() && !MI->isInvariantLoad(AA))
    return false;

  // If any of the registers accessed are non-constant, conservatively assume
  // the instruction is not rematerializable.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0)
      continue;

    // Check for a well-behaved physical register.
    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      if (MO.isUse()) {
        // If the physreg has no defs anywhere, it's just an ambient register
        // and we can freely move its uses. Alternatively, if it's allocatable,
        // it could get allocated to something with a def during allocation.
        if (!MRI.isConstantPhysReg(Reg, MF))
          return false;
      } else {
        // A physreg def. We can't remat it.
        return false;
      }
      continue;
    }

    // Only allow one virtual-register def.  There may be multiple defs of the
    // same virtual register, though.
    if (MO.isDef() && Reg != DefReg)
      return false;

    // Don't allow any virtual-register uses. Rematting an instruction with
    // virtual register uses would length the live ranges of the uses, which
    // is not necessarily a good idea, certainly not "trivial".
    if (MO.isUse())
      return false;
  }

  // Everything checked out.
  return true;
}

/// isSchedulingBoundary - Test if the given instruction should be
/// considered a scheduling boundary. This primarily includes labels
/// and terminators.
bool TargetInstrInfoImpl::isSchedulingBoundary(const MachineInstr *MI,
                                               const MachineBasicBlock *MBB,
                                               const MachineFunction &MF) const{
  // Terminators and labels can't be scheduled around.
  if (MI->isTerminator() || MI->isLabel())
    return true;

  // Don't attempt to schedule around any instruction that defines
  // a stack-oriented pointer, as it's unlikely to be profitable. This
  // saves compile time, because it doesn't require every single
  // stack slot reference to depend on the instruction that does the
  // modification.
  const TargetLowering &TLI = *MF.getTarget().getTargetLowering();
  if (MI->definesRegister(TLI.getStackPointerRegisterToSaveRestore()))
    return true;

  return false;
}

// Provide a global flag for disabling the PreRA hazard recognizer that targets
// may choose to honor.
bool TargetInstrInfoImpl::usePreRAHazardRecognizer() const {
  return !DisableHazardRecognizer;
}

// Default implementation of CreateTargetRAHazardRecognizer.
ScheduleHazardRecognizer *TargetInstrInfoImpl::
CreateTargetHazardRecognizer(const TargetMachine *TM,
                             const ScheduleDAG *DAG) const {
  // Dummy hazard recognizer allows all instructions to issue.
  return new ScheduleHazardRecognizer();
}

// Default implementation of CreateTargetMIHazardRecognizer.
ScheduleHazardRecognizer *TargetInstrInfoImpl::
CreateTargetMIHazardRecognizer(const InstrItineraryData *II,
                               const ScheduleDAG *DAG) const {
  return (ScheduleHazardRecognizer *)
    new ScoreboardHazardRecognizer(II, DAG, "misched");
}

// Default implementation of CreateTargetPostRAHazardRecognizer.
ScheduleHazardRecognizer *TargetInstrInfoImpl::
CreateTargetPostRAHazardRecognizer(const InstrItineraryData *II,
                                   const ScheduleDAG *DAG) const {
  return (ScheduleHazardRecognizer *)
    new ScoreboardHazardRecognizer(II, DAG, "post-RA-sched");
}

//===----------------------------------------------------------------------===//
//  SelectionDAG latency interface.
//===----------------------------------------------------------------------===//

int
TargetInstrInfoImpl::getOperandLatency(const InstrItineraryData *ItinData,
                                       SDNode *DefNode, unsigned DefIdx,
                                       SDNode *UseNode, unsigned UseIdx) const {
  if (!ItinData || ItinData->isEmpty())
    return -1;

  if (!DefNode->isMachineOpcode())
    return -1;

  unsigned DefClass = get(DefNode->getMachineOpcode()).getSchedClass();
  if (!UseNode->isMachineOpcode())
    return ItinData->getOperandCycle(DefClass, DefIdx);
  unsigned UseClass = get(UseNode->getMachineOpcode()).getSchedClass();
  return ItinData->getOperandLatency(DefClass, DefIdx, UseClass, UseIdx);
}

int TargetInstrInfoImpl::getInstrLatency(const InstrItineraryData *ItinData,
                                         SDNode *N) const {
  if (!ItinData || ItinData->isEmpty())
    return 1;

  if (!N->isMachineOpcode())
    return 1;

  return ItinData->getStageLatency(get(N->getMachineOpcode()).getSchedClass());
}

//===----------------------------------------------------------------------===//
//  MachineInstr latency interface.
//===----------------------------------------------------------------------===//

unsigned
TargetInstrInfoImpl::getNumMicroOps(const InstrItineraryData *ItinData,
                                    const MachineInstr *MI) const {
  if (!ItinData || ItinData->isEmpty())
    return 1;

  unsigned Class = MI->getDesc().getSchedClass();
  int UOps = ItinData->Itineraries[Class].NumMicroOps;
  if (UOps >= 0)
    return UOps;

  // The # of u-ops is dynamically determined. The specific target should
  // override this function to return the right number.
  return 1;
}

/// Return the default expected latency for a def based on it's opcode.
unsigned TargetInstrInfo::defaultDefLatency(const MCSchedModel *SchedModel,
                                            const MachineInstr *DefMI) const {
  if (DefMI->isTransient())
    return 0;
  if (DefMI->mayLoad())
    return SchedModel->LoadLatency;
  if (isHighLatencyDef(DefMI->getOpcode()))
    return SchedModel->HighLatency;
  return 1;
}

unsigned TargetInstrInfoImpl::
getInstrLatency(const InstrItineraryData *ItinData,
                const MachineInstr *MI,
                unsigned *PredCost) const {
  // Default to one cycle for no itinerary. However, an "empty" itinerary may
  // still have a MinLatency property, which getStageLatency checks.
  if (!ItinData)
    return MI->mayLoad() ? 2 : 1;

  return ItinData->getStageLatency(MI->getDesc().getSchedClass());
}

bool TargetInstrInfoImpl::hasLowDefLatency(const InstrItineraryData *ItinData,
                                           const MachineInstr *DefMI,
                                           unsigned DefIdx) const {
  if (!ItinData || ItinData->isEmpty())
    return false;

  unsigned DefClass = DefMI->getDesc().getSchedClass();
  int DefCycle = ItinData->getOperandCycle(DefClass, DefIdx);
  return (DefCycle != -1 && DefCycle <= 1);
}

/// Both DefMI and UseMI must be valid.  By default, call directly to the
/// itinerary. This may be overriden by the target.
int TargetInstrInfoImpl::
getOperandLatency(const InstrItineraryData *ItinData,
                  const MachineInstr *DefMI, unsigned DefIdx,
                  const MachineInstr *UseMI, unsigned UseIdx) const {
  unsigned DefClass = DefMI->getDesc().getSchedClass();
  unsigned UseClass = UseMI->getDesc().getSchedClass();
  return ItinData->getOperandLatency(DefClass, DefIdx, UseClass, UseIdx);
}

/// If we can determine the operand latency from the def only, without itinerary
/// lookup, do so. Otherwise return -1.
int TargetInstrInfo::computeDefOperandLatency(
  const InstrItineraryData *ItinData,
  const MachineInstr *DefMI, bool FindMin) const {

  // Let the target hook getInstrLatency handle missing itineraries.
  if (!ItinData)
    return getInstrLatency(ItinData, DefMI);

  // Return a latency based on the itinerary properties and defining instruction
  // if possible. Some common subtargets don't require per-operand latency,
  // especially for minimum latencies.
  if (FindMin) {
    // If MinLatency is valid, call getInstrLatency. This uses Stage latency if
    // it exists before defaulting to MinLatency.
    if (ItinData->SchedModel->MinLatency >= 0)
      return getInstrLatency(ItinData, DefMI);

    // If MinLatency is invalid, OperandLatency is interpreted as MinLatency.
    // For empty itineraries, short-cirtuit the check and default to one cycle.
    if (ItinData->isEmpty())
      return 1;
  }
  else if(ItinData->isEmpty())
    return defaultDefLatency(ItinData->SchedModel, DefMI);

  // ...operand lookup required
  return -1;
}

/// computeOperandLatency - Compute and return the latency of the given data
/// dependent def and use when the operand indices are already known. UseMI may
/// be NULL for an unknown use.
///
/// FindMin may be set to get the minimum vs. expected latency. Minimum
/// latency is used for scheduling groups, while expected latency is for
/// instruction cost and critical path.
///
/// Depending on the subtarget's itinerary properties, this may or may not need
/// to call getOperandLatency(). For most subtargets, we don't need DefIdx or
/// UseIdx to compute min latency.
unsigned TargetInstrInfo::
computeOperandLatency(const InstrItineraryData *ItinData,
                      const MachineInstr *DefMI, unsigned DefIdx,
                      const MachineInstr *UseMI, unsigned UseIdx,
                      bool FindMin) const {

  int DefLatency = computeDefOperandLatency(ItinData, DefMI, FindMin);
  if (DefLatency >= 0)
    return DefLatency;

  assert(ItinData && !ItinData->isEmpty() && "computeDefOperandLatency fail");

  int OperLatency = 0;
  if (UseMI)
    OperLatency = getOperandLatency(ItinData, DefMI, DefIdx, UseMI, UseIdx);
  else {
    unsigned DefClass = DefMI->getDesc().getSchedClass();
    OperLatency = ItinData->getOperandCycle(DefClass, DefIdx);
  }
  if (OperLatency >= 0)
    return OperLatency;

  // No operand latency was found.
  unsigned InstrLatency = getInstrLatency(ItinData, DefMI);

  // Expected latency is the max of the stage latency and itinerary props.
  if (!FindMin)
    InstrLatency = std::max(InstrLatency,
                            defaultDefLatency(ItinData->SchedModel, DefMI));
  return InstrLatency;
}
