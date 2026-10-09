//===---- ScheduleDAGInstrs.cpp - MachineInstr Rescheduling ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This implements the ScheduleDAGInstrs class, which implements
/// re-scheduling of MachineInstrs.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ScheduleDAGInstrs.h"

#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDFS.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <list>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

static cl::opt<bool>
    EnableAASchedMI("enable-aa-sched-mi", cl::Hidden,
                    cl::desc("Enable use of AA during MI DAG construction"));

static cl::opt<bool> UseTBAA("use-tbaa-in-sched-mi", cl::Hidden,
    cl::init(true), cl::desc("Enable use of TBAA during MI DAG construction"));

static cl::opt<bool>
    EnableSchedModel("schedmodel", cl::Hidden, cl::init(true),
                     cl::desc("Use TargetSchedModel for latency lookup"));

static cl::opt<bool>
    EnableSchedItins("scheditins", cl::Hidden, cl::init(true),
                     cl::desc("Use InstrItineraryData for latency lookup"));

// Note: the two options below might be used in tuning compile time vs output
// quality. Setting these options so large that they will never be reached means
// best-effort, but may be slow.
static cl::opt<unsigned> MaxFrontierSize(
    "store-sequencing-max-frontier-size", cl::Hidden, cl::init(500),
    cl::desc("With store-sequencing DAG construction, after the number of "
             "memops in a frontier exceeds this threshold, automatically "
             "promote the next store and clear the frontier."));

static cl::opt<unsigned> MaxTotalMemops(
    "store-sequencing-max-total-memops", cl::Hidden, cl::init(3000),
    cl::desc("With store-sequencing DAG construction, if the occupancy of all "
             "frontiers exceeds this threshold, automatically promote the next "
             "memory operation to a global memory barrier and clear the "
             "frontiers."));

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
static cl::opt<bool> SchedPrintCycles(
    "sched-print-cycles", cl::Hidden, cl::init(false),
    cl::desc("Report top/bottom cycles when dumping SUnit instances"));
#endif

ScheduleDAGInstrs::ScheduleDAGInstrs(MachineFunction &mf,
                                     const MachineLoopInfo *mli,
                                     bool RemoveKillFlags)
    : ScheduleDAG(mf), MLI(mli), MFI(mf.getFrameInfo()),
      RemoveKillFlags(RemoveKillFlags), Topo(SUnits, &ExitSU) {
  DbgValues.clear();

  const TargetSubtargetInfo &ST = mf.getSubtarget();
  SchedModel.init(&ST, EnableSchedModel, EnableSchedItins);
}

/// If this machine instruction has memory reference information, collect the
/// list of underlying objects in \p Objects. If any of these objects are
/// unknown or may alias anything, return false. Atomic and volatile memory
/// operands are skipped.
static bool getUnderlyingObjectsForInstr(const MachineInstr *MI,
                                         const MachineFrameInfo &MFI,
                                         UnderlyingObjectsVector &Objects,
                                         const DataLayout &DL) {
  bool AllObjectsIdentified = true;

  for (const MachineMemOperand *MMO : MI->memoperands()) {
    // TODO: Figure out whether isAtomic is really necessary (see D57601).
    if (MMO->isVolatile() || MMO->isAtomic()) {
      AllObjectsIdentified = false;
      continue;
    }

    if (const PseudoSourceValue *PSV = MMO->getPseudoValue()) {
      if (MFI.hasTailCall()) {
        // Function that contain tail calls don't have unique PseudoSourceValue
        // objects. Two PseudoSourceValues might refer to the same or
        // overlapping locations. The client code calling this function assumes
        // this is not the case. So return a conservative answer of no known
        // object.
        AllObjectsIdentified = false;
      } else if (PSV->isAliased(&MFI)) {
        // For now, ignore PseudoSourceValues which may alias LLVM IR values
        // because the code that uses this function has no way to cope with such
        // aliases.
        AllObjectsIdentified = false;
      }

      Objects.push_back(PSV);
    } else if (const Value *V = MMO->getValue()) {
      SmallVector<Value *, 4> Objs;
      bool ObjectsIdentified = getUnderlyingObjectsForCodeGen(V, Objs);
      AllObjectsIdentified &= ObjectsIdentified;

      for (Value *V : Objs) {
        assert(!ObjectsIdentified || isIdentifiedObject(V));
        Objects.push_back(V);
      }
    } else {
      AllObjectsIdentified = false;
    }
  }

  return AllObjectsIdentified;
}

void ScheduleDAGInstrs::startBlock(MachineBasicBlock *bb) {
  BB = bb;
}

void ScheduleDAGInstrs::finishBlock() {
  // Subclasses should no longer refer to the old block.
  BB = nullptr;
}

void ScheduleDAGInstrs::enterRegion(MachineBasicBlock *bb,
                                    MachineBasicBlock::iterator begin,
                                    MachineBasicBlock::iterator end,
                                    unsigned regioninstrs) {
  assert(bb == BB && "startBlock should set BB");
  RegionBegin = begin;
  RegionEnd = end;
  NumRegionInstrs = regioninstrs;
}

void ScheduleDAGInstrs::exitRegion() {
  // Nothing to do.
}

void ScheduleDAGInstrs::addSchedBarrierDeps() {
  MachineInstr *ExitMI =
      RegionEnd != BB->end()
          ? &*skipDebugInstructionsBackward(RegionEnd, RegionBegin)
          : nullptr;
  ExitSU.setInstr(ExitMI);
  // Add dependencies on the defs and uses of the instruction.
  if (ExitMI) {
    const MCInstrDesc &MIDesc = ExitMI->getDesc();
    for (const MachineOperand &MO : ExitMI->all_uses()) {
      unsigned OpIdx = MO.getOperandNo();
      Register Reg = MO.getReg();
      if (Reg.isPhysical()) {
        // addPhysRegDataDeps uses the provided operand index to retrieve
        // the operand use cycle from the scheduling model. If the operand
        // is "fake" (e.g., an operand of a call instruction used to pass
        // an argument to the called function.), the scheduling model may not
        // have an entry for it. If this is the case, pass -1 as operand index,
        // which will cause addPhysRegDataDeps to add an artificial dependency.
        // FIXME: Using hasImplicitUseOfPhysReg here is inaccurate as it misses
        //  aliases. When fixing, make sure to update addPhysRegDataDeps, too.
        bool IsRealUse = OpIdx < MIDesc.getNumOperands() ||
                         MIDesc.hasImplicitUseOfPhysReg(Reg);
        for (MCRegUnit Unit : TRI->regunits(Reg))
          Uses.insert(PhysRegSUOper(&ExitSU, IsRealUse ? OpIdx : -1, Unit));
      } else if (Reg.isVirtual() && MO.readsReg()) {
        addVRegUseDeps(&ExitSU, OpIdx);
      }
    }
  }
  if (!ExitMI || (!ExitMI->isCall() && !ExitMI->isBarrier())) {
    // For others, e.g. fallthrough, conditional branch, assume the exit
    // uses all the registers that are livein to the successor blocks.
    for (const MachineBasicBlock *Succ : BB->successors()) {
      for (const auto &LI : Succ->liveins()) {
        for (MCRegUnitMaskIterator U(LI.PhysReg, TRI); U.isValid(); ++U) {
          auto [Unit, Mask] = *U;
          if ((Mask & LI.LaneMask).any() && !Uses.contains(Unit))
            Uses.insert(PhysRegSUOper(&ExitSU, -1, Unit));
        }
      }
    }
  }
}

/// MO is an operand of SU's instruction that defines a physical register. Adds
/// data dependencies from SU to any uses of the physical register.
void ScheduleDAGInstrs::addPhysRegDataDeps(SUnit *SU, unsigned OperIdx) {
  const MachineOperand &MO = SU->getInstr()->getOperand(OperIdx);
  assert(MO.isDef() && "expect physreg def");
  Register Reg = MO.getReg();

  // Ask the target if address-backscheduling is desirable, and if so how much.
  const TargetSubtargetInfo &ST = MF.getSubtarget();

  // Only use any non-zero latency for real defs/uses, in contrast to
  // "fake" operands added by regalloc.
  const MCInstrDesc &DefMIDesc = SU->getInstr()->getDesc();
  bool ImplicitPseudoDef = (OperIdx >= DefMIDesc.getNumOperands() &&
                            !DefMIDesc.hasImplicitDefOfPhysReg(Reg));
  for (MCRegUnit Unit : TRI->regunits(Reg)) {
    for (RegUnit2SUnitsMap::iterator I = Uses.find(Unit); I != Uses.end();
         ++I) {
      SUnit *UseSU = I->SU;
      if (UseSU == SU)
        continue;

      // Adjust the dependence latency using operand def/use information,
      // then allow the target to perform its own adjustments.
      MachineInstr *UseInstr = nullptr;
      int UseOpIdx = I->OpIdx;
      bool ImplicitPseudoUse = false;
      SDep Dep;
      if (UseOpIdx < 0) {
        Dep = SDep(SU, SDep::Artificial);
      } else {
        // Set the hasPhysRegDefs only for physreg defs that have a use within
        // the scheduling region.
        SU->hasPhysRegDefs = true;

        UseInstr = UseSU->getInstr();
        Register UseReg = UseInstr->getOperand(UseOpIdx).getReg();
        const MCInstrDesc &UseMIDesc = UseInstr->getDesc();
        ImplicitPseudoUse = UseOpIdx >= ((int)UseMIDesc.getNumOperands()) &&
                            !UseMIDesc.hasImplicitUseOfPhysReg(UseReg);

        Dep = SDep(SU, SDep::Data, UseReg);
      }
      if (!ImplicitPseudoDef && !ImplicitPseudoUse) {
        Dep.setLatency(SchedModel.computeOperandLatency(SU->getInstr(), OperIdx,
                                                        UseInstr, UseOpIdx));
      } else {
        Dep.setLatency(0);
      }
      ST.adjustSchedDependency(SU, OperIdx, UseSU, UseOpIdx, Dep, &SchedModel);
      UseSU->addPred(Dep);
    }
  }
}

/// Adds register dependencies (data, anti, and output) from this SUnit
/// to following instructions in the same scheduling region that depend the
/// physical register referenced at OperIdx.
void ScheduleDAGInstrs::addPhysRegDeps(SUnit *SU, unsigned OperIdx) {
  MachineInstr *MI = SU->getInstr();
  MachineOperand &MO = MI->getOperand(OperIdx);
  Register Reg = MO.getReg();
  // We do not need to track any dependencies for constant registers.
  if (MRI.isConstantPhysReg(Reg))
    return;

  const TargetSubtargetInfo &ST = MF.getSubtarget();

  // Optionally add output and anti dependencies. For anti
  // dependencies we use a latency of 0 because for a multi-issue
  // target we want to allow the defining instruction to issue
  // in the same cycle as the using instruction.
  // TODO: Using a latency of 1 here for output dependencies assumes
  //       there's no cost for reusing registers.
  SDep::Kind Kind = MO.isUse() ? SDep::Anti : SDep::Output;
  for (MCRegUnit Unit : TRI->regunits(Reg)) {
    for (RegUnit2SUnitsMap::iterator I = Defs.find(Unit); I != Defs.end();
         ++I) {
      SUnit *DefSU = I->SU;
      if (DefSU == &ExitSU)
        continue;
      MachineInstr *DefInstr = DefSU->getInstr();
      MachineOperand &DefMO = DefInstr->getOperand(I->OpIdx);
      if (DefSU != SU &&
          (Kind != SDep::Output || !MO.isDead() || !DefMO.isDead())) {
        SDep Dep(SU, Kind, DefMO.getReg());
        if (Kind != SDep::Anti) {
          Dep.setLatency(
              SchedModel.computeOutputLatency(MI, OperIdx, DefInstr));
        }
        ST.adjustSchedDependency(SU, OperIdx, DefSU, I->OpIdx, Dep,
                                 &SchedModel);
        DefSU->addPred(Dep);
      }
    }
  }

  if (MO.isUse()) {
    SU->hasPhysRegUses = true;
    // Either insert a new Reg2SUnits entry with an empty SUnits list, or
    // retrieve the existing SUnits list for this register's uses.
    // Push this SUnit on the use list.
    for (MCRegUnit Unit : TRI->regunits(Reg))
      Uses.insert(PhysRegSUOper(SU, OperIdx, Unit));
    if (RemoveKillFlags)
      MO.setIsKill(false);
  } else {
    addPhysRegDataDeps(SU, OperIdx);

    // Clear previous uses and defs of this register and its subregisters.
    for (MCRegUnit Unit : TRI->regunits(Reg)) {
      Uses.eraseAll(Unit);
      if (!MO.isDead())
        Defs.eraseAll(Unit);
    }

    if (MO.isDead() && SU->isCall) {
      // Calls will not be reordered because of chain dependencies (see
      // below). Since call operands are dead, calls may continue to be added
      // to the DefList making dependence checking quadratic in the size of
      // the block. Instead, we leave only one call at the back of the
      // DefList.
      for (MCRegUnit Unit : TRI->regunits(Reg)) {
        RegUnit2SUnitsMap::RangePair P = Defs.equal_range(Unit);
        RegUnit2SUnitsMap::iterator B = P.first;
        RegUnit2SUnitsMap::iterator I = P.second;
        for (bool isBegin = I == B; !isBegin; /* empty */) {
          isBegin = (--I) == B;
          if (!I->SU->isCall)
            break;
          I = Defs.erase(I);
        }
      }
    }

    // Defs are pushed in the order they are visited and never reordered.
    for (MCRegUnit Unit : TRI->regunits(Reg))
      Defs.insert(PhysRegSUOper(SU, OperIdx, Unit));
  }
}

LaneBitmask ScheduleDAGInstrs::getLaneMaskForMO(const MachineOperand &MO) const
{
  Register Reg = MO.getReg();
  // No point in tracking lanemasks if we don't have interesting subregisters.
  const TargetRegisterClass &RC = *MRI.getRegClass(Reg);
  if (!RC.HasDisjunctSubRegs)
    return LaneBitmask::getAll();

  unsigned SubReg = MO.getSubReg();
  if (SubReg == 0)
    return RC.getLaneMask();
  return TRI->getSubRegIndexLaneMask(SubReg);
}

bool ScheduleDAGInstrs::deadDefHasNoUse(const MachineOperand &MO) {
  auto RegUse = CurrentVRegUses.find(MO.getReg());
  if (RegUse == CurrentVRegUses.end())
    return true;
  return (RegUse->LaneMask & getLaneMaskForMO(MO)).none();
}

/// Adds register output and data dependencies from this SUnit to instructions
/// that occur later in the same scheduling region if they read from or write to
/// the virtual register defined at OperIdx.
///
/// TODO: Hoist loop induction variable increments. This has to be
/// reevaluated. Generally, IV scheduling should be done before coalescing.
void ScheduleDAGInstrs::addVRegDefDeps(SUnit *SU, unsigned OperIdx) {
  MachineInstr *MI = SU->getInstr();
  MachineOperand &MO = MI->getOperand(OperIdx);
  Register Reg = MO.getReg();

  LaneBitmask DefLaneMask;
  LaneBitmask KillLaneMask;
  if (TrackLaneMasks) {
    bool IsKill = MO.getSubReg() == 0 || MO.isUndef();
    DefLaneMask = getLaneMaskForMO(MO);
    // If we have a <read-undef> flag, none of the lane values comes from an
    // earlier instruction.
    KillLaneMask = IsKill ? LaneBitmask::getAll() : DefLaneMask;

    if (MO.getSubReg() != 0 && MO.isUndef()) {
      // There may be other subregister defs on the same instruction of the same
      // register in later operands. The lanes of other defs will now be live
      // after this instruction, so these should not be treated as killed by the
      // instruction even though they appear to be killed in this one operand.
      for (const MachineOperand &OtherMO :
           llvm::drop_begin(MI->operands(), OperIdx + 1))
        if (OtherMO.isReg() && OtherMO.isDef() && OtherMO.getReg() == Reg)
          KillLaneMask &= ~getLaneMaskForMO(OtherMO);
    }

    // Clear undef flag, we'll re-add it later once we know which subregister
    // Def is first.
    MO.setIsUndef(false);
  } else {
    DefLaneMask = LaneBitmask::getAll();
    KillLaneMask = LaneBitmask::getAll();
  }

  if (MO.isDead()) {
    assert(deadDefHasNoUse(MO) && "Dead defs should have no uses");
  } else {
    // Add data dependence to all uses we found so far.
    const TargetSubtargetInfo &ST = MF.getSubtarget();
    for (VReg2SUnitOperIdxMultiMap::iterator I = CurrentVRegUses.find(Reg),
         E = CurrentVRegUses.end(); I != E; /*empty*/) {
      LaneBitmask LaneMask = I->LaneMask;
      // Ignore uses of other lanes.
      if ((LaneMask & KillLaneMask).none()) {
        ++I;
        continue;
      }

      if ((LaneMask & DefLaneMask).any()) {
        SUnit *UseSU = I->SU;
        MachineInstr *Use = UseSU->getInstr();
        SDep Dep(SU, SDep::Data, Reg);
        Dep.setLatency(SchedModel.computeOperandLatency(MI, OperIdx, Use,
                                                        I->OperandIndex));
        ST.adjustSchedDependency(SU, OperIdx, UseSU, I->OperandIndex, Dep,
                                 &SchedModel);
        UseSU->addPred(Dep);
      }

      LaneMask &= ~KillLaneMask;
      // If we found a Def for all lanes of this use, remove it from the list.
      if (LaneMask.any()) {
        I->LaneMask = LaneMask;
        ++I;
      } else
        I = CurrentVRegUses.erase(I);
    }
  }

  // Shortcut: Singly defined vregs do not have output/anti dependencies.
  if (MRI.hasOneDef(Reg))
    return;

  // Add output dependence to the next nearest defs of this vreg.
  //
  // Unless this definition is dead, the output dependence should be
  // transitively redundant with antidependencies from this definition's
  // uses. We're conservative for now until we have a way to guarantee the uses
  // are not eliminated sometime during scheduling. The output dependence edge
  // is also useful if output latency exceeds def-use latency.
  LaneBitmask LaneMask = DefLaneMask;
  for (VReg2SUnit &V2SU : make_range(CurrentVRegDefs.find(Reg),
                                     CurrentVRegDefs.end())) {
    // Ignore defs for other lanes.
    if ((V2SU.LaneMask & LaneMask).none())
      continue;
    // Add an output dependence.
    SUnit *DefSU = V2SU.SU;
    // Ignore additional defs of the same lanes in one instruction. This can
    // happen because lanemasks are shared for targets with too many
    // subregisters. We also use some representration tricks/hacks where we
    // add super-register defs/uses, to imply that although we only access parts
    // of the reg we care about the full one.
    if (DefSU == SU)
      continue;
    SDep Dep(SU, SDep::Output, Reg);
    Dep.setLatency(
      SchedModel.computeOutputLatency(MI, OperIdx, DefSU->getInstr()));
    DefSU->addPred(Dep);

    // Update current definition. This can get tricky if the def was about a
    // bigger lanemask before. We then have to shrink it and create a new
    // VReg2SUnit for the non-overlapping part.
    LaneBitmask OverlapMask = V2SU.LaneMask & LaneMask;
    LaneBitmask NonOverlapMask = V2SU.LaneMask & ~LaneMask;
    V2SU.SU = SU;
    V2SU.LaneMask = OverlapMask;
    if (NonOverlapMask.any())
      CurrentVRegDefs.insert(VReg2SUnit(Reg, NonOverlapMask, DefSU));
  }
  // If there was no CurrentVRegDefs entry for some lanes yet, create one.
  if (LaneMask.any())
    CurrentVRegDefs.insert(VReg2SUnit(Reg, LaneMask, SU));
}

/// Adds a register data dependency if the instruction that defines the
/// virtual register used at OperIdx is mapped to an SUnit. Add a register
/// antidependency from this SUnit to instructions that occur later in the same
/// scheduling region if they write the virtual register.
///
/// TODO: Handle ExitSU "uses" properly.
void ScheduleDAGInstrs::addVRegUseDeps(SUnit *SU, unsigned OperIdx) {
  const MachineInstr *MI = SU->getInstr();
  assert(!MI->isDebugOrPseudoInstr());

  const MachineOperand &MO = MI->getOperand(OperIdx);
  Register Reg = MO.getReg();

  // Remember the use. Data dependencies will be added when we find the def.
  LaneBitmask LaneMask = TrackLaneMasks ? getLaneMaskForMO(MO)
                                        : LaneBitmask::getAll();
  CurrentVRegUses.insert(VReg2SUnitOperIdx(Reg, LaneMask, OperIdx, SU));

  // Add antidependences to the following defs of the vreg.
  for (VReg2SUnit &V2SU : make_range(CurrentVRegDefs.find(Reg),
                                     CurrentVRegDefs.end())) {
    // Ignore defs for unrelated lanes.
    LaneBitmask PrevDefLaneMask = V2SU.LaneMask;
    if ((PrevDefLaneMask & LaneMask).none())
      continue;
    if (V2SU.SU == SU)
      continue;

    V2SU.SU->addPred(SDep(SU, SDep::Anti, Reg));
  }
}

/// Creates an SUnit for each real instruction, numbered in top-down
/// topological order. The instruction order A < B, implies that no edge exists
/// from B to A.
///
/// Map each real instruction to its SUnit.
///
/// After initSUnits, the SUnits vector cannot be resized and the scheduler may
/// hang onto SUnit pointers. We may relax this in the future by using SUnit IDs
/// instead of pointers.
///
/// MachineScheduler relies on initSUnits numbering the nodes by their order in
/// the original instruction list.
void ScheduleDAGInstrs::initSUnits() {
  // We'll be allocating one SUnit for each real instruction in the region,
  // which is contained within a basic block.
  SUnits.reserve(NumRegionInstrs);

  for (MachineInstr &MI : make_range(RegionBegin, RegionEnd)) {
    if (MI.isDebugOrPseudoInstr())
      continue;

    SUnit *SU = newSUnit(&MI);
    MISUnitMap[&MI] = SU;

    SU->isCall = MI.isCall();
    SU->isCommutable = MI.isCommutable();

    // Assign the Latency field of SU using target-provided information.
    SU->Latency = SchedModel.computeInstrLatency(SU->getInstr());

    // If this SUnit uses a reserved or unbuffered resource, mark it as such.
    //
    // Reserved resources block an instruction from issuing and stall the
    // entire pipeline. These are identified by BufferSize=0.
    //
    // Unbuffered resources prevent execution of subsequent instructions that
    // require the same resources. This is used for in-order execution pipelines
    // within an out-of-order core. These are identified by BufferSize=1.
    if (SchedModel.hasInstrSchedModel()) {
      const MCSchedClassDesc *SC = getSchedClass(SU);
      for (const MCWriteProcResEntry &PRE :
           make_range(SchedModel.getWriteProcResBegin(SC),
                      SchedModel.getWriteProcResEnd(SC))) {
        switch (SchedModel.getResourceBufferSize(PRE.ProcResourceIdx)) {
        case 0:
          SU->hasReservedResource = true;
          break;
        case 1:
          SU->isUnbuffered = true;
          break;
        default:
          break;
        }
      }
    }
  }
}

namespace {
/// A list of SUnits, used in Value2SUsMap, during DAG construction.
/// FIXME: to gain speed it might be worth investigating an optimized
/// implementation of this data structure, such as a singly linked list
/// with a memory pool (SmallVector was tried but slow and SparseSet is not
/// applicable).
using SUList = std::list<SUnit *>;
} // end anonymous namespace

namespace llvm {
class ScheduleDAGDependencyBuilder {
private:
  ScheduleDAGInstrs &DAG;

  BatchAAResults *AA;
  RegPressureTracker *RPTracker;
  PressureDiffs *PDiffs;
  LiveIntervals *LIS;

  // Track all instructions that may raise floating-point exceptions.
  // These do not depend on one other (or normal loads or stores), but
  // must not be rescheduled across global barriers.
  SUList FPExceptions;

  /// A frontier of memory operations.
  ///
  /// As we process memory operations (bottom to top), each new memory operation
  /// needs to be checked for required control dependencies against later memory
  /// operations. A naive implementation issues AA calls quadratically in the
  /// number of memory operations and is therefore unacceptable.
  ///
  /// To bound the complexity, we only ever consider a frontier of memory
  /// operations which we frequently clear, modeled by this struct. To ensure
  /// that we are not missing any necessary control dependencies, we promote
  /// certain store instructions to what we term 'sequencing stores'. These
  /// sequencing stores are treated carefully to ensure that all previously seen
  /// memory operations that have been cleared from the frontier necessarily
  /// transitively succeed the current sequencing store.
  ///
  /// By frequently updating the sequencing store, we bound the number of alias
  /// analysis queries and transitively redundant control dependencies we
  /// generate. However, doing so we necessarily overconstrain the DAG. Indeed,
  /// if we were to update the sequencing store every time we see a store
  /// instruction, the effect would be to linearize the stores.
  struct MemopFrontier {
    // The current sequencing store.
    SUnit *SequencingStore = nullptr;
    /// The memory operations in the frontier.
    SUList Stores, Loads;

    /// The number of memops in the frontier, excluding the sequencing store.
    unsigned size() const;

    /// Clears the frontier.
    void clear();
  };

  // For memory operations that can be analyzed to an identified underlying
  // object, we maintain one frontier per underlying object. We expect the
  // analyzable frontiers to remain small and benefit from alias analysis, so
  // only clear when they grow too large.
  MapVector<ValueType, MemopFrontier> AnalyzableFrontiers;

  /// We maintain an additional frontier of unanalyzable memory operations,
  /// equipped with some extra state designed to clear the frontier more
  /// frequently. The conditions we choose to update the sequencing store are as
  /// follows:
  /// 1. If an incoming store instruction refers to an underlying object outside
  ///    of the set of underlying objects of the current sequencing store;
  /// 2. If an incoming store instruction precedes an intervening load from an
  ///    underlying object outside of the set of underlying objects of the
  ///    current sequencing store (a 'sequencing load').
  /// In these cases, a BasicAA query will (generally) already induce a
  /// (transitive) edge from the incoming store to the current sequencing store.
  /// However, this process introduces spurious (transitive) barrier edges in
  /// the following cases:
  /// 1. From all preceding loads to the new sequencing store;
  /// 2. From the new sequencing store to all stores in the current frontier;
  /// 3. From all preceding unanalyzable stores to (a subset of) the same base
  ///    objects, independent of AA results;
  /// 4. From all preceding analyzable stores, independent of AA results.
  struct UnanalyzableMemopFrontier {
    /// The underlying memop frontier.
    MemopFrontier Frontier;
    /// The underlying objects of the current sequencing store.
    UnderlyingObjectsVector BaseObjects;
    /// `true` if the frontier contains a load whose underlying objects escape
    /// #BaseObjects.
    bool SeenSequencingLoad = false;

    /// Given a store instruction with underlying objects \p Objs, returns true
    /// if the store should be promoted to a sequencing store and false
    /// otherwise.
    bool shouldUpdate(const UnderlyingObjectsVector &Objs);

    /// Returns true if any of \p Objs does not belong to #BaseObjects.
    bool escapesBaseObjects(const UnderlyingObjectsVector &Objs);

    /// Clears the frontier and resets the associated state.
    void clear();
  };

  /// Track a frontier of unanalyzable memory operations.
  ///
  /// FIXME(@cofibrant): For some platforms, a single frontier is too coarse and
  /// we should be maintaining one frontier per address space.
  UnanalyzableMemopFrontier UnanalyzableFrontier;

  /// Remember a generic side-effecting instruction as we proceed.
  /// No other SU ever gets scheduled around it (except in the special
  /// case of a huge region that gets reduced).
  SUnit *BarrierChain = nullptr;

  /// The total number of memops amongst all frontiers.
  unsigned TotalMemops = 0;

public:
  ScheduleDAGDependencyBuilder(ScheduleDAGInstrs &DAG, BatchAAResults *AA,
                               RegPressureTracker *RPTracker,
                               PressureDiffs *PDiffs, LiveIntervals *LIS)
      : DAG(DAG), AA(AA), RPTracker(RPTracker), PDiffs(PDiffs), LIS(LIS) {}

private:
  /// Adds a chain edge between SUa and SUb, but only if both
  /// AAResults and Target fail to deny the dependency.
  ///
  /// Returns true if an edge was inserted.
  bool addChainDependency(SUnit *SUa, SUnit *SUb, unsigned Latency = 0);

  /// Adds dependencies as needed from all SUs in list to SU.
  void addChainDependencies(SUnit *SU, SUList &SUs, unsigned Latency);
  void addChainDependencies(SUnit *SU, MemopFrontier &F, bool IsStore);

  void addBarrierChain(SUList &SUs);
  void addBarrierChain(MemopFrontier &F);
  void addBarrierChain(UnanalyzableMemopFrontier &UF);

  /// Promote the store \p SU to be a new sequencing store in the frontier \p F.
  /// If \p PreserveNoAliasLoads is true, loads in the frontier that do not
  /// alias with \p SU will be retained after promotion.
  void updateSequencingStore(MemopFrontier &F, SUnit *SU,
                             bool PreserveNoAliasLoads);
  /// Promote the store \p SU with underlying objects \p Objs to be a new
  /// sequencing store in the unanalyzable frontier \p UF.
  void updateSequencingStore(UnanalyzableMemopFrontier &UF, SUnit *SU,
                             UnderlyingObjectsVector &&Objs);

public:
  void buildDeps();
};
} // end namespace llvm

unsigned ScheduleDAGDependencyBuilder::MemopFrontier::size() const {
  return Loads.size() + Stores.size();
}

void ScheduleDAGDependencyBuilder::MemopFrontier::clear() {
  SequencingStore = nullptr;
  Stores.clear();
  Loads.clear();
}

bool ScheduleDAGDependencyBuilder::UnanalyzableMemopFrontier::shouldUpdate(
    const UnderlyingObjectsVector &Objs) {
  return !Frontier.SequencingStore || SeenSequencingLoad ||
         escapesBaseObjects(Objs) || Frontier.size() >= MaxFrontierSize;
}

bool ScheduleDAGDependencyBuilder::UnanalyzableMemopFrontier::
    escapesBaseObjects(const UnderlyingObjectsVector &Objs) {
  for (const ValueType V : Objs) {
    if (!is_contained(BaseObjects, V))
      return true;
  }

  return false;
}

void ScheduleDAGDependencyBuilder::UnanalyzableMemopFrontier::clear() {
  Frontier.clear();
  SeenSequencingLoad = false;
  BaseObjects.clear();
}

bool ScheduleDAGDependencyBuilder::addChainDependency(SUnit *SUa, SUnit *SUb,
                                                      unsigned Latency) {
  if (!SUa->getInstr()->mayAlias(AA, *SUb->getInstr(), UseTBAA))
    return false;

  SDep Dep(SUa, SDep::MayAliasMem);
  Dep.setLatency(Latency);
  SUb->addPred(Dep);
  return true;
}

void ScheduleDAGDependencyBuilder::addChainDependencies(SUnit *SU, SUList &SUs,
                                                        unsigned Latency) {
  for (SUnit *Entry : SUs)
    addChainDependency(SU, Entry, Latency);
}

void ScheduleDAGDependencyBuilder::addChainDependencies(SUnit *SU,
                                                        MemopFrontier &F,
                                                        bool IsStore) {
  if (F.SequencingStore)
    F.SequencingStore->addPredBarrier(SU);

  addChainDependencies(SU, F.Stores, /*Latency=*/0);
  if (IsStore)
    addChainDependencies(SU, F.Loads, /*Latency=*/1);
}

void ScheduleDAGDependencyBuilder::addBarrierChain(SUList &SUs) {
  assert(BarrierChain != nullptr);

  for (auto *SU : SUs)
    SU->addPredBarrier(BarrierChain);

  SUs.clear();
}

void ScheduleDAGDependencyBuilder::addBarrierChain(MemopFrontier &F) {
  assert(BarrierChain != nullptr);

  addBarrierChain(F.Stores);
  addBarrierChain(F.Loads);

  if (F.SequencingStore)
    F.SequencingStore->addPredBarrier(BarrierChain);

  F.clear();
}

void ScheduleDAGDependencyBuilder::addBarrierChain(
    UnanalyzableMemopFrontier &UF) {
  assert(BarrierChain != nullptr);

  addBarrierChain(UF.Frontier);
  UF.clear();
}

void ScheduleDAGDependencyBuilder::updateSequencingStore(
    MemopFrontier &F, SUnit *SU, bool PreserveNoAliasLoads) {
  assert(SU->getInstr()->mayStore() &&
         "Only store instructions should be used as sequencing stores");
  LLVM_DEBUG(dbgs() << "Promoting " << *SU << " to sequencing store\n");

  TotalMemops -= F.size() + (F.SequencingStore ? 1 : 0);

  if (F.SequencingStore)
    F.SequencingStore->addPredBarrier(SU);

  // Sequence all stores against the sequencing store and clear the stores.
  for (SUnit *S : F.Stores)
    S->addPredBarrier(SU);

  F.Stores.clear();

  if (PreserveNoAliasLoads && F.size() < MaxFrontierSize) {
    // If asked, rather than naively adding pred barriers to all loads and
    // clearing the map, we check whether each load genuinely needs to sequence
    // against the new sequencing store. Wherever an edge is not required, we
    // retain the load and avoid the spurious control dependency.
    for (auto It = F.Loads.begin(); It != F.Loads.end();) {
      if (addChainDependency(SU, *It, /*TrueMemOrderLatency=*/1))
        It = F.Loads.erase(It);
      else
        ++It;
    }
  } else {
    // Otherwise, sequence all loads against the sequencing store and clear the
    // loads.
    for (SUnit *S : F.Loads)
      S->addPredBarrier(SU);

    F.Loads.clear();
  }

  F.SequencingStore = SU;

  TotalMemops += F.size() + 1;
}

void ScheduleDAGDependencyBuilder::updateSequencingStore(
    UnanalyzableMemopFrontier &UF, SUnit *SU, UnderlyingObjectsVector &&Objs) {
  assert(SU->getInstr()->mayStore() &&
         "Only store instructions should be used as sequencing stores");
  updateSequencingStore(UF.Frontier, SU, /*PreserveNoAliasLoads=*/true);
  UF.BaseObjects = std::move(Objs);
  UF.SeenSequencingLoad = false;
}

void ScheduleDAGDependencyBuilder::buildDeps() {
  const TargetSubtargetInfo &ST = DAG.MF.getSubtarget();

  // We build scheduling units by walking a block's instruction list
  // from bottom to top.

  // Model data dependencies between instructions being scheduled and the
  // ExitSU.
  DAG.addSchedBarrierDeps();

  // Walk the list of instructions, from bottom moving up.
  MachineInstr *DbgMI = nullptr;
  for (MachineBasicBlock::iterator MII = DAG.RegionEnd, MIE = DAG.RegionBegin;
       MII != MIE; --MII) {
    MachineInstr &MI = *std::prev(MII);
    if (DbgMI) {
      DAG.DbgValues.emplace_back(DbgMI, &MI);
      DbgMI = nullptr;
    }

    if (MI.isDebugValue() || MI.isDebugPHI()) {
      DbgMI = &MI;
      continue;
    }

    if (MI.isDebugLabel() || MI.isDebugRef() || MI.isPseudoProbe())
      continue;

    SUnit *SU = DAG.MISUnitMap[&MI];
    assert(SU && "No SUnit mapped to this MI");

    if (RPTracker) {
      RegisterOperands RegOpers;
      RegOpers.collect(MI, *DAG.TRI, DAG.MRI, DAG.TrackLaneMasks, false);
      if (DAG.TrackLaneMasks) {
        SlotIndex SlotIdx = LIS->getInstructionIndex(MI);
        RegOpers.adjustLaneLiveness(*LIS, DAG.MRI, SlotIdx);
      }
      if (PDiffs != nullptr)
        PDiffs->addInstruction(SU->NodeNum, RegOpers, DAG.MRI);

      if (RPTracker->getPos() == DAG.RegionEnd || &*RPTracker->getPos() != &MI)
        RPTracker->recedeSkipDebugValues();
      assert(&*RPTracker->getPos() == &MI && "RPTracker in sync");
      RPTracker->recede(RegOpers);
    }

    assert((DAG.CanHandleTerminators ||
            (!MI.isTerminator() && !MI.isPosition())) &&
           "Cannot schedule terminators or labels!");

    // Add register-based dependencies (data, anti, and output).
    // For some instructions (calls, returns, inline-asm, etc.) there can
    // be explicit uses and implicit defs, in which case the use will appear
    // on the operand list before the def. Do two passes over the operand
    // list to make sure that defs are processed before any uses.
    bool HasVRegDef = false;
    for (unsigned j = 0, n = MI.getNumOperands(); j != n; ++j) {
      const MachineOperand &MO = MI.getOperand(j);
      if (!MO.isReg() || !MO.isDef())
        continue;
      Register Reg = MO.getReg();
      if (Reg.isPhysical()) {
        DAG.addPhysRegDeps(SU, j);
      } else if (Reg.isVirtual()) {
        HasVRegDef = true;
        DAG.addVRegDefDeps(SU, j);
      }
    }
    // Now process all uses.
    for (unsigned j = 0, n = MI.getNumOperands(); j != n; ++j) {
      const MachineOperand &MO = MI.getOperand(j);
      // Only look at use operands.
      // We do not need to check for MO.readsReg() here because subsequent
      // subregister defs will get output dependence edges and need no
      // additional use dependencies.
      if (!MO.isReg() || !MO.isUse())
        continue;
      Register Reg = MO.getReg();
      if (Reg.isPhysical()) {
        DAG.addPhysRegDeps(SU, j);
      } else if (Reg.isVirtual() && MO.readsReg()) {
        DAG.addVRegUseDeps(SU, j);
      }
    }

    // If we haven't seen any uses in this scheduling region, create a
    // dependence edge to ExitSU to model the live-out latency. This is required
    // for vreg defs with no in-region use, and prefetches with no vreg def.
    //
    // FIXME: NumDataSuccs would be more precise than NumSuccs here. This
    // check currently relies on being called before adding chain deps.
    if (SU->NumSuccs == 0 && SU->Latency > 1 && (HasVRegDef || MI.mayLoad())) {
      SDep Dep(SU, SDep::Artificial);
      Dep.setLatency(SU->Latency - 1);
      DAG.ExitSU.addPred(Dep);
    }

    // Add memory dependencies (Note: isStoreToStackSlot and
    // isLoadFromStackSLot are not usable after stack slots are lowered to
    // actual addresses).

    const TargetInstrInfo *TII = ST.getInstrInfo();
    // This is a barrier event that acts as a pivotal node in the DAG.
    if (TII->isGlobalMemoryObject(&MI)) {

      // Become the barrier chain.
      if (BarrierChain)
        BarrierChain->addPredBarrier(SU);
      BarrierChain = SU;

      LLVM_DEBUG(dbgs() << "Global memory object and new barrier chain: "
                        << *BarrierChain << ".\n");

      // Add dependencies against everything below SU and clear state.
      addBarrierChain(FPExceptions);
      addBarrierChain(UnanalyzableFrontier);
      for (auto &[_, AF] : AnalyzableFrontiers)
        addBarrierChain(AF);
      AnalyzableFrontiers.clear();
      TotalMemops = 0;
      continue;
    }

    if (MI.mayRaiseFPException()) {
      // Instructions that may raise FP exceptions may not be moved
      // across any global barriers.
      if (BarrierChain)
        BarrierChain->addPredBarrier(SU);
      FPExceptions.push_back(SU);
    }

    // If it's not a store or a variant load, we're done.
    if (!MI.mayStore() &&
        !(MI.mayLoad() && !MI.isDereferenceableInvariantLoad()))
      continue;

    // Always add dependecy edge to BarrierChain if present.
    if (BarrierChain)
      BarrierChain->addPredBarrier(SU);

    // If our frontiers exceed the maximum size, promote SU to the be the
    // barrier chain and clear the frontiers.
    if (TotalMemops > MaxTotalMemops) {
      BarrierChain = SU;
      addBarrierChain(UnanalyzableFrontier);
      for (auto &[_, AF] : AnalyzableFrontiers)
        addBarrierChain(AF);
      AnalyzableFrontiers.clear();
      TotalMemops = 0;
      continue;
    }

    // Find the underlying objects for SU. The Objs vector is either
    // empty, or filled with the Values of memory locations which this
    // SU depends on.
    UnderlyingObjectsVector Objs;
    bool ObjsIdentified = getUnderlyingObjectsForInstr(
        SU->getInstr(), DAG.MFI, Objs, DAG.MF.getDataLayout());
    bool IsStore = SU->getInstr()->mayStore();

    if (!ObjsIdentified) {
      // An unanalyzable memory operation depends on all analyzable frontiers.
      for (auto &[_, AF] : AnalyzableFrontiers)
        addChainDependencies(SU, AF, IsStore);

      if (IsStore && UnanalyzableFrontier.shouldUpdate(Objs)) {
        updateSequencingStore(UnanalyzableFrontier, SU, std::move(Objs));
      } else {
        if (!IsStore && UnanalyzableFrontier.escapesBaseObjects(Objs))
          UnanalyzableFrontier.SeenSequencingLoad = true;
        addChainDependencies(SU, UnanalyzableFrontier.Frontier, IsStore);
        (IsStore ? UnanalyzableFrontier.Frontier.Stores
                 : UnanalyzableFrontier.Frontier.Loads)
            .push_back(SU);
        ++TotalMemops;
      }
    } else {
      // Add precise dependencies against all previously seen memory
      // accesses mapped to the same Value(s).
      SmallSet<ValueType, 2> SeenValues;
      for (const ValueType V : Objs) {
        // Deduplicate values to avoid adding self loops.
        if (!SeenValues.insert(V).second)
          continue;

        MemopFrontier &Frontier = AnalyzableFrontiers[V];

        if (IsStore && Frontier.size() >= MaxFrontierSize) {
          updateSequencingStore(Frontier, SU, false);
          continue;
        }

        addChainDependencies(SU, Frontier, IsStore);
        (IsStore ? Frontier.Stores : Frontier.Loads).push_back(SU);
        ++TotalMemops;
      }

      // The memop may have dependencies to unanalyzable stores.
      addChainDependencies(SU, UnanalyzableFrontier.Frontier, IsStore);
    }
  }

  if (DbgMI)
    DAG.FirstDbgValue = DbgMI;
}

void ScheduleDAGInstrs::buildSchedGraph(AAResults *AA,
                                        RegPressureTracker *RPTracker,
                                        PressureDiffs *PDiffs,
                                        LiveIntervals *LIS,
                                        bool TrackLaneMasks) {
  const TargetSubtargetInfo &ST = MF.getSubtarget();
  bool UseAA =
      EnableAASchedMI.getNumOccurrences() > 0 ? EnableAASchedMI : ST.useAA();
  this->TrackLaneMasks = TrackLaneMasks;
  MISUnitMap.clear();
  ScheduleDAG::clearDAG();

  // Create an SUnit for each real instruction.
  initSUnits();

  if (PDiffs)
    PDiffs->init(SUnits.size());

  // Remove any stale debug info; sometimes BuildSchedGraph is called again
  // without emitting the info from the previous call.
  DbgValues.clear();
  FirstDbgValue = nullptr;

  assert(Defs.empty() && Uses.empty() &&
         "Only BuildGraph should update Defs/Uses");
  Defs.setUniverse(TRI->getNumRegs());
  Uses.setUniverse(TRI->getNumRegs());

  assert(CurrentVRegDefs.empty() && "nobody else should use CurrentVRegDefs");
  assert(CurrentVRegUses.empty() && "nobody else should use CurrentVRegUses");
  unsigned NumVirtRegs = MRI.getNumVirtRegs();
  CurrentVRegDefs.setUniverse(NumVirtRegs);
  CurrentVRegUses.setUniverse(NumVirtRegs);

  std::optional<BatchAAResults> BatchAA;
  if (UseAA && AA)
    BatchAA.emplace(*AA);

  ScheduleDAGDependencyBuilder DepBuilder(
      *this, BatchAA.has_value() ? &BatchAA.value() : nullptr, RPTracker,
      PDiffs, LIS);
  DepBuilder.buildDeps();

  Defs.clear();
  Uses.clear();
  CurrentVRegDefs.clear();
  CurrentVRegUses.clear();

  Topo.MarkDirty();
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PseudoSourceValue* PSV) {
  PSV->printCustom(OS);
  return OS;
}

static void toggleKills(const MachineRegisterInfo &MRI, LiveRegUnits &LiveRegs,
                        MachineInstr &MI, bool addToLiveRegs) {
  for (MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.readsReg())
      continue;
    Register Reg = MO.getReg();
    if (!Reg)
      continue;

    // Things that are available after the instruction are killed by it.
    bool IsKill = LiveRegs.available(Reg);

    // Exception: Do not kill reserved registers
    MO.setIsKill(IsKill && !MRI.isReserved(Reg));
    if (addToLiveRegs)
      LiveRegs.addReg(Reg);
  }
}

void ScheduleDAGInstrs::fixupKills(MachineBasicBlock &MBB) {
  LLVM_DEBUG(dbgs() << "Fixup kills for " << printMBBReference(MBB) << '\n');

  LiveRegs.init(*TRI);
  LiveRegs.addLiveOuts(MBB);

  // Examine block from end to start...
  for (MachineInstr &MI : llvm::reverse(MBB)) {
    if (MI.isDebugOrPseudoInstr())
      continue;

    // Update liveness.  Registers that are defed but not used in this
    // instruction are now dead. Mark register and all subregs as they
    // are completely defined.
    for (ConstMIBundleOperands O(MI); O.isValid(); ++O) {
      const MachineOperand &MO = *O;
      if (MO.isReg()) {
        if (!MO.isDef())
          continue;
        Register Reg = MO.getReg();
        if (!Reg)
          continue;
        LiveRegs.removeReg(Reg);
      } else if (MO.isRegMask()) {
        LiveRegs.removeRegsNotPreserved(MO.getRegMask());
      }
    }

    // If there is a bundle header fix it up first.
    if (!MI.isBundled()) {
      toggleKills(MRI, LiveRegs, MI, true);
    } else {
      MachineBasicBlock::instr_iterator Bundle = MI.getIterator();
      if (MI.isBundle())
        toggleKills(MRI, LiveRegs, MI, false);

      // Some targets make the (questionable) assumtion that the instructions
      // inside the bundle are ordered and consequently only the last use of
      // a register inside the bundle can kill it.
      MachineBasicBlock::instr_iterator I = std::next(Bundle);
      while (I->isBundledWithSucc())
        ++I;
      do {
        if (!I->isDebugOrPseudoInstr())
          toggleKills(MRI, LiveRegs, *I, true);
        --I;
      } while (I != Bundle);
    }
  }
}

void ScheduleDAGInstrs::dumpNode(const SUnit &SU) const {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  dumpNodeName(SU);
  if (SchedPrintCycles)
    dbgs() << " [TopReadyCycle = " << SU.TopReadyCycle
           << ", BottomReadyCycle = " << SU.BotReadyCycle << "]";
  dbgs() << ": ";
  SU.getInstr()->dump();
#endif
}

void ScheduleDAGInstrs::dump() const {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  if (EntrySU.getInstr() != nullptr)
    dumpNodeAll(EntrySU);
  for (const SUnit &SU : SUnits)
    dumpNodeAll(SU);
  if (ExitSU.getInstr() != nullptr)
    dumpNodeAll(ExitSU);
#endif
}

#if !defined(NDEBUG) && LLVM_ENABLE_ABI_BREAKING_CHECKS
std::string ScheduleDAGInstrs::getGraphNodeLabel(const SUnit *SU) const {
  std::string s;
  raw_string_ostream oss(s);
  if (SU == &EntrySU)
    oss << "<entry>";
  else if (SU == &ExitSU)
    oss << "<exit>";
  else
    SU->getInstr()->print(oss, /*IsStandalone=*/true);
  return s;
}
#endif

/// Return the basic block label. It is not necessarily unique because a block
/// contains multiple scheduling regions. But it is fine for visualization.
std::string ScheduleDAGInstrs::getDAGName() const {
  return "dag." + BB->getFullName();
}

bool ScheduleDAGInstrs::canAddEdge(SUnit *SuccSU, SUnit *PredSU) {
  return SuccSU == &ExitSU || !Topo.IsReachable(PredSU, SuccSU);
}

bool ScheduleDAGInstrs::addEdge(SUnit *SuccSU, const SDep &PredDep) {
  if (SuccSU != &ExitSU) {
    // Do not use WillCreateCycle, it assumes SD scheduling.
    // If Pred is reachable from Succ, then the edge creates a cycle.
    if (Topo.IsReachable(PredDep.getSUnit(), SuccSU))
      return false;
    Topo.AddPredQueued(SuccSU, PredDep.getSUnit());
  }
  SuccSU->addPred(PredDep, /*Required=*/!PredDep.isArtificial());
  // Return true regardless of whether a new edge needed to be inserted.
  return true;
}

//===----------------------------------------------------------------------===//
// SchedDFSResult Implementation
//===----------------------------------------------------------------------===//

namespace llvm {

/// Internal state used to compute SchedDFSResult.
class SchedDFSImpl {
  SchedDFSResult &R;

  /// Join DAG nodes into equivalence classes by their subtree.
  IntEqClasses SubtreeClasses;
  /// List PredSU, SuccSU pairs that represent data edges between subtrees.
  std::vector<std::pair<const SUnit *, const SUnit*>> ConnectionPairs;

  struct RootData {
    unsigned NodeID;
    unsigned ParentNodeID;  ///< Parent node (member of the parent subtree).
    unsigned SubInstrCount = 0; ///< Instr count in this tree only, not
                                /// children.

    RootData(unsigned id): NodeID(id),
                           ParentNodeID(SchedDFSResult::InvalidSubtreeID) {}

    unsigned getSparseSetIndex() const { return NodeID; }
  };

  SparseSet<RootData> RootSet;

public:
  SchedDFSImpl(SchedDFSResult &r): R(r), SubtreeClasses(R.DFSNodeData.size()) {
    RootSet.setUniverse(R.DFSNodeData.size());
  }

  /// Returns true if this node been visited by the DFS traversal.
  ///
  /// During visitPostorderNode the Node's SubtreeID is assigned to the Node
  /// ID. Later, SubtreeID is updated but remains valid.
  bool isVisited(const SUnit *SU) const {
    return R.DFSNodeData[SU->NodeNum].SubtreeID
      != SchedDFSResult::InvalidSubtreeID;
  }

  /// Initializes this node's instruction count. We don't need to flag the node
  /// visited until visitPostorder because the DAG cannot have cycles.
  void visitPreorder(const SUnit *SU) {
    R.DFSNodeData[SU->NodeNum].InstrCount =
      SU->getInstr()->isTransient() ? 0 : 1;
  }

  /// Called once for each node after all predecessors are visited. Revisit this
  /// node's predecessors and potentially join them now that we know the ILP of
  /// the other predecessors.
  void visitPostorderNode(const SUnit *SU) {
    // Mark this node as the root of a subtree. It may be joined with its
    // successors later.
    R.DFSNodeData[SU->NodeNum].SubtreeID = SU->NodeNum;
    RootData RData(SU->NodeNum);
    RData.SubInstrCount = SU->getInstr()->isTransient() ? 0 : 1;

    // If any predecessors are still in their own subtree, they either cannot be
    // joined or are large enough to remain separate. If this parent node's
    // total instruction count is not greater than a child subtree by at least
    // the subtree limit, then try to join it now since splitting subtrees is
    // only useful if multiple high-pressure paths are possible.
    unsigned InstrCount = R.DFSNodeData[SU->NodeNum].InstrCount;
    for (const SDep &PredDep : SU->Preds) {
      if (PredDep.getKind() != SDep::Data)
        continue;
      unsigned PredNum = PredDep.getSUnit()->NodeNum;
      if ((InstrCount - R.DFSNodeData[PredNum].InstrCount) < R.SubtreeLimit)
        joinPredSubtree(PredDep, SU, /*CheckLimit=*/false);

      // Either link or merge the TreeData entry from the child to the parent.
      if (R.DFSNodeData[PredNum].SubtreeID == PredNum) {
        // If the predecessor's parent is invalid, this is a tree edge and the
        // current node is the parent.
        if (RootSet[PredNum].ParentNodeID == SchedDFSResult::InvalidSubtreeID)
          RootSet[PredNum].ParentNodeID = SU->NodeNum;
      }
      else if (RootSet.count(PredNum)) {
        // The predecessor is not a root, but is still in the root set. This
        // must be the new parent that it was just joined to. Note that
        // RootSet[PredNum].ParentNodeID may either be invalid or may still be
        // set to the original parent.
        RData.SubInstrCount += RootSet[PredNum].SubInstrCount;
        RootSet.erase(PredNum);
      }
    }
    RootSet[SU->NodeNum] = RData;
  }

  /// Called once for each tree edge after calling visitPostOrderNode on
  /// the predecessor. Increment the parent node's instruction count and
  /// preemptively join this subtree to its parent's if it is small enough.
  void visitPostorderEdge(const SDep &PredDep, const SUnit *Succ) {
    R.DFSNodeData[Succ->NodeNum].InstrCount
      += R.DFSNodeData[PredDep.getSUnit()->NodeNum].InstrCount;
    joinPredSubtree(PredDep, Succ);
  }

  /// Adds a connection for cross edges.
  void visitCrossEdge(const SDep &PredDep, const SUnit *Succ) {
    ConnectionPairs.emplace_back(PredDep.getSUnit(), Succ);
  }

  /// Sets each node's subtree ID to the representative ID and record
  /// connections between trees.
  void finalize() {
    SubtreeClasses.compress();
    R.DFSTreeData.resize(SubtreeClasses.getNumClasses());
    assert(SubtreeClasses.getNumClasses() == RootSet.size()
           && "number of roots should match trees");
    for (const RootData &Root : RootSet) {
      unsigned TreeID = SubtreeClasses[Root.NodeID];
      if (Root.ParentNodeID != SchedDFSResult::InvalidSubtreeID)
        R.DFSTreeData[TreeID].ParentTreeID = SubtreeClasses[Root.ParentNodeID];
      R.DFSTreeData[TreeID].SubInstrCount = Root.SubInstrCount;
      // Note that SubInstrCount may be greater than InstrCount if we joined
      // subtrees across a cross edge. InstrCount will be attributed to the
      // original parent, while SubInstrCount will be attributed to the joined
      // parent.
    }
    R.SubtreeConnections.resize(SubtreeClasses.getNumClasses());
    R.SubtreeConnectLevels.resize(SubtreeClasses.getNumClasses());
    LLVM_DEBUG(dbgs() << R.getNumSubtrees() << " subtrees:\n");
    for (unsigned Idx = 0, End = R.DFSNodeData.size(); Idx != End; ++Idx) {
      R.DFSNodeData[Idx].SubtreeID = SubtreeClasses[Idx];
      LLVM_DEBUG(dbgs() << "  SU(" << Idx << ") in tree "
                        << R.DFSNodeData[Idx].SubtreeID << '\n');
    }
    for (const auto &[Pred, Succ] : ConnectionPairs) {
      unsigned PredTree = SubtreeClasses[Pred->NodeNum];
      unsigned SuccTree = SubtreeClasses[Succ->NodeNum];
      if (PredTree == SuccTree)
        continue;
      unsigned Depth = Pred->getDepth();
      addConnection(PredTree, SuccTree, Depth);
      addConnection(SuccTree, PredTree, Depth);
    }
  }

protected:
  /// Joins the predecessor subtree with the successor that is its DFS parent.
  /// Applies some heuristics before joining.
  bool joinPredSubtree(const SDep &PredDep, const SUnit *Succ,
                       bool CheckLimit = true) {
    assert(PredDep.getKind() == SDep::Data && "Subtrees are for data edges");

    // Check if the predecessor is already joined.
    const SUnit *PredSU = PredDep.getSUnit();
    unsigned PredNum = PredSU->NodeNum;
    if (R.DFSNodeData[PredNum].SubtreeID != PredNum)
      return false;

    // Four is the magic number of successors before a node is considered a
    // pinch point.
    unsigned NumDataSucs = 0;
    for (const SDep &SuccDep : PredSU->Succs) {
      if (SuccDep.getKind() == SDep::Data) {
        if (++NumDataSucs >= 4)
          return false;
      }
    }
    if (CheckLimit && R.DFSNodeData[PredNum].InstrCount > R.SubtreeLimit)
      return false;
    R.DFSNodeData[PredNum].SubtreeID = Succ->NodeNum;
    SubtreeClasses.join(Succ->NodeNum, PredNum);
    return true;
  }

  /// Called by finalize() to record a connection between trees.
  void addConnection(unsigned FromTree, unsigned ToTree, unsigned Depth) {
    if (!Depth)
      return;

    do {
      SmallVectorImpl<SchedDFSResult::Connection> &Connections =
        R.SubtreeConnections[FromTree];
      for (SchedDFSResult::Connection &C : Connections) {
        if (C.TreeID == ToTree) {
          C.Level = std::max(C.Level, Depth);
          return;
        }
      }
      Connections.push_back(SchedDFSResult::Connection(ToTree, Depth));
      FromTree = R.DFSTreeData[FromTree].ParentTreeID;
    } while (FromTree != SchedDFSResult::InvalidSubtreeID);
  }
};

} // end namespace llvm

namespace {

/// Manage the stack used by a reverse depth-first search over the DAG.
class SchedDAGReverseDFS {
  std::vector<std::pair<const SUnit *, SUnit::const_pred_iterator>> DFSStack;

public:
  bool isComplete() const { return DFSStack.empty(); }

  void follow(const SUnit *SU) {
    DFSStack.emplace_back(SU, SU->Preds.begin());
  }
  void advance() { ++DFSStack.back().second; }

  const SDep *backtrack() {
    DFSStack.pop_back();
    return DFSStack.empty() ? nullptr : std::prev(DFSStack.back().second);
  }

  const SUnit *getCurr() const { return DFSStack.back().first; }

  SUnit::const_pred_iterator getPred() const { return DFSStack.back().second; }

  SUnit::const_pred_iterator getPredEnd() const {
    return getCurr()->Preds.end();
  }
};

} // end anonymous namespace

static bool hasDataSucc(const SUnit *SU) {
  for (const SDep &SuccDep : SU->Succs) {
    if (SuccDep.getKind() == SDep::Data &&
        !SuccDep.getSUnit()->isBoundaryNode())
      return true;
  }
  return false;
}

/// Computes an ILP metric for all nodes in the subDAG reachable via depth-first
/// search from this root.
void SchedDFSResult::compute(ArrayRef<SUnit> SUnits) {
  if (!IsBottomUp)
    llvm_unreachable("Top-down ILP metric is unimplemented");

  SchedDFSImpl Impl(*this);
  for (const SUnit &SU : SUnits) {
    if (Impl.isVisited(&SU) || hasDataSucc(&SU))
      continue;

    SchedDAGReverseDFS DFS;
    Impl.visitPreorder(&SU);
    DFS.follow(&SU);
    while (true) {
      // Traverse the leftmost path as far as possible.
      while (DFS.getPred() != DFS.getPredEnd()) {
        const SDep &PredDep = *DFS.getPred();
        DFS.advance();
        // Ignore non-data edges.
        if (PredDep.getKind() != SDep::Data
            || PredDep.getSUnit()->isBoundaryNode()) {
          continue;
        }
        // An already visited edge is a cross edge, assuming an acyclic DAG.
        if (Impl.isVisited(PredDep.getSUnit())) {
          Impl.visitCrossEdge(PredDep, DFS.getCurr());
          continue;
        }
        Impl.visitPreorder(PredDep.getSUnit());
        DFS.follow(PredDep.getSUnit());
      }
      // Visit the top of the stack in postorder and backtrack.
      const SUnit *Child = DFS.getCurr();
      const SDep *PredDep = DFS.backtrack();
      Impl.visitPostorderNode(Child);
      if (PredDep)
        Impl.visitPostorderEdge(*PredDep, DFS.getCurr());
      if (DFS.isComplete())
        break;
    }
  }
  Impl.finalize();
}

/// The root of the given SubtreeID was just scheduled. For all subtrees
/// connected to this tree, record the depth of the connection so that the
/// nearest connected subtrees can be prioritized.
void SchedDFSResult::scheduleTree(unsigned SubtreeID) {
  for (const Connection &C : SubtreeConnections[SubtreeID]) {
    SubtreeConnectLevels[C.TreeID] =
      std::max(SubtreeConnectLevels[C.TreeID], C.Level);
    LLVM_DEBUG(dbgs() << "  Tree: " << C.TreeID << " @"
                      << SubtreeConnectLevels[C.TreeID] << '\n');
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void ILPValue::print(raw_ostream &OS) const {
  OS << InstrCount << " / " << Length << " = ";
  if (!Length)
    OS << "BADILP";
  else
    OS << format("%g", ((double)InstrCount / Length));
}

LLVM_DUMP_METHOD void ILPValue::dump() const {
  dbgs() << *this << '\n';
}

[[maybe_unused]]
raw_ostream &llvm::operator<<(raw_ostream &OS, const ILPValue &Val) {
  Val.print(OS);
  return OS;
}

#endif
