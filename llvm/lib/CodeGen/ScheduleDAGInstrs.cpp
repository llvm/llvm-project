//===---- ScheduleDAGInstrs.cpp - MachineInstr Rescheduling ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the ScheduleDAGInstrs class, which implements re-scheduling
// of MachineInstrs.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDFS.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#define DEBUG_TYPE "misched"

static cl::opt<bool> EnableAASchedMI("enable-aa-sched-mi", cl::Hidden,
    cl::ZeroOrMore, cl::init(false),
    cl::desc("Enable use of AA during MI DAG construction"));

static cl::opt<bool> UseTBAA("use-tbaa-in-sched-mi", cl::Hidden,
    cl::init(true), cl::desc("Enable use of TBAA during MI DAG construction"));

// Note: the two options below might be used in tuning compile time vs
// output quality. Setting HugeRegion so large that it will never be
// reached means best-effort, but may be slow.

// When Stores and Loads maps (or NonAliasStores and NonAliasLoads)
// together hold this many SUs, a reduction of maps will be done.
static cl::opt<unsigned> HugeRegion("dag-maps-huge-region", cl::Hidden,
    cl::init(1000), cl::desc("The limit to use while constructing the DAG "
                             "prior to scheduling, at which point a trade-off "
                             "is made to avoid excessive compile time."));

static cl::opt<unsigned> ReductionSize(
    "dag-maps-reduction-size", cl::Hidden,
    cl::desc("A huge scheduling region will have maps reduced by this many "
             "nodes at a time. Defaults to HugeRegion / 2."));

static unsigned getReductionSize() {
  // Always reduce a huge region with half of the elements, except
  // when user sets this number explicitly.
  if (ReductionSize.getNumOccurrences() == 0)
    return HugeRegion / 2;
  return ReductionSize;
}

static void dumpSUList(ScheduleDAGInstrs::SUList &L) {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  dbgs() << "{ ";
  for (auto *su : L) {
    dbgs() << "SU(" << su->NodeNum << ")";
    if (su != L.back())
      dbgs() << ", ";
  }
  dbgs() << "}\n";
#endif
}

ScheduleDAGInstrs::ScheduleDAGInstrs(MachineFunction &mf,
                                     const MachineLoopInfo *mli,
                                     bool RemoveKillFlags)
    : ScheduleDAG(mf), MLI(mli), MFI(mf.getFrameInfo()),
      RemoveKillFlags(RemoveKillFlags), CanHandleTerminators(false),
      TrackLaneMasks(false), AAForDep(nullptr), BarrierChain(nullptr),
      UnknownValue(UndefValue::get(
                     Type::getVoidTy(mf.getFunction()->getContext()))),
      FirstDbgValue(nullptr) {
  DbgValues.clear();

  const TargetSubtargetInfo &ST = mf.getSubtarget();
  SchedModel.init(ST.getSchedModel(), &ST, TII);
}

/// getUnderlyingObjectFromInt - This is the function that does the work of
/// looking through basic ptrtoint+arithmetic+inttoptr sequences.
static const Value *getUnderlyingObjectFromInt(const Value *V) {
  do {
    if (const Operator *U = dyn_cast<Operator>(V)) {
      // If we find a ptrtoint, we can transfer control back to the
      // regular getUnderlyingObjectFromInt.
      if (U->getOpcode() == Instruction::PtrToInt)
        return U->getOperand(0);
      // If we find an add of a constant, a multiplied value, or a phi, it's
      // likely that the other operand will lead us to the base
      // object. We don't have to worry about the case where the
      // object address is somehow being computed by the multiply,
      // because our callers only care when the result is an
      // identifiable object.
      if (U->getOpcode() != Instruction::Add ||
          (!isa<ConstantInt>(U->getOperand(1)) &&
           Operator::getOpcode(U->getOperand(1)) != Instruction::Mul &&
           !isa<PHINode>(U->getOperand(1))))
        return V;
      V = U->getOperand(0);
    } else {
      return V;
    }
    assert(V->getType()->isIntegerTy() && "Unexpected operand type!");
  } while (1);
}

/// getUnderlyingObjects - This is a wrapper around GetUnderlyingObjects
/// and adds support for basic ptrtoint+arithmetic+inttoptr sequences.
static void getUnderlyingObjects(const Value *V,
                                 SmallVectorImpl<Value *> &Objects,
                                 const DataLayout &DL) {
  SmallPtrSet<const Value *, 16> Visited;
  SmallVector<const Value *, 4> Working(1, V);
  do {
    V = Working.pop_back_val();

    SmallVector<Value *, 4> Objs;
    GetUnderlyingObjects(const_cast<Value *>(V), Objs, DL);

    for (SmallVectorImpl<Value *>::iterator I = Objs.begin(), IE = Objs.end();
         I != IE; ++I) {
      V = *I;
      if (!Visited.insert(V).second)
        continue;
      if (Operator::getOpcode(V) == Instruction::IntToPtr) {
        const Value *O =
          getUnderlyingObjectFromInt(cast<User>(V)->getOperand(0));
        if (O->getType()->isPointerTy()) {
          Working.push_back(O);
          continue;
        }
      }
      Objects.push_back(const_cast<Value *>(V));
    }
  } while (!Working.empty());
}

/// getUnderlyingObjectsForInstr - If this machine instr has memory reference
/// information and it can be tracked to a normal reference to a known
/// object, return the Value for that object.
static void getUnderlyingObjectsForInstr(const MachineInstr *MI,
                                         const MachineFrameInfo *MFI,
                                         UnderlyingObjectsVector &Objects,
                                         const DataLayout &DL) {
  auto allMMOsOkay = [&]() {
    for (const MachineMemOperand *MMO : MI->memoperands()) {
      if (MMO->isVolatile())
        return false;

      if (const PseudoSourceValue *PSV = MMO->getPseudoValue()) {
        // Function that contain tail calls don't have unique PseudoSourceValue
        // objects. Two PseudoSourceValues might refer to the same or
        // overlapping locations. The client code calling this function assumes
        // this is not the case. So return a conservative answer of no known
        // object.
        if (MFI->hasTailCall())
          return false;

        // For now, ignore PseudoSourceValues which may alias LLVM IR values
        // because the code that uses this function has no way to cope with
        // such aliases.
        if (PSV->isAliased(MFI))
          return false;

        bool MayAlias = PSV->mayAlias(MFI);
        Objects.push_back(UnderlyingObjectsVector::value_type(PSV, MayAlias));
      } else if (const Value *V = MMO->getValue()) {
        SmallVector<Value *, 4> Objs;
        getUnderlyingObjects(V, Objs, DL);

        for (Value *V : Objs) {
          if (!isIdentifiedObject(V))
            return false;

          Objects.push_back(UnderlyingObjectsVector::value_type(V, true));
        }
      } else
        return false;
    }
    return true;
  };

  if (!allMMOsOkay())
    Objects.clear();
}

void ScheduleDAGInstrs::startBlock(MachineBasicBlock *bb) {
  BB = bb;
}

void ScheduleDAGInstrs::finishBlock() {
  // Subclasses should no longer refer to the old block.
  BB = nullptr;
}

/// Initialize the DAG and common scheduler state for the current scheduling
/// region. This does not actually create the DAG, only clears it. The
/// scheduling driver may call BuildSchedGraph multiple times per scheduling
/// region.
void ScheduleDAGInstrs::enterRegion(MachineBasicBlock *bb,
                                    MachineBasicBlock::iterator begin,
                                    MachineBasicBlock::iterator end,
                                    unsigned regioninstrs) {
  assert(bb == BB && "startBlock should set BB");
  RegionBegin = begin;
  RegionEnd = end;
  NumRegionInstrs = regioninstrs;
}

/// Close the current scheduling region. Don't clear any state in case the
/// driver wants to refer to the previous scheduling region.
void ScheduleDAGInstrs::exitRegion() {
  // Nothing to do.
}

/// addSchedBarrierDeps - Add dependencies from instructions in the current
/// list of instructions being scheduled to scheduling barrier by adding
/// the exit SU to the register defs and use list. This is because we want to
/// make sure instructions which define registers that are either used by
/// the terminator or are live-out are properly scheduled. This is
/// especially important when the definition latency of the return value(s)
/// are too high to be hidden by the branch or when the liveout registers
/// used by instructions in the fallthrough block.
void ScheduleDAGInstrs::addSchedBarrierDeps() {
  MachineInstr *ExitMI = RegionEnd != BB->end() ? &*RegionEnd : nullptr;
  ExitSU.setInstr(ExitMI);
  bool AllDepKnown = ExitMI &&
    (ExitMI->isCall() || ExitMI->isBarrier());
  if (ExitMI && AllDepKnown) {
    // If it's a call or a barrier, add dependencies on the defs and uses of
    // instruction.
    for (unsigned i = 0, e = ExitMI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = ExitMI->getOperand(i);
      if (!MO.isReg() || MO.isDef()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;

      if (TRI->isPhysicalRegister(Reg))
        Uses.insert(PhysRegSUOper(&ExitSU, -1, Reg));
      else if (MO.readsReg()) // ignore undef operands
        addVRegUseDeps(&ExitSU, i);
    }
  } else {
    // For others, e.g. fallthrough, conditional branch, assume the exit
    // uses all the registers that are livein to the successor blocks.
    assert(Uses.empty() && "Uses in set before adding deps?");
    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
           SE = BB->succ_end(); SI != SE; ++SI)
      for (const auto &LI : (*SI)->liveins()) {
        if (!Uses.contains(LI.PhysReg))
          Uses.insert(PhysRegSUOper(&ExitSU, -1, LI.PhysReg));
      }
  }
}

/// MO is an operand of SU's instruction that defines a physical register. Add
/// data dependencies from SU to any uses of the physical register.
void ScheduleDAGInstrs::addPhysRegDataDeps(SUnit *SU, unsigned OperIdx) {
  const MachineOperand &MO = SU->getInstr()->getOperand(OperIdx);
  assert(MO.isDef() && "expect physreg def");

  // Ask the target if address-backscheduling is desirable, and if so how much.
  const TargetSubtargetInfo &ST = MF.getSubtarget();

  for (MCRegAliasIterator Alias(MO.getReg(), TRI, true);
       Alias.isValid(); ++Alias) {
    if (!Uses.contains(*Alias))
      continue;
    for (Reg2SUnitsMap::iterator I = Uses.find(*Alias); I != Uses.end(); ++I) {
      SUnit *UseSU = I->SU;
      if (UseSU == SU)
        continue;

      // Adjust the dependence latency using operand def/use information,
      // then allow the target to perform its own adjustments.
      int UseOp = I->OpIdx;
      MachineInstr *RegUse = nullptr;
      SDep Dep;
      if (UseOp < 0)
        Dep = SDep(SU, SDep::Artificial);
      else {
        // Set the hasPhysRegDefs only for physreg defs that have a use within
        // the scheduling region.
        SU->hasPhysRegDefs = true;
        Dep = SDep(SU, SDep::Data, *Alias);
        RegUse = UseSU->getInstr();
      }
      Dep.setLatency(
        SchedModel.computeOperandLatency(SU->getInstr(), OperIdx, RegUse,
                                         UseOp));

      ST.adjustSchedDependency(SU, UseSU, Dep);
      UseSU->addPred(Dep);
    }
  }
}

/// addPhysRegDeps - Add register dependencies (data, anti, and output) from
/// this SUnit to following instructions in the same scheduling region that
/// depend the physical register referenced at OperIdx.
void ScheduleDAGInstrs::addPhysRegDeps(SUnit *SU, unsigned OperIdx) {
  MachineInstr *MI = SU->getInstr();
  MachineOperand &MO = MI->getOperand(OperIdx);

  // Optionally add output and anti dependencies. For anti
  // dependencies we use a latency of 0 because for a multi-issue
  // target we want to allow the defining instruction to issue
  // in the same cycle as the using instruction.
  // TODO: Using a latency of 1 here for output dependencies assumes
  //       there's no cost for reusing registers.
  SDep::Kind Kind = MO.isUse() ? SDep::Anti : SDep::Output;
  for (MCRegAliasIterator Alias(MO.getReg(), TRI, true);
       Alias.isValid(); ++Alias) {
    if (!Defs.contains(*Alias))
      continue;
    for (Reg2SUnitsMap::iterator I = Defs.find(*Alias); I != Defs.end(); ++I) {
      SUnit *DefSU = I->SU;
      if (DefSU == &ExitSU)
        continue;
      if (DefSU != SU &&
          (Kind != SDep::Output || !MO.isDead() ||
           !DefSU->getInstr()->registerDefIsDead(*Alias))) {
        if (Kind == SDep::Anti)
          DefSU->addPred(SDep(SU, Kind, /*Reg=*/*Alias));
        else {
          SDep Dep(SU, Kind, /*Reg=*/*Alias);
          Dep.setLatency(
            SchedModel.computeOutputLatency(MI, OperIdx, DefSU->getInstr()));
          DefSU->addPred(Dep);
        }
      }
    }
  }

  if (!MO.isDef()) {
    SU->hasPhysRegUses = true;
    // Either insert a new Reg2SUnits entry with an empty SUnits list, or
    // retrieve the existing SUnits list for this register's uses.
    // Push this SUnit on the use list.
    Uses.insert(PhysRegSUOper(SU, OperIdx, MO.getReg()));
    if (RemoveKillFlags)
      MO.setIsKill(false);
  }
  else {
    addPhysRegDataDeps(SU, OperIdx);
    unsigned Reg = MO.getReg();

    // clear this register's use list
    if (Uses.contains(Reg))
      Uses.eraseAll(Reg);

    if (!MO.isDead()) {
      Defs.eraseAll(Reg);
    } else if (SU->isCall) {
      // Calls will not be reordered because of chain dependencies (see
      // below). Since call operands are dead, calls may continue to be added
      // to the DefList making dependence checking quadratic in the size of
      // the block. Instead, we leave only one call at the back of the
      // DefList.
      Reg2SUnitsMap::RangePair P = Defs.equal_range(Reg);
      Reg2SUnitsMap::iterator B = P.first;
      Reg2SUnitsMap::iterator I = P.second;
      for (bool isBegin = I == B; !isBegin; /* empty */) {
        isBegin = (--I) == B;
        if (!I->SU->isCall)
          break;
        I = Defs.erase(I);
      }
    }

    // Defs are pushed in the order they are visited and never reordered.
    Defs.insert(PhysRegSUOper(SU, OperIdx, Reg));
  }
}

LaneBitmask ScheduleDAGInstrs::getLaneMaskForMO(const MachineOperand &MO) const
{
  unsigned Reg = MO.getReg();
  // No point in tracking lanemasks if we don't have interesting subregisters.
  const TargetRegisterClass &RC = *MRI.getRegClass(Reg);
  if (!RC.HasDisjunctSubRegs)
    return ~0u;

  unsigned SubReg = MO.getSubReg();
  if (SubReg == 0)
    return RC.getLaneMask();
  return TRI->getSubRegIndexLaneMask(SubReg);
}

/// addVRegDefDeps - Add register output and data dependencies from this SUnit
/// to instructions that occur later in the same scheduling region if they read
/// from or write to the virtual register defined at OperIdx.
///
/// TODO: Hoist loop induction variable increments. This has to be
/// reevaluated. Generally, IV scheduling should be done before coalescing.
void ScheduleDAGInstrs::addVRegDefDeps(SUnit *SU, unsigned OperIdx) {
  MachineInstr *MI = SU->getInstr();
  MachineOperand &MO = MI->getOperand(OperIdx);
  unsigned Reg = MO.getReg();

  LaneBitmask DefLaneMask;
  LaneBitmask KillLaneMask;
  if (TrackLaneMasks) {
    bool IsKill = MO.getSubReg() == 0 || MO.isUndef();
    DefLaneMask = getLaneMaskForMO(MO);
    // If we have a <read-undef> flag, none of the lane values comes from an
    // earlier instruction.
    KillLaneMask = IsKill ? ~0u : DefLaneMask;

    // Clear undef flag, we'll re-add it later once we know which subregister
    // Def is first.
    MO.setIsUndef(false);
  } else {
    DefLaneMask = ~0u;
    KillLaneMask = ~0u;
  }

  if (MO.isDead()) {
    assert(CurrentVRegUses.find(Reg) == CurrentVRegUses.end() &&
           "Dead defs should have no uses");
  } else {
    // Add data dependence to all uses we found so far.
    const TargetSubtargetInfo &ST = MF.getSubtarget();
    for (VReg2SUnitOperIdxMultiMap::iterator I = CurrentVRegUses.find(Reg),
         E = CurrentVRegUses.end(); I != E; /*empty*/) {
      LaneBitmask LaneMask = I->LaneMask;
      // Ignore uses of other lanes.
      if ((LaneMask & KillLaneMask) == 0) {
        ++I;
        continue;
      }

      if ((LaneMask & DefLaneMask) != 0) {
        SUnit *UseSU = I->SU;
        MachineInstr *Use = UseSU->getInstr();
        SDep Dep(SU, SDep::Data, Reg);
        Dep.setLatency(SchedModel.computeOperandLatency(MI, OperIdx, Use,
                                                        I->OperandIndex));
        ST.adjustSchedDependency(SU, UseSU, Dep);
        UseSU->addPred(Dep);
      }

      LaneMask &= ~KillLaneMask;
      // If we found a Def for all lanes of this use, remove it from the list.
      if (LaneMask != 0) {
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
    if ((V2SU.LaneMask & LaneMask) == 0)
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
    if (NonOverlapMask != 0)
      CurrentVRegDefs.insert(VReg2SUnit(Reg, NonOverlapMask, DefSU));
  }
  // If there was no CurrentVRegDefs entry for some lanes yet, create one.
  if (LaneMask != 0)
    CurrentVRegDefs.insert(VReg2SUnit(Reg, LaneMask, SU));
}

/// addVRegUseDeps - Add a register data dependency if the instruction that
/// defines the virtual register used at OperIdx is mapped to an SUnit. Add a
/// register antidependency from this SUnit to instructions that occur later in
/// the same scheduling region if they write the virtual register.
///
/// TODO: Handle ExitSU "uses" properly.
void ScheduleDAGInstrs::addVRegUseDeps(SUnit *SU, unsigned OperIdx) {
  const MachineInstr *MI = SU->getInstr();
  const MachineOperand &MO = MI->getOperand(OperIdx);
  unsigned Reg = MO.getReg();

  // Remember the use. Data dependencies will be added when we find the def.
  LaneBitmask LaneMask = TrackLaneMasks ? getLaneMaskForMO(MO) : ~0u;
  CurrentVRegUses.insert(VReg2SUnitOperIdx(Reg, LaneMask, OperIdx, SU));

  // Add antidependences to the following defs of the vreg.
  for (VReg2SUnit &V2SU : make_range(CurrentVRegDefs.find(Reg),
                                     CurrentVRegDefs.end())) {
    // Ignore defs for unrelated lanes.
    LaneBitmask PrevDefLaneMask = V2SU.LaneMask;
    if ((PrevDefLaneMask & LaneMask) == 0)
      continue;
    if (V2SU.SU == SU)
      continue;

    V2SU.SU->addPred(SDep(SU, SDep::Anti, Reg));
  }
}

/// Return true if MI is an instruction we are unable to reason about
/// (like a call or something with unmodeled side effects).
static inline bool isGlobalMemoryObject(AliasAnalysis *AA, MachineInstr *MI) {
  return MI->isCall() || MI->hasUnmodeledSideEffects() ||
         (MI->hasOrderedMemoryRef() && !MI->isInvariantLoad(AA));
}

/// This returns true if the two MIs need a chain edge between them.
/// This is called on normal stores and loads.
static bool MIsNeedChainEdge(AliasAnalysis *AA, const MachineFrameInfo *MFI,
                             const DataLayout &DL, MachineInstr *MIa,
                             MachineInstr *MIb) {
  const MachineFunction *MF = MIa->getParent()->getParent();
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();

  assert ((MIa->mayStore() || MIb->mayStore()) &&
          "Dependency checked between two loads");

  // Let the target decide if memory accesses cannot possibly overlap.
  if (TII->areMemAccessesTriviallyDisjoint(MIa, MIb, AA))
    return false;

  // To this point analysis is generic. From here on we do need AA.
  if (!AA)
    return true;

  // FIXME: Need to handle multiple memory operands to support all targets.
  if (!MIa->hasOneMemOperand() || !MIb->hasOneMemOperand())
    return true;

  MachineMemOperand *MMOa = *MIa->memoperands_begin();
  MachineMemOperand *MMOb = *MIb->memoperands_begin();

  if (!MMOa->getValue() || !MMOb->getValue())
    return true;

  // The following interface to AA is fashioned after DAGCombiner::isAlias
  // and operates with MachineMemOperand offset with some important
  // assumptions:
  //   - LLVM fundamentally assumes flat address spaces.
  //   - MachineOperand offset can *only* result from legalization and
  //     cannot affect queries other than the trivial case of overlap
  //     checking.
  //   - These offsets never wrap and never step outside
  //     of allocated objects.
  //   - There should never be any negative offsets here.
  //
  // FIXME: Modify API to hide this math from "user"
  // FIXME: Even before we go to AA we can reason locally about some
  // memory objects. It can save compile time, and possibly catch some
  // corner cases not currently covered.

  assert ((MMOa->getOffset() >= 0) && "Negative MachineMemOperand offset");
  assert ((MMOb->getOffset() >= 0) && "Negative MachineMemOperand offset");

  int64_t MinOffset = std::min(MMOa->getOffset(), MMOb->getOffset());
  int64_t Overlapa = MMOa->getSize() + MMOa->getOffset() - MinOffset;
  int64_t Overlapb = MMOb->getSize() + MMOb->getOffset() - MinOffset;

  AliasResult AAResult =
      AA->alias(MemoryLocation(MMOa->getValue(), Overlapa,
                               UseTBAA ? MMOa->getAAInfo() : AAMDNodes()),
                MemoryLocation(MMOb->getValue(), Overlapb,
                               UseTBAA ? MMOb->getAAInfo() : AAMDNodes()));

  return (AAResult != NoAlias);
}

/// Check whether two objects need a chain edge and add it if needed.
void ScheduleDAGInstrs::addChainDependency (SUnit *SUa, SUnit *SUb,
                                            unsigned Latency) {
  if (MIsNeedChainEdge(AAForDep, MFI, MF.getDataLayout(), SUa->getInstr(),
		       SUb->getInstr())) {
    SDep Dep(SUa, SDep::MayAliasMem);
    Dep.setLatency(Latency);
    SUb->addPred(Dep);
  }
}

/// Create an SUnit for each real instruction, numbered in top-down topological
/// order. The instruction order A < B, implies that no edge exists from B to A.
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

  for (MachineBasicBlock::iterator I = RegionBegin; I != RegionEnd; ++I) {
    MachineInstr *MI = I;
    if (MI->isDebugValue())
      continue;

    SUnit *SU = newSUnit(MI);
    MISUnitMap[MI] = SU;

    SU->isCall = MI->isCall();
    SU->isCommutable = MI->isCommutable();

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
      for (TargetSchedModel::ProcResIter
             PI = SchedModel.getWriteProcResBegin(SC),
             PE = SchedModel.getWriteProcResEnd(SC); PI != PE; ++PI) {
        switch (SchedModel.getProcResource(PI->ProcResourceIdx)->BufferSize) {
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

void ScheduleDAGInstrs::collectVRegUses(SUnit *SU) {
  const MachineInstr *MI = SU->getInstr();
  for (const MachineOperand &MO : MI->operands()) {
    if (!MO.isReg())
      continue;
    if (!MO.readsReg())
      continue;
    if (TrackLaneMasks && !MO.isUse())
      continue;

    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;

    // Ignore re-defs.
    if (TrackLaneMasks) {
      bool FoundDef = false;
      for (const MachineOperand &MO2 : MI->operands()) {
        if (MO2.isReg() && MO2.isDef() && MO2.getReg() == Reg && !MO2.isDead()) {
          FoundDef = true;
          break;
        }
      }
      if (FoundDef)
        continue;
    }

    // Record this local VReg use.
    VReg2SUnitMultiMap::iterator UI = VRegUses.find(Reg);
    for (; UI != VRegUses.end(); ++UI) {
      if (UI->SU == SU)
        break;
    }
    if (UI == VRegUses.end())
      VRegUses.insert(VReg2SUnit(Reg, 0, SU));
  }
}

class ScheduleDAGInstrs::Value2SUsMap : public MapVector<ValueType, SUList> {

  /// Current total number of SUs in map.
  unsigned NumNodes;

  /// 1 for loads, 0 for stores. (see comment in SUList)
  unsigned TrueMemOrderLatency;
public:

  Value2SUsMap(unsigned lat = 0) : NumNodes(0), TrueMemOrderLatency(lat) {}

  /// To keep NumNodes up to date, insert() is used instead of
  /// this operator w/ push_back().
  ValueType &operator[](const SUList &Key) {
    llvm_unreachable("Don't use. Use insert() instead."); };

  /// Add SU to the SUList of V. If Map grows huge, reduce its size
  /// by calling reduce().
  void inline insert(SUnit *SU, ValueType V) {
    MapVector::operator[](V).push_back(SU);
    NumNodes++;
  }

  /// Clears the list of SUs mapped to V.
  void inline clearList(ValueType V) {
    iterator Itr = find(V);
    if (Itr != end()) {
      assert (NumNodes >= Itr->second.size());
      NumNodes -= Itr->second.size();

      Itr->second.clear();
    }
  }

  /// Clears map from all contents.
  void clear() {
    MapVector<ValueType, SUList>::clear();
    NumNodes = 0;
  }

  unsigned inline size() const { return NumNodes; }

  /// Count the number of SUs in this map after a reduction.
  void reComputeSize(void) {
    NumNodes = 0;
    for (auto &I : *this)
      NumNodes += I.second.size();
  }

  unsigned inline getTrueMemOrderLatency() const {
    return TrueMemOrderLatency;
  }

  void dump();
};

void ScheduleDAGInstrs::addChainDependencies(SUnit *SU,
                                             Value2SUsMap &Val2SUsMap) {
  for (auto &I : Val2SUsMap)
    addChainDependencies(SU, I.second,
                         Val2SUsMap.getTrueMemOrderLatency());
}

void ScheduleDAGInstrs::addChainDependencies(SUnit *SU,
                                             Value2SUsMap &Val2SUsMap,
                                             ValueType V) {
  Value2SUsMap::iterator Itr = Val2SUsMap.find(V);
  if (Itr != Val2SUsMap.end())
    addChainDependencies(SU, Itr->second,
                         Val2SUsMap.getTrueMemOrderLatency());
}

void ScheduleDAGInstrs::addBarrierChain(Value2SUsMap &map) {
  assert (BarrierChain != nullptr);

  for (auto &I : map) {
    SUList &sus = I.second;
    for (auto *SU : sus)
      SU->addPredBarrier(BarrierChain);
  }
  map.clear();
}

void ScheduleDAGInstrs::insertBarrierChain(Value2SUsMap &map) {
  assert (BarrierChain != nullptr);

  // Go through all lists of SUs.
  for (Value2SUsMap::iterator I = map.begin(), EE = map.end(); I != EE;) {
    Value2SUsMap::iterator CurrItr = I++;
    SUList &sus = CurrItr->second;
    SUList::iterator SUItr = sus.begin(), SUEE = sus.end();
    for (; SUItr != SUEE; ++SUItr) {
      // Stop on BarrierChain or any instruction above it.
      if ((*SUItr)->NodeNum <= BarrierChain->NodeNum)
        break;

      (*SUItr)->addPredBarrier(BarrierChain);
    }

    // Remove also the BarrierChain from list if present.
    if (SUItr != SUEE && *SUItr == BarrierChain)
      SUItr++;

    // Remove all SUs that are now successors of BarrierChain.
    if (SUItr != sus.begin())
      sus.erase(sus.begin(), SUItr);
  }

  // Remove all entries with empty su lists.
  map.remove_if([&](std::pair<ValueType, SUList> &mapEntry) {
      return (mapEntry.second.empty()); });

  // Recompute the size of the map (NumNodes).
  map.reComputeSize();
}

/// If RegPressure is non-null, compute register pressure as a side effect. The
/// DAG builder is an efficient place to do it because it already visits
/// operands.
void ScheduleDAGInstrs::buildSchedGraph(AliasAnalysis *AA,
                                        RegPressureTracker *RPTracker,
                                        PressureDiffs *PDiffs,
                                        LiveIntervals *LIS,
                                        bool TrackLaneMasks) {
  const TargetSubtargetInfo &ST = MF.getSubtarget();
  bool UseAA = EnableAASchedMI.getNumOccurrences() > 0 ? EnableAASchedMI
                                                       : ST.useAA();
  AAForDep = UseAA ? AA : nullptr;

  BarrierChain = nullptr;

  this->TrackLaneMasks = TrackLaneMasks;
  MISUnitMap.clear();
  ScheduleDAG::clearDAG();

  // Create an SUnit for each real instruction.
  initSUnits();

  if (PDiffs)
    PDiffs->init(SUnits.size());

  // We build scheduling units by walking a block's instruction list
  // from bottom to top.

  // Each MIs' memory operand(s) is analyzed to a list of underlying
  // objects. The SU is then inserted in the SUList(s) mapped from the
  // Value(s). Each Value thus gets mapped to lists of SUs depending
  // on it, stores and loads kept separately. Two SUs are trivially
  // non-aliasing if they both depend on only identified Values and do
  // not share any common Value.
  Value2SUsMap Stores, Loads(1 /*TrueMemOrderLatency*/);

  // Certain memory accesses are known to not alias any SU in Stores
  // or Loads, and have therefore their own 'NonAlias'
  // domain. E.g. spill / reload instructions never alias LLVM I/R
  // Values. It would be nice to assume that this type of memory
  // accesses always have a proper memory operand modelling, and are
  // therefore never unanalyzable, but this is conservatively not
  // done.
  Value2SUsMap NonAliasStores, NonAliasLoads(1 /*TrueMemOrderLatency*/);

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

  VRegUses.clear();
  VRegUses.setUniverse(NumVirtRegs);

  // Model data dependencies between instructions being scheduled and the
  // ExitSU.
  addSchedBarrierDeps();

  // Walk the list of instructions, from bottom moving up.
  MachineInstr *DbgMI = nullptr;
  for (MachineBasicBlock::iterator MII = RegionEnd, MIE = RegionBegin;
       MII != MIE; --MII) {
    MachineInstr *MI = std::prev(MII);
    if (MI && DbgMI) {
      DbgValues.push_back(std::make_pair(DbgMI, MI));
      DbgMI = nullptr;
    }

    if (MI->isDebugValue()) {
      DbgMI = MI;
      continue;
    }
    SUnit *SU = MISUnitMap[MI];
    assert(SU && "No SUnit mapped to this MI");

    if (RPTracker) {
      collectVRegUses(SU);

      RegisterOperands RegOpers;
      RegOpers.collect(*MI, *TRI, MRI, TrackLaneMasks, false);
      if (TrackLaneMasks) {
        SlotIndex SlotIdx = LIS->getInstructionIndex(*MI);
        RegOpers.adjustLaneLiveness(*LIS, MRI, SlotIdx);
      }
      if (PDiffs != nullptr)
        PDiffs->addInstruction(SU->NodeNum, RegOpers, MRI);

      RPTracker->recedeSkipDebugValues();
      assert(&*RPTracker->getPos() == MI && "RPTracker in sync");
      RPTracker->recede(RegOpers);
    }

    assert(
        (CanHandleTerminators || (!MI->isTerminator() && !MI->isPosition())) &&
        "Cannot schedule terminators or labels!");

    // Add register-based dependencies (data, anti, and output).
    // For some instructions (calls, returns, inline-asm, etc.) there can
    // be explicit uses and implicit defs, in which case the use will appear
    // on the operand list before the def. Do two passes over the operand
    // list to make sure that defs are processed before any uses.
    bool HasVRegDef = false;
    for (unsigned j = 0, n = MI->getNumOperands(); j != n; ++j) {
      const MachineOperand &MO = MI->getOperand(j);
      if (!MO.isReg() || !MO.isDef())
        continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0)
        continue;

      if (TRI->isPhysicalRegister(Reg))
        addPhysRegDeps(SU, j);
      else {
        HasVRegDef = true;
        addVRegDefDeps(SU, j);
      }
    }
    // Now process all uses.
    for (unsigned j = 0, n = MI->getNumOperands(); j != n; ++j) {
      const MachineOperand &MO = MI->getOperand(j);
      // Only look at use operands.
      // We do not need to check for MO.readsReg() here because subsequent
      // subregister defs will get output dependence edges and need no
      // additional use dependencies.
      if (!MO.isReg() || !MO.isUse())
        continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0)
        continue;

      if (TRI->isPhysicalRegister(Reg))
        addPhysRegDeps(SU, j);
      else if (MO.readsReg()) // ignore undef operands
        addVRegUseDeps(SU, j);
    }

    // If we haven't seen any uses in this scheduling region, create a
    // dependence edge to ExitSU to model the live-out latency. This is required
    // for vreg defs with no in-region use, and prefetches with no vreg def.
    //
    // FIXME: NumDataSuccs would be more precise than NumSuccs here. This
    // check currently relies on being called before adding chain deps.
    if (SU->NumSuccs == 0 && SU->Latency > 1
        && (HasVRegDef || MI->mayLoad())) {
      SDep Dep(SU, SDep::Artificial);
      Dep.setLatency(SU->Latency - 1);
      ExitSU.addPred(Dep);
    }

    // Add memory dependencies (Note: isStoreToStackSlot and
    // isLoadFromStackSLot are not usable after stack slots are lowered to
    // actual addresses).

    // This is a barrier event that acts as a pivotal node in the DAG.
    if (isGlobalMemoryObject(AA, MI)) {

      // Become the barrier chain.
      if (BarrierChain)
        BarrierChain->addPredBarrier(SU);
      BarrierChain = SU;

      DEBUG(dbgs() << "Global memory object and new barrier chain: SU("
            << BarrierChain->NodeNum << ").\n";);

      // Add dependencies against everything below it and clear maps.
      addBarrierChain(Stores);
      addBarrierChain(Loads);
      addBarrierChain(NonAliasStores);
      addBarrierChain(NonAliasLoads);

      continue;
    }

    // If it's not a store or a variant load, we're done.
    if (!MI->mayStore() && !(MI->mayLoad() && !MI->isInvariantLoad(AA)))
      continue;

    // Always add dependecy edge to BarrierChain if present.
    if (BarrierChain)
      BarrierChain->addPredBarrier(SU);

    // Find the underlying objects for MI. The Objs vector is either
    // empty, or filled with the Values of memory locations which this
    // SU depends on. An empty vector means the memory location is
    // unknown, and may alias anything.
    UnderlyingObjectsVector Objs;
    getUnderlyingObjectsForInstr(MI, MFI, Objs, MF.getDataLayout());

    if (MI->mayStore()) {
      if (Objs.empty()) {
        // An unknown store depends on all stores and loads.
        addChainDependencies(SU, Stores);
        addChainDependencies(SU, NonAliasStores);
        addChainDependencies(SU, Loads);
        addChainDependencies(SU, NonAliasLoads);

        // Map this store to 'UnknownValue'.
        Stores.insert(SU, UnknownValue);
      } else {
        // Add precise dependencies against all previously seen memory
        // accesses mapped to the same Value(s).
        for (const UnderlyingObject &UnderlObj : Objs) {
          ValueType V = UnderlObj.getValue();
          bool ThisMayAlias = UnderlObj.mayAlias();

          // Add dependencies to previous stores and loads mapped to V.
          addChainDependencies(SU, (ThisMayAlias ? Stores : NonAliasStores), V);
          addChainDependencies(SU, (ThisMayAlias ? Loads : NonAliasLoads), V);
        }
        // Update the store map after all chains have been added to avoid adding
        // self-loop edge if multiple underlying objects are present.
        for (const UnderlyingObject &UnderlObj : Objs) {
          ValueType V = UnderlObj.getValue();
          bool ThisMayAlias = UnderlObj.mayAlias();

          // Map this store to V.
          (ThisMayAlias ? Stores : NonAliasStores).insert(SU, V);
        }
        // The store may have dependencies to unanalyzable loads and
        // stores.
        addChainDependencies(SU, Loads, UnknownValue);
        addChainDependencies(SU, Stores, UnknownValue);
      }
    } else { // SU is a load.
      if (Objs.empty()) {
        // An unknown load depends on all stores.
        addChainDependencies(SU, Stores);
        addChainDependencies(SU, NonAliasStores);

        Loads.insert(SU, UnknownValue);
      } else {
        for (const UnderlyingObject &UnderlObj : Objs) {
          ValueType V = UnderlObj.getValue();
          bool ThisMayAlias = UnderlObj.mayAlias();

          // Add precise dependencies against all previously seen stores
          // mapping to the same Value(s).
          addChainDependencies(SU, (ThisMayAlias ? Stores : NonAliasStores), V);

          // Map this load to V.
          (ThisMayAlias ? Loads : NonAliasLoads).insert(SU, V);
        }
        // The load may have dependencies to unanalyzable stores.
        addChainDependencies(SU, Stores, UnknownValue);
      }
    }

    // Reduce maps if they grow huge.
    if (Stores.size() + Loads.size() >= HugeRegion) {
      DEBUG(dbgs() << "Reducing Stores and Loads maps.\n";);
      reduceHugeMemNodeMaps(Stores, Loads, getReductionSize());
    }
    if (NonAliasStores.size() + NonAliasLoads.size() >= HugeRegion) {
      DEBUG(dbgs() << "Reducing NonAliasStores and NonAliasLoads maps.\n";);
      reduceHugeMemNodeMaps(NonAliasStores, NonAliasLoads, getReductionSize());
    }
  }

  if (DbgMI)
    FirstDbgValue = DbgMI;

  Defs.clear();
  Uses.clear();
  CurrentVRegDefs.clear();
  CurrentVRegUses.clear();
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PseudoSourceValue* PSV) {
  PSV->printCustom(OS);
  return OS;
}

void ScheduleDAGInstrs::Value2SUsMap::dump() {
  for (auto &Itr : *this) {
    if (Itr.first.is<const Value*>()) {
      const Value *V = Itr.first.get<const Value*>();
      if (isa<UndefValue>(V))
        dbgs() << "Unknown";
      else
        V->printAsOperand(dbgs());
    }
    else if (Itr.first.is<const PseudoSourceValue*>())
      dbgs() <<  Itr.first.get<const PseudoSourceValue*>();
    else
      llvm_unreachable("Unknown Value type.");

    dbgs() << " : ";
    dumpSUList(Itr.second);
  }
}

/// Reduce maps in FIFO order, by N SUs. This is better than turning
/// every Nth memory SU into BarrierChain in buildSchedGraph(), since
/// it avoids unnecessary edges between seen SUs above the new
/// BarrierChain, and those below it.
void ScheduleDAGInstrs::reduceHugeMemNodeMaps(Value2SUsMap &stores,
                                              Value2SUsMap &loads, unsigned N) {
  DEBUG(dbgs() << "Before reduction:\nStoring SUnits:\n";
        stores.dump();
        dbgs() << "Loading SUnits:\n";
        loads.dump());

  // Insert all SU's NodeNums into a vector and sort it.
  std::vector<unsigned> NodeNums;
  NodeNums.reserve(stores.size() + loads.size());
  for (auto &I : stores)
    for (auto *SU : I.second)
      NodeNums.push_back(SU->NodeNum);
  for (auto &I : loads)
    for (auto *SU : I.second)
      NodeNums.push_back(SU->NodeNum);
  std::sort(NodeNums.begin(), NodeNums.end());

  // The N last elements in NodeNums will be removed, and the SU with
  // the lowest NodeNum of them will become the new BarrierChain to
  // let the not yet seen SUs have a dependency to the removed SUs.
  assert (N <= NodeNums.size());
  SUnit *newBarrierChain = &SUnits[*(NodeNums.end() - N)];
  if (BarrierChain) {
    // The aliasing and non-aliasing maps reduce independently of each
    // other, but share a common BarrierChain. Check if the
    // newBarrierChain is above the former one. If it is not, it may
    // introduce a loop to use newBarrierChain, so keep the old one.
    if (newBarrierChain->NodeNum < BarrierChain->NodeNum) {
      BarrierChain->addPredBarrier(newBarrierChain);
      BarrierChain = newBarrierChain;
      DEBUG(dbgs() << "Inserting new barrier chain: SU("
            << BarrierChain->NodeNum << ").\n";);
    }
    else
      DEBUG(dbgs() << "Keeping old barrier chain: SU("
            << BarrierChain->NodeNum << ").\n";);
  }
  else
    BarrierChain = newBarrierChain;

  insertBarrierChain(stores);
  insertBarrierChain(loads);

  DEBUG(dbgs() << "After reduction:\nStoring SUnits:\n";
        stores.dump();
        dbgs() << "Loading SUnits:\n";
        loads.dump());
}

/// \brief Initialize register live-range state for updating kills.
void ScheduleDAGInstrs::startBlockForKills(MachineBasicBlock *BB) {
  // Start with no live registers.
  LiveRegs.reset();

  // Examine the live-in regs of all successors.
  for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
       SE = BB->succ_end(); SI != SE; ++SI) {
    for (const auto &LI : (*SI)->liveins()) {
      // Repeat, for reg and all subregs.
      for (MCSubRegIterator SubRegs(LI.PhysReg, TRI, /*IncludeSelf=*/true);
           SubRegs.isValid(); ++SubRegs)
        LiveRegs.set(*SubRegs);
    }
  }
}

/// \brief If we change a kill flag on the bundle instruction implicit register
/// operands, then we also need to propagate that to any instructions inside
/// the bundle which had the same kill state.
static void toggleBundleKillFlag(MachineInstr *MI, unsigned Reg,
                                 bool NewKillState,
                                 const TargetRegisterInfo *TRI) {
  if (MI->getOpcode() != TargetOpcode::BUNDLE)
    return;

  // Walk backwards from the last instruction in the bundle to the first.
  // Once we set a kill flag on an instruction, we bail out, as otherwise we
  // might set it on too many operands.  We will clear as many flags as we
  // can though.
  MachineBasicBlock::instr_iterator Begin = MI->getIterator();
  MachineBasicBlock::instr_iterator End = getBundleEnd(*MI);
  while (Begin != End) {
    if (NewKillState) {
      if ((--End)->addRegisterKilled(Reg, TRI, /* addIfNotFound= */ false))
         return;
    } else
        (--End)->clearRegisterKills(Reg, TRI);
  }
}

bool ScheduleDAGInstrs::toggleKillFlag(MachineInstr *MI, MachineOperand &MO) {
  // Setting kill flag...
  if (!MO.isKill()) {
    MO.setIsKill(true);
    toggleBundleKillFlag(MI, MO.getReg(), true, TRI);
    return false;
  }

  // If MO itself is live, clear the kill flag...
  if (LiveRegs.test(MO.getReg())) {
    MO.setIsKill(false);
    toggleBundleKillFlag(MI, MO.getReg(), false, TRI);
    return false;
  }

  // If any subreg of MO is live, then create an imp-def for that
  // subreg and keep MO marked as killed.
  MO.setIsKill(false);
  toggleBundleKillFlag(MI, MO.getReg(), false, TRI);
  bool AllDead = true;
  const unsigned SuperReg = MO.getReg();
  MachineInstrBuilder MIB(MF, MI);
  for (MCSubRegIterator SubRegs(SuperReg, TRI); SubRegs.isValid(); ++SubRegs) {
    if (LiveRegs.test(*SubRegs)) {
      MIB.addReg(*SubRegs, RegState::ImplicitDefine);
      AllDead = false;
    }
  }

  if(AllDead) {
    MO.setIsKill(true);
    toggleBundleKillFlag(MI, MO.getReg(), true, TRI);
  }
  return false;
}

// FIXME: Reuse the LivePhysRegs utility for this.
void ScheduleDAGInstrs::fixupKills(MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Fixup kills for BB#" << MBB->getNumber() << '\n');

  LiveRegs.resize(TRI->getNumRegs());
  BitVector killedRegs(TRI->getNumRegs());

  startBlockForKills(MBB);

  // Examine block from end to start...
  unsigned Count = MBB->size();
  for (MachineBasicBlock::iterator I = MBB->end(), E = MBB->begin();
       I != E; --Count) {
    MachineInstr *MI = --I;
    if (MI->isDebugValue())
      continue;

    // Update liveness.  Registers that are defed but not used in this
    // instruction are now dead. Mark register and all subregs as they
    // are completely defined.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isRegMask())
        LiveRegs.clearBitsNotInMask(MO.getRegMask());
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;
      if (!MO.isDef()) continue;
      // Ignore two-addr defs.
      if (MI->isRegTiedToUseOperand(i)) continue;

      // Repeat for reg and all subregs.
      for (MCSubRegIterator SubRegs(Reg, TRI, /*IncludeSelf=*/true);
           SubRegs.isValid(); ++SubRegs)
        LiveRegs.reset(*SubRegs);
    }

    // Examine all used registers and set/clear kill flag. When a
    // register is used multiple times we only set the kill flag on
    // the first use. Don't set kill flags on undef operands.
    killedRegs.reset();
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isUse() || MO.isUndef()) continue;
      unsigned Reg = MO.getReg();
      if ((Reg == 0) || MRI.isReserved(Reg)) continue;

      bool kill = false;
      if (!killedRegs.test(Reg)) {
        kill = true;
        // A register is not killed if any subregs are live...
        for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs) {
          if (LiveRegs.test(*SubRegs)) {
            kill = false;
            break;
          }
        }

        // If subreg is not live, then register is killed if it became
        // live in this instruction
        if (kill)
          kill = !LiveRegs.test(Reg);
      }

      if (MO.isKill() != kill) {
        DEBUG(dbgs() << "Fixing " << MO << " in ");
        // Warning: toggleKillFlag may invalidate MO.
        toggleKillFlag(MI, MO);
        DEBUG(MI->dump());
        DEBUG(if (MI->getOpcode() == TargetOpcode::BUNDLE) {
          MachineBasicBlock::instr_iterator Begin = MI->getIterator();
          MachineBasicBlock::instr_iterator End = getBundleEnd(*MI);
          while (++Begin != End)
            DEBUG(Begin->dump());
        });
      }

      killedRegs.set(Reg);
    }

    // Mark any used register (that is not using undef) and subregs as
    // now live...
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isUse() || MO.isUndef()) continue;
      unsigned Reg = MO.getReg();
      if ((Reg == 0) || MRI.isReserved(Reg)) continue;

      for (MCSubRegIterator SubRegs(Reg, TRI, /*IncludeSelf=*/true);
           SubRegs.isValid(); ++SubRegs)
        LiveRegs.set(*SubRegs);
    }
  }
}

void ScheduleDAGInstrs::dumpNode(const SUnit *SU) const {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  SU->getInstr()->dump();
#endif
}

std::string ScheduleDAGInstrs::getGraphNodeLabel(const SUnit *SU) const {
  std::string s;
  raw_string_ostream oss(s);
  if (SU == &EntrySU)
    oss << "<entry>";
  else if (SU == &ExitSU)
    oss << "<exit>";
  else
    SU->getInstr()->print(oss, /*SkipOpers=*/true);
  return oss.str();
}

/// Return the basic block label. It is not necessarilly unique because a block
/// contains multiple scheduling regions. But it is fine for visualization.
std::string ScheduleDAGInstrs::getDAGName() const {
  return "dag." + BB->getFullName();
}

//===----------------------------------------------------------------------===//
// SchedDFSResult Implementation
//===----------------------------------------------------------------------===//

namespace llvm {
/// \brief Internal state used to compute SchedDFSResult.
class SchedDFSImpl {
  SchedDFSResult &R;

  /// Join DAG nodes into equivalence classes by their subtree.
  IntEqClasses SubtreeClasses;
  /// List PredSU, SuccSU pairs that represent data edges between subtrees.
  std::vector<std::pair<const SUnit*, const SUnit*> > ConnectionPairs;

  struct RootData {
    unsigned NodeID;
    unsigned ParentNodeID;  // Parent node (member of the parent subtree).
    unsigned SubInstrCount; // Instr count in this tree only, not children.

    RootData(unsigned id): NodeID(id),
                           ParentNodeID(SchedDFSResult::InvalidSubtreeID),
                           SubInstrCount(0) {}

    unsigned getSparseSetIndex() const { return NodeID; }
  };

  SparseSet<RootData> RootSet;

public:
  SchedDFSImpl(SchedDFSResult &r): R(r), SubtreeClasses(R.DFSNodeData.size()) {
    RootSet.setUniverse(R.DFSNodeData.size());
  }

  /// Return true if this node been visited by the DFS traversal.
  ///
  /// During visitPostorderNode the Node's SubtreeID is assigned to the Node
  /// ID. Later, SubtreeID is updated but remains valid.
  bool isVisited(const SUnit *SU) const {
    return R.DFSNodeData[SU->NodeNum].SubtreeID
      != SchedDFSResult::InvalidSubtreeID;
  }

  /// Initialize this node's instruction count. We don't need to flag the node
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
    for (SUnit::const_pred_iterator
           PI = SU->Preds.begin(), PE = SU->Preds.end(); PI != PE; ++PI) {
      if (PI->getKind() != SDep::Data)
        continue;
      unsigned PredNum = PI->getSUnit()->NodeNum;
      if ((InstrCount - R.DFSNodeData[PredNum].InstrCount) < R.SubtreeLimit)
        joinPredSubtree(*PI, SU, /*CheckLimit=*/false);

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

  /// Called once for each tree edge after calling visitPostOrderNode on the
  /// predecessor. Increment the parent node's instruction count and
  /// preemptively join this subtree to its parent's if it is small enough.
  void visitPostorderEdge(const SDep &PredDep, const SUnit *Succ) {
    R.DFSNodeData[Succ->NodeNum].InstrCount
      += R.DFSNodeData[PredDep.getSUnit()->NodeNum].InstrCount;
    joinPredSubtree(PredDep, Succ);
  }

  /// Add a connection for cross edges.
  void visitCrossEdge(const SDep &PredDep, const SUnit *Succ) {
    ConnectionPairs.push_back(std::make_pair(PredDep.getSUnit(), Succ));
  }

  /// Set each node's subtree ID to the representative ID and record connections
  /// between trees.
  void finalize() {
    SubtreeClasses.compress();
    R.DFSTreeData.resize(SubtreeClasses.getNumClasses());
    assert(SubtreeClasses.getNumClasses() == RootSet.size()
           && "number of roots should match trees");
    for (SparseSet<RootData>::const_iterator
           RI = RootSet.begin(), RE = RootSet.end(); RI != RE; ++RI) {
      unsigned TreeID = SubtreeClasses[RI->NodeID];
      if (RI->ParentNodeID != SchedDFSResult::InvalidSubtreeID)
        R.DFSTreeData[TreeID].ParentTreeID = SubtreeClasses[RI->ParentNodeID];
      R.DFSTreeData[TreeID].SubInstrCount = RI->SubInstrCount;
      // Note that SubInstrCount may be greater than InstrCount if we joined
      // subtrees across a cross edge. InstrCount will be attributed to the
      // original parent, while SubInstrCount will be attributed to the joined
      // parent.
    }
    R.SubtreeConnections.resize(SubtreeClasses.getNumClasses());
    R.SubtreeConnectLevels.resize(SubtreeClasses.getNumClasses());
    DEBUG(dbgs() << R.getNumSubtrees() << " subtrees:\n");
    for (unsigned Idx = 0, End = R.DFSNodeData.size(); Idx != End; ++Idx) {
      R.DFSNodeData[Idx].SubtreeID = SubtreeClasses[Idx];
      DEBUG(dbgs() << "  SU(" << Idx << ") in tree "
            << R.DFSNodeData[Idx].SubtreeID << '\n');
    }
    for (std::vector<std::pair<const SUnit*, const SUnit*> >::const_iterator
           I = ConnectionPairs.begin(), E = ConnectionPairs.end();
         I != E; ++I) {
      unsigned PredTree = SubtreeClasses[I->first->NodeNum];
      unsigned SuccTree = SubtreeClasses[I->second->NodeNum];
      if (PredTree == SuccTree)
        continue;
      unsigned Depth = I->first->getDepth();
      addConnection(PredTree, SuccTree, Depth);
      addConnection(SuccTree, PredTree, Depth);
    }
  }

protected:
  /// Join the predecessor subtree with the successor that is its DFS
  /// parent. Apply some heuristics before joining.
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
    for (SUnit::const_succ_iterator SI = PredSU->Succs.begin(),
           SE = PredSU->Succs.end(); SI != SE; ++SI) {
      if (SI->getKind() == SDep::Data) {
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
      for (SmallVectorImpl<SchedDFSResult::Connection>::iterator
             I = Connections.begin(), E = Connections.end(); I != E; ++I) {
        if (I->TreeID == ToTree) {
          I->Level = std::max(I->Level, Depth);
          return;
        }
      }
      Connections.push_back(SchedDFSResult::Connection(ToTree, Depth));
      FromTree = R.DFSTreeData[FromTree].ParentTreeID;
    } while (FromTree != SchedDFSResult::InvalidSubtreeID);
  }
};
} // namespace llvm

namespace {
/// \brief Manage the stack used by a reverse depth-first search over the DAG.
class SchedDAGReverseDFS {
  std::vector<std::pair<const SUnit*, SUnit::const_pred_iterator> > DFSStack;
public:
  bool isComplete() const { return DFSStack.empty(); }

  void follow(const SUnit *SU) {
    DFSStack.push_back(std::make_pair(SU, SU->Preds.begin()));
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
} // anonymous

static bool hasDataSucc(const SUnit *SU) {
  for (SUnit::const_succ_iterator
         SI = SU->Succs.begin(), SE = SU->Succs.end(); SI != SE; ++SI) {
    if (SI->getKind() == SDep::Data && !SI->getSUnit()->isBoundaryNode())
      return true;
  }
  return false;
}

/// Compute an ILP metric for all nodes in the subDAG reachable via depth-first
/// search from this root.
void SchedDFSResult::compute(ArrayRef<SUnit> SUnits) {
  if (!IsBottomUp)
    llvm_unreachable("Top-down ILP metric is unimplemnted");

  SchedDFSImpl Impl(*this);
  for (ArrayRef<SUnit>::const_iterator
         SI = SUnits.begin(), SE = SUnits.end(); SI != SE; ++SI) {
    const SUnit *SU = &*SI;
    if (Impl.isVisited(SU) || hasDataSucc(SU))
      continue;

    SchedDAGReverseDFS DFS;
    Impl.visitPreorder(SU);
    DFS.follow(SU);
    for (;;) {
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
  for (SmallVectorImpl<Connection>::const_iterator
         I = SubtreeConnections[SubtreeID].begin(),
         E = SubtreeConnections[SubtreeID].end(); I != E; ++I) {
    SubtreeConnectLevels[I->TreeID] =
      std::max(SubtreeConnectLevels[I->TreeID], I->Level);
    DEBUG(dbgs() << "  Tree: " << I->TreeID
          << " @" << SubtreeConnectLevels[I->TreeID] << '\n');
  }
}

LLVM_DUMP_METHOD
void ILPValue::print(raw_ostream &OS) const {
  OS << InstrCount << " / " << Length << " = ";
  if (!Length)
    OS << "BADILP";
  else
    OS << format("%g", ((double)InstrCount / Length));
}

LLVM_DUMP_METHOD
void ILPValue::dump() const {
  dbgs() << *this << '\n';
}

namespace llvm {

LLVM_DUMP_METHOD
raw_ostream &operator<<(raw_ostream &OS, const ILPValue &Val) {
  Val.print(OS);
  return OS;
}

} // namespace llvm
