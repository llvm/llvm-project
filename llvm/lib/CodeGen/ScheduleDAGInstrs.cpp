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

#define DEBUG_TYPE "sched-instrs"
#include "llvm/Operator.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAGILP.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace llvm;

static cl::opt<bool> EnableAASchedMI("enable-aa-sched-mi", cl::Hidden,
    cl::ZeroOrMore, cl::init(false),
    cl::desc("Enable use of AA during MI GAD construction"));

ScheduleDAGInstrs::ScheduleDAGInstrs(MachineFunction &mf,
                                     const MachineLoopInfo &mli,
                                     const MachineDominatorTree &mdt,
                                     bool IsPostRAFlag,
                                     LiveIntervals *lis)
  : ScheduleDAG(mf), MLI(mli), MDT(mdt), MFI(mf.getFrameInfo()), LIS(lis),
    IsPostRA(IsPostRAFlag), CanHandleTerminators(false), FirstDbgValue(0) {
  assert((IsPostRA || LIS) && "PreRA scheduling requires LiveIntervals");
  DbgValues.clear();
  assert(!(IsPostRA && MRI.getNumVirtRegs()) &&
         "Virtual registers must be removed prior to PostRA scheduling");

  const TargetSubtargetInfo &ST = TM.getSubtarget<TargetSubtargetInfo>();
  SchedModel.init(*ST.getSchedModel(), &ST, TII);
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
      // If we find an add of a constant or a multiplied value, it's
      // likely that the other operand will lead us to the base
      // object. We don't have to worry about the case where the
      // object address is somehow being computed by the multiply,
      // because our callers only care when the result is an
      // identifiable object.
      if (U->getOpcode() != Instruction::Add ||
          (!isa<ConstantInt>(U->getOperand(1)) &&
           Operator::getOpcode(U->getOperand(1)) != Instruction::Mul))
        return V;
      V = U->getOperand(0);
    } else {
      return V;
    }
    assert(V->getType()->isIntegerTy() && "Unexpected operand type!");
  } while (1);
}

/// getUnderlyingObject - This is a wrapper around GetUnderlyingObject
/// and adds support for basic ptrtoint+arithmetic+inttoptr sequences.
static const Value *getUnderlyingObject(const Value *V) {
  // First just call Value::getUnderlyingObject to let it do what it does.
  do {
    V = GetUnderlyingObject(V);
    // If it found an inttoptr, use special code to continue climing.
    if (Operator::getOpcode(V) != Instruction::IntToPtr)
      break;
    const Value *O = getUnderlyingObjectFromInt(cast<User>(V)->getOperand(0));
    // If that succeeded in finding a pointer, continue the search.
    if (!O->getType()->isPointerTy())
      break;
    V = O;
  } while (1);
  return V;
}

/// getUnderlyingObjectForInstr - If this machine instr has memory reference
/// information and it can be tracked to a normal reference to a known
/// object, return the Value for that object. Otherwise return null.
static const Value *getUnderlyingObjectForInstr(const MachineInstr *MI,
                                                const MachineFrameInfo *MFI,
                                                bool &MayAlias) {
  MayAlias = true;
  if (!MI->hasOneMemOperand() ||
      !(*MI->memoperands_begin())->getValue() ||
      (*MI->memoperands_begin())->isVolatile())
    return 0;

  const Value *V = (*MI->memoperands_begin())->getValue();
  if (!V)
    return 0;

  V = getUnderlyingObject(V);
  if (const PseudoSourceValue *PSV = dyn_cast<PseudoSourceValue>(V)) {
    // For now, ignore PseudoSourceValues which may alias LLVM IR values
    // because the code that uses this function has no way to cope with
    // such aliases.
    if (PSV->isAliased(MFI))
      return 0;

    MayAlias = PSV->mayAlias(MFI);
    return V;
  }

  if (isIdentifiedObject(V))
    return V;

  return 0;
}

void ScheduleDAGInstrs::startBlock(MachineBasicBlock *bb) {
  BB = bb;
}

void ScheduleDAGInstrs::finishBlock() {
  // Subclasses should no longer refer to the old block.
  BB = 0;
}

/// Initialize the map with the number of registers.
void Reg2SUnitsMap::setRegLimit(unsigned Limit) {
  PhysRegSet.setUniverse(Limit);
  SUnits.resize(Limit);
}

/// Clear the map without deallocating storage.
void Reg2SUnitsMap::clear() {
  for (const_iterator I = reg_begin(), E = reg_end(); I != E; ++I) {
    SUnits[*I].clear();
  }
  PhysRegSet.clear();
}

/// Initialize the DAG and common scheduler state for the current scheduling
/// region. This does not actually create the DAG, only clears it. The
/// scheduling driver may call BuildSchedGraph multiple times per scheduling
/// region.
void ScheduleDAGInstrs::enterRegion(MachineBasicBlock *bb,
                                    MachineBasicBlock::iterator begin,
                                    MachineBasicBlock::iterator end,
                                    unsigned endcount) {
  assert(bb == BB && "startBlock should set BB");
  RegionBegin = begin;
  RegionEnd = end;
  EndIndex = endcount;
  MISUnitMap.clear();

  ScheduleDAG::clearDAG();
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
  MachineInstr *ExitMI = RegionEnd != BB->end() ? &*RegionEnd : 0;
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
        Uses[Reg].push_back(PhysRegSUOper(&ExitSU, -1));
      else {
        assert(!IsPostRA && "Virtual register encountered after regalloc.");
        addVRegUseDeps(&ExitSU, i);
      }
    }
  } else {
    // For others, e.g. fallthrough, conditional branch, assume the exit
    // uses all the registers that are livein to the successor blocks.
    assert(Uses.empty() && "Uses in set before adding deps?");
    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
           SE = BB->succ_end(); SI != SE; ++SI)
      for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
             E = (*SI)->livein_end(); I != E; ++I) {
        unsigned Reg = *I;
        if (!Uses.contains(Reg))
          Uses[Reg].push_back(PhysRegSUOper(&ExitSU, -1));
      }
  }
}

/// MO is an operand of SU's instruction that defines a physical register. Add
/// data dependencies from SU to any uses of the physical register.
void ScheduleDAGInstrs::addPhysRegDataDeps(SUnit *SU, unsigned OperIdx) {
  const MachineOperand &MO = SU->getInstr()->getOperand(OperIdx);
  assert(MO.isDef() && "expect physreg def");

  // Ask the target if address-backscheduling is desirable, and if so how much.
  const TargetSubtargetInfo &ST = TM.getSubtarget<TargetSubtargetInfo>();

  for (MCRegAliasIterator Alias(MO.getReg(), TRI, true);
       Alias.isValid(); ++Alias) {
    if (!Uses.contains(*Alias))
      continue;
    std::vector<PhysRegSUOper> &UseList = Uses[*Alias];
    for (unsigned i = 0, e = UseList.size(); i != e; ++i) {
      SUnit *UseSU = UseList[i].SU;
      if (UseSU == SU)
        continue;

      SDep dep(SU, SDep::Data, *Alias);

      // Adjust the dependence latency using operand def/use information,
      // then allow the target to perform its own adjustments.
      int UseOp = UseList[i].OpIdx;
      MachineInstr *RegUse = UseOp < 0 ? 0 : UseSU->getInstr();
      dep.setLatency(
        SchedModel.computeOperandLatency(SU->getInstr(), OperIdx,
                                         RegUse, UseOp, /*FindMin=*/false));
      dep.setMinLatency(
        SchedModel.computeOperandLatency(SU->getInstr(), OperIdx,
                                         RegUse, UseOp, /*FindMin=*/true));

      ST.adjustSchedDependency(SU, UseSU, dep);
      UseSU->addPred(dep);
    }
  }
}

/// addPhysRegDeps - Add register dependencies (data, anti, and output) from
/// this SUnit to following instructions in the same scheduling region that
/// depend the physical register referenced at OperIdx.
void ScheduleDAGInstrs::addPhysRegDeps(SUnit *SU, unsigned OperIdx) {
  const MachineInstr *MI = SU->getInstr();
  const MachineOperand &MO = MI->getOperand(OperIdx);

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
    std::vector<PhysRegSUOper> &DefList = Defs[*Alias];
    for (unsigned i = 0, e = DefList.size(); i != e; ++i) {
      SUnit *DefSU = DefList[i].SU;
      if (DefSU == &ExitSU)
        continue;
      if (DefSU != SU &&
          (Kind != SDep::Output || !MO.isDead() ||
           !DefSU->getInstr()->registerDefIsDead(*Alias))) {
        if (Kind == SDep::Anti)
          DefSU->addPred(SDep(SU, Kind, /*Reg=*/*Alias));
        else {
          SDep Dep(SU, Kind, /*Reg=*/*Alias);
          unsigned OutLatency =
            SchedModel.computeOutputLatency(MI, OperIdx, DefSU->getInstr());
          Dep.setMinLatency(OutLatency);
          Dep.setLatency(OutLatency);
          DefSU->addPred(Dep);
        }
      }
    }
  }

  if (!MO.isDef()) {
    // Either insert a new Reg2SUnits entry with an empty SUnits list, or
    // retrieve the existing SUnits list for this register's uses.
    // Push this SUnit on the use list.
    Uses[MO.getReg()].push_back(PhysRegSUOper(SU, OperIdx));
  }
  else {
    addPhysRegDataDeps(SU, OperIdx);

    // Either insert a new Reg2SUnits entry with an empty SUnits list, or
    // retrieve the existing SUnits list for this register's defs.
    std::vector<PhysRegSUOper> &DefList = Defs[MO.getReg()];

    // clear this register's use list
    if (Uses.contains(MO.getReg()))
      Uses[MO.getReg()].clear();

    if (!MO.isDead())
      DefList.clear();

    // Calls will not be reordered because of chain dependencies (see
    // below). Since call operands are dead, calls may continue to be added
    // to the DefList making dependence checking quadratic in the size of
    // the block. Instead, we leave only one call at the back of the
    // DefList.
    if (SU->isCall) {
      while (!DefList.empty() && DefList.back().SU->isCall)
        DefList.pop_back();
    }
    // Defs are pushed in the order they are visited and never reordered.
    DefList.push_back(PhysRegSUOper(SU, OperIdx));
  }
}

/// addVRegDefDeps - Add register output and data dependencies from this SUnit
/// to instructions that occur later in the same scheduling region if they read
/// from or write to the virtual register defined at OperIdx.
///
/// TODO: Hoist loop induction variable increments. This has to be
/// reevaluated. Generally, IV scheduling should be done before coalescing.
void ScheduleDAGInstrs::addVRegDefDeps(SUnit *SU, unsigned OperIdx) {
  const MachineInstr *MI = SU->getInstr();
  unsigned Reg = MI->getOperand(OperIdx).getReg();

  // Singly defined vregs do not have output/anti dependencies.
  // The current operand is a def, so we have at least one.
  // Check here if there are any others...
  if (MRI.hasOneDef(Reg))
    return;

  // Add output dependence to the next nearest def of this vreg.
  //
  // Unless this definition is dead, the output dependence should be
  // transitively redundant with antidependencies from this definition's
  // uses. We're conservative for now until we have a way to guarantee the uses
  // are not eliminated sometime during scheduling. The output dependence edge
  // is also useful if output latency exceeds def-use latency.
  VReg2SUnitMap::iterator DefI = VRegDefs.find(Reg);
  if (DefI == VRegDefs.end())
    VRegDefs.insert(VReg2SUnit(Reg, SU));
  else {
    SUnit *DefSU = DefI->SU;
    if (DefSU != SU && DefSU != &ExitSU) {
      SDep Dep(SU, SDep::Output, Reg);
      unsigned OutLatency =
        SchedModel.computeOutputLatency(MI, OperIdx, DefSU->getInstr());
      Dep.setMinLatency(OutLatency);
      Dep.setLatency(OutLatency);
      DefSU->addPred(Dep);
    }
    DefI->SU = SU;
  }
}

/// addVRegUseDeps - Add a register data dependency if the instruction that
/// defines the virtual register used at OperIdx is mapped to an SUnit. Add a
/// register antidependency from this SUnit to instructions that occur later in
/// the same scheduling region if they write the virtual register.
///
/// TODO: Handle ExitSU "uses" properly.
void ScheduleDAGInstrs::addVRegUseDeps(SUnit *SU, unsigned OperIdx) {
  MachineInstr *MI = SU->getInstr();
  unsigned Reg = MI->getOperand(OperIdx).getReg();

  // Lookup this operand's reaching definition.
  assert(LIS && "vreg dependencies requires LiveIntervals");
  LiveRangeQuery LRQ(LIS->getInterval(Reg), LIS->getInstructionIndex(MI));
  VNInfo *VNI = LRQ.valueIn();

  // VNI will be valid because MachineOperand::readsReg() is checked by caller.
  assert(VNI && "No value to read by operand");
  MachineInstr *Def = LIS->getInstructionFromIndex(VNI->def);
  // Phis and other noninstructions (after coalescing) have a NULL Def.
  if (Def) {
    SUnit *DefSU = getSUnit(Def);
    if (DefSU) {
      // The reaching Def lives within this scheduling region.
      // Create a data dependence.
      SDep dep(DefSU, SDep::Data, Reg);
      // Adjust the dependence latency using operand def/use information, then
      // allow the target to perform its own adjustments.
      int DefOp = Def->findRegisterDefOperandIdx(Reg);
      dep.setLatency(
        SchedModel.computeOperandLatency(Def, DefOp, MI, OperIdx, false));
      dep.setMinLatency(
        SchedModel.computeOperandLatency(Def, DefOp, MI, OperIdx, true));

      const TargetSubtargetInfo &ST = TM.getSubtarget<TargetSubtargetInfo>();
      ST.adjustSchedDependency(DefSU, SU, const_cast<SDep &>(dep));
      SU->addPred(dep);
    }
  }

  // Add antidependence to the following def of the vreg it uses.
  VReg2SUnitMap::iterator DefI = VRegDefs.find(Reg);
  if (DefI != VRegDefs.end() && DefI->SU != SU)
    DefI->SU->addPred(SDep(SU, SDep::Anti, Reg));
}

/// Return true if MI is an instruction we are unable to reason about
/// (like a call or something with unmodeled side effects).
static inline bool isGlobalMemoryObject(AliasAnalysis *AA, MachineInstr *MI) {
  if (MI->isCall() || MI->hasUnmodeledSideEffects() ||
      (MI->hasOrderedMemoryRef() &&
       (!MI->mayLoad() || !MI->isInvariantLoad(AA))))
    return true;
  return false;
}

// This MI might have either incomplete info, or known to be unsafe
// to deal with (i.e. volatile object).
static inline bool isUnsafeMemoryObject(MachineInstr *MI,
                                        const MachineFrameInfo *MFI) {
  if (!MI || MI->memoperands_empty())
    return true;
  // We purposefully do no check for hasOneMemOperand() here
  // in hope to trigger an assert downstream in order to
  // finish implementation.
  if ((*MI->memoperands_begin())->isVolatile() ||
       MI->hasUnmodeledSideEffects())
    return true;

  const Value *V = (*MI->memoperands_begin())->getValue();
  if (!V)
    return true;

  V = getUnderlyingObject(V);
  if (const PseudoSourceValue *PSV = dyn_cast<PseudoSourceValue>(V)) {
    // Similarly to getUnderlyingObjectForInstr:
    // For now, ignore PseudoSourceValues which may alias LLVM IR values
    // because the code that uses this function has no way to cope with
    // such aliases.
    if (PSV->isAliased(MFI))
      return true;
  }
  // Does this pointer refer to a distinct and identifiable object?
  if (!isIdentifiedObject(V))
    return true;

  return false;
}

/// This returns true if the two MIs need a chain edge betwee them.
/// If these are not even memory operations, we still may need
/// chain deps between them. The question really is - could
/// these two MIs be reordered during scheduling from memory dependency
/// point of view.
static bool MIsNeedChainEdge(AliasAnalysis *AA, const MachineFrameInfo *MFI,
                             MachineInstr *MIa,
                             MachineInstr *MIb) {
  // Cover a trivial case - no edge is need to itself.
  if (MIa == MIb)
    return false;

  if (isUnsafeMemoryObject(MIa, MFI) || isUnsafeMemoryObject(MIb, MFI))
    return true;

  // If we are dealing with two "normal" loads, we do not need an edge
  // between them - they could be reordered.
  if (!MIa->mayStore() && !MIb->mayStore())
    return false;

  // To this point analysis is generic. From here on we do need AA.
  if (!AA)
    return true;

  MachineMemOperand *MMOa = *MIa->memoperands_begin();
  MachineMemOperand *MMOb = *MIb->memoperands_begin();

  // FIXME: Need to handle multiple memory operands to support all targets.
  if (!MIa->hasOneMemOperand() || !MIb->hasOneMemOperand())
    llvm_unreachable("Multiple memory operands.");

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

  AliasAnalysis::AliasResult AAResult = AA->alias(
  AliasAnalysis::Location(MMOa->getValue(), Overlapa,
                          MMOa->getTBAAInfo()),
  AliasAnalysis::Location(MMOb->getValue(), Overlapb,
                          MMOb->getTBAAInfo()));

  return (AAResult != AliasAnalysis::NoAlias);
}

/// This recursive function iterates over chain deps of SUb looking for
/// "latest" node that needs a chain edge to SUa.
static unsigned
iterateChainSucc(AliasAnalysis *AA, const MachineFrameInfo *MFI,
                 SUnit *SUa, SUnit *SUb, SUnit *ExitSU, unsigned *Depth,
                 SmallPtrSet<const SUnit*, 16> &Visited) {
  if (!SUa || !SUb || SUb == ExitSU)
    return *Depth;

  // Remember visited nodes.
  if (!Visited.insert(SUb))
      return *Depth;
  // If there is _some_ dependency already in place, do not
  // descend any further.
  // TODO: Need to make sure that if that dependency got eliminated or ignored
  // for any reason in the future, we would not violate DAG topology.
  // Currently it does not happen, but makes an implicit assumption about
  // future implementation.
  //
  // Independently, if we encounter node that is some sort of global
  // object (like a call) we already have full set of dependencies to it
  // and we can stop descending.
  if (SUa->isSucc(SUb) ||
      isGlobalMemoryObject(AA, SUb->getInstr()))
    return *Depth;

  // If we do need an edge, or we have exceeded depth budget,
  // add that edge to the predecessors chain of SUb,
  // and stop descending.
  if (*Depth > 200 ||
      MIsNeedChainEdge(AA, MFI, SUa->getInstr(), SUb->getInstr())) {
    SUb->addPred(SDep(SUa, SDep::MayAliasMem));
    return *Depth;
  }
  // Track current depth.
  (*Depth)++;
  // Iterate over chain dependencies only.
  for (SUnit::const_succ_iterator I = SUb->Succs.begin(), E = SUb->Succs.end();
       I != E; ++I)
    if (I->isCtrl())
      iterateChainSucc (AA, MFI, SUa, I->getSUnit(), ExitSU, Depth, Visited);
  return *Depth;
}

/// This function assumes that "downward" from SU there exist
/// tail/leaf of already constructed DAG. It iterates downward and
/// checks whether SU can be aliasing any node dominated
/// by it.
static void adjustChainDeps(AliasAnalysis *AA, const MachineFrameInfo *MFI,
                            SUnit *SU, SUnit *ExitSU, std::set<SUnit *> &CheckList,
                            unsigned LatencyToLoad) {
  if (!SU)
    return;

  SmallPtrSet<const SUnit*, 16> Visited;
  unsigned Depth = 0;

  for (std::set<SUnit *>::iterator I = CheckList.begin(), IE = CheckList.end();
       I != IE; ++I) {
    if (SU == *I)
      continue;
    if (MIsNeedChainEdge(AA, MFI, SU->getInstr(), (*I)->getInstr())) {
      SDep Dep(SU, SDep::MayAliasMem);
      Dep.setLatency(((*I)->getInstr()->mayLoad()) ? LatencyToLoad : 0);
      (*I)->addPred(Dep);
    }
    // Now go through all the chain successors and iterate from them.
    // Keep track of visited nodes.
    for (SUnit::const_succ_iterator J = (*I)->Succs.begin(),
         JE = (*I)->Succs.end(); J != JE; ++J)
      if (J->isCtrl())
        iterateChainSucc (AA, MFI, SU, J->getSUnit(),
                          ExitSU, &Depth, Visited);
  }
}

/// Check whether two objects need a chain edge, if so, add it
/// otherwise remember the rejected SU.
static inline
void addChainDependency (AliasAnalysis *AA, const MachineFrameInfo *MFI,
                         SUnit *SUa, SUnit *SUb,
                         std::set<SUnit *> &RejectList,
                         unsigned TrueMemOrderLatency = 0,
                         bool isNormalMemory = false) {
  // If this is a false dependency,
  // do not add the edge, but rememeber the rejected node.
  if (!EnableAASchedMI ||
      MIsNeedChainEdge(AA, MFI, SUa->getInstr(), SUb->getInstr())) {
    SDep Dep(SUa, isNormalMemory ? SDep::MayAliasMem : SDep::Barrier);
    Dep.setLatency(TrueMemOrderLatency);
    SUb->addPred(Dep);
  }
  else {
    // Duplicate entries should be ignored.
    RejectList.insert(SUb);
    DEBUG(dbgs() << "\tReject chain dep between SU("
          << SUa->NodeNum << ") and SU("
          << SUb->NodeNum << ")\n");
  }
}

/// Create an SUnit for each real instruction, numbered in top-down toplological
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
  SUnits.reserve(BB->size());

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
  }
}

/// If RegPressure is non null, compute register pressure as a side effect. The
/// DAG builder is an efficient place to do it because it already visits
/// operands.
void ScheduleDAGInstrs::buildSchedGraph(AliasAnalysis *AA,
                                        RegPressureTracker *RPTracker) {
  // Create an SUnit for each real instruction.
  initSUnits();

  // We build scheduling units by walking a block's instruction list from bottom
  // to top.

  // Remember where a generic side-effecting instruction is as we procede.
  SUnit *BarrierChain = 0, *AliasChain = 0;

  // Memory references to specific known memory locations are tracked
  // so that they can be given more precise dependencies. We track
  // separately the known memory locations that may alias and those
  // that are known not to alias
  std::map<const Value *, SUnit *> AliasMemDefs, NonAliasMemDefs;
  std::map<const Value *, std::vector<SUnit *> > AliasMemUses, NonAliasMemUses;
  std::set<SUnit*> RejectMemNodes;

  // Remove any stale debug info; sometimes BuildSchedGraph is called again
  // without emitting the info from the previous call.
  DbgValues.clear();
  FirstDbgValue = NULL;

  assert(Defs.empty() && Uses.empty() &&
         "Only BuildGraph should update Defs/Uses");
  Defs.setRegLimit(TRI->getNumRegs());
  Uses.setRegLimit(TRI->getNumRegs());

  assert(VRegDefs.empty() && "Only BuildSchedGraph may access VRegDefs");
  // FIXME: Allow SparseSet to reserve space for the creation of virtual
  // registers during scheduling. Don't artificially inflate the Universe
  // because we want to assert that vregs are not created during DAG building.
  VRegDefs.setUniverse(MRI.getNumVirtRegs());

  // Model data dependencies between instructions being scheduled and the
  // ExitSU.
  addSchedBarrierDeps();

  // Walk the list of instructions, from bottom moving up.
  MachineInstr *PrevMI = NULL;
  for (MachineBasicBlock::iterator MII = RegionEnd, MIE = RegionBegin;
       MII != MIE; --MII) {
    MachineInstr *MI = prior(MII);
    if (MI && PrevMI) {
      DbgValues.push_back(std::make_pair(PrevMI, MI));
      PrevMI = NULL;
    }

    if (MI->isDebugValue()) {
      PrevMI = MI;
      continue;
    }
    if (RPTracker) {
      RPTracker->recede();
      assert(RPTracker->getPos() == prior(MII) && "RPTracker can't find MI");
    }

    assert((!MI->isTerminator() || CanHandleTerminators) && !MI->isLabel() &&
           "Cannot schedule terminators or labels!");

    SUnit *SU = MISUnitMap[MI];
    assert(SU && "No SUnit mapped to this MI");

    // Add register-based dependencies (data, anti, and output).
    for (unsigned j = 0, n = MI->getNumOperands(); j != n; ++j) {
      const MachineOperand &MO = MI->getOperand(j);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;

      if (TRI->isPhysicalRegister(Reg))
        addPhysRegDeps(SU, j);
      else {
        assert(!IsPostRA && "Virtual register encountered!");
        if (MO.isDef())
          addVRegDefDeps(SU, j);
        else if (MO.readsReg()) // ignore undef operands
          addVRegUseDeps(SU, j);
      }
    }

    // Add chain dependencies.
    // Chain dependencies used to enforce memory order should have
    // latency of 0 (except for true dependency of Store followed by
    // aliased Load... we estimate that with a single cycle of latency
    // assuming the hardware will bypass)
    // Note that isStoreToStackSlot and isLoadFromStackSLot are not usable
    // after stack slots are lowered to actual addresses.
    // TODO: Use an AliasAnalysis and do real alias-analysis queries, and
    // produce more precise dependence information.
    unsigned TrueMemOrderLatency = MI->mayStore() ? 1 : 0;
    if (isGlobalMemoryObject(AA, MI)) {
      // Be conservative with these and add dependencies on all memory
      // references, even those that are known to not alias.
      for (std::map<const Value *, SUnit *>::iterator I =
             NonAliasMemDefs.begin(), E = NonAliasMemDefs.end(); I != E; ++I) {
        I->second->addPred(SDep(SU, SDep::Barrier));
      }
      for (std::map<const Value *, std::vector<SUnit *> >::iterator I =
             NonAliasMemUses.begin(), E = NonAliasMemUses.end(); I != E; ++I) {
        for (unsigned i = 0, e = I->second.size(); i != e; ++i) {
          SDep Dep(SU, SDep::Barrier);
          Dep.setLatency(TrueMemOrderLatency);
          I->second[i]->addPred(Dep);
        }
      }
      // Add SU to the barrier chain.
      if (BarrierChain)
        BarrierChain->addPred(SDep(SU, SDep::Barrier));
      BarrierChain = SU;
      // This is a barrier event that acts as a pivotal node in the DAG,
      // so it is safe to clear list of exposed nodes.
      adjustChainDeps(AA, MFI, SU, &ExitSU, RejectMemNodes,
                      TrueMemOrderLatency);
      RejectMemNodes.clear();
      NonAliasMemDefs.clear();
      NonAliasMemUses.clear();

      // fall-through
    new_alias_chain:
      // Chain all possibly aliasing memory references though SU.
      if (AliasChain) {
        unsigned ChainLatency = 0;
        if (AliasChain->getInstr()->mayLoad())
          ChainLatency = TrueMemOrderLatency;
        addChainDependency(AA, MFI, SU, AliasChain, RejectMemNodes,
                           ChainLatency);
      }
      AliasChain = SU;
      for (unsigned k = 0, m = PendingLoads.size(); k != m; ++k)
        addChainDependency(AA, MFI, SU, PendingLoads[k], RejectMemNodes,
                           TrueMemOrderLatency);
      for (std::map<const Value *, SUnit *>::iterator I = AliasMemDefs.begin(),
           E = AliasMemDefs.end(); I != E; ++I)
        addChainDependency(AA, MFI, SU, I->second, RejectMemNodes);
      for (std::map<const Value *, std::vector<SUnit *> >::iterator I =
           AliasMemUses.begin(), E = AliasMemUses.end(); I != E; ++I) {
        for (unsigned i = 0, e = I->second.size(); i != e; ++i)
          addChainDependency(AA, MFI, SU, I->second[i], RejectMemNodes,
                             TrueMemOrderLatency);
      }
      adjustChainDeps(AA, MFI, SU, &ExitSU, RejectMemNodes,
                      TrueMemOrderLatency);
      PendingLoads.clear();
      AliasMemDefs.clear();
      AliasMemUses.clear();
    } else if (MI->mayStore()) {
      bool MayAlias = true;
      if (const Value *V = getUnderlyingObjectForInstr(MI, MFI, MayAlias)) {
        // A store to a specific PseudoSourceValue. Add precise dependencies.
        // Record the def in MemDefs, first adding a dep if there is
        // an existing def.
        std::map<const Value *, SUnit *>::iterator I =
          ((MayAlias) ? AliasMemDefs.find(V) : NonAliasMemDefs.find(V));
        std::map<const Value *, SUnit *>::iterator IE =
          ((MayAlias) ? AliasMemDefs.end() : NonAliasMemDefs.end());
        if (I != IE) {
          addChainDependency(AA, MFI, SU, I->second, RejectMemNodes,
                             0, true);
          I->second = SU;
        } else {
          if (MayAlias)
            AliasMemDefs[V] = SU;
          else
            NonAliasMemDefs[V] = SU;
        }
        // Handle the uses in MemUses, if there are any.
        std::map<const Value *, std::vector<SUnit *> >::iterator J =
          ((MayAlias) ? AliasMemUses.find(V) : NonAliasMemUses.find(V));
        std::map<const Value *, std::vector<SUnit *> >::iterator JE =
          ((MayAlias) ? AliasMemUses.end() : NonAliasMemUses.end());
        if (J != JE) {
          for (unsigned i = 0, e = J->second.size(); i != e; ++i)
            addChainDependency(AA, MFI, SU, J->second[i], RejectMemNodes,
                               TrueMemOrderLatency, true);
          J->second.clear();
        }
        if (MayAlias) {
          // Add dependencies from all the PendingLoads, i.e. loads
          // with no underlying object.
          for (unsigned k = 0, m = PendingLoads.size(); k != m; ++k)
            addChainDependency(AA, MFI, SU, PendingLoads[k], RejectMemNodes,
                               TrueMemOrderLatency);
          // Add dependence on alias chain, if needed.
          if (AliasChain)
            addChainDependency(AA, MFI, SU, AliasChain, RejectMemNodes);
          // But we also should check dependent instructions for the
          // SU in question.
          adjustChainDeps(AA, MFI, SU, &ExitSU, RejectMemNodes,
                          TrueMemOrderLatency);
        }
        // Add dependence on barrier chain, if needed.
        // There is no point to check aliasing on barrier event. Even if
        // SU and barrier _could_ be reordered, they should not. In addition,
        // we have lost all RejectMemNodes below barrier.
        if (BarrierChain)
          BarrierChain->addPred(SDep(SU, SDep::Barrier));
      } else {
        // Treat all other stores conservatively.
        goto new_alias_chain;
      }

      if (!ExitSU.isPred(SU))
        // Push store's up a bit to avoid them getting in between cmp
        // and branches.
        ExitSU.addPred(SDep(SU, SDep::Artificial));
    } else if (MI->mayLoad()) {
      bool MayAlias = true;
      if (MI->isInvariantLoad(AA)) {
        // Invariant load, no chain dependencies needed!
      } else {
        if (const Value *V =
            getUnderlyingObjectForInstr(MI, MFI, MayAlias)) {
          // A load from a specific PseudoSourceValue. Add precise dependencies.
          std::map<const Value *, SUnit *>::iterator I =
            ((MayAlias) ? AliasMemDefs.find(V) : NonAliasMemDefs.find(V));
          std::map<const Value *, SUnit *>::iterator IE =
            ((MayAlias) ? AliasMemDefs.end() : NonAliasMemDefs.end());
          if (I != IE)
            addChainDependency(AA, MFI, SU, I->second, RejectMemNodes, 0, true);
          if (MayAlias)
            AliasMemUses[V].push_back(SU);
          else
            NonAliasMemUses[V].push_back(SU);
        } else {
          // A load with no underlying object. Depend on all
          // potentially aliasing stores.
          for (std::map<const Value *, SUnit *>::iterator I =
                 AliasMemDefs.begin(), E = AliasMemDefs.end(); I != E; ++I)
            addChainDependency(AA, MFI, SU, I->second, RejectMemNodes);

          PendingLoads.push_back(SU);
          MayAlias = true;
        }
        if (MayAlias)
          adjustChainDeps(AA, MFI, SU, &ExitSU, RejectMemNodes, /*Latency=*/0);
        // Add dependencies on alias and barrier chains, if needed.
        if (MayAlias && AliasChain)
          addChainDependency(AA, MFI, SU, AliasChain, RejectMemNodes);
        if (BarrierChain)
          BarrierChain->addPred(SDep(SU, SDep::Barrier));
      }
    }
  }
  if (PrevMI)
    FirstDbgValue = PrevMI;

  Defs.clear();
  Uses.clear();
  VRegDefs.clear();
  PendingLoads.clear();
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
    SU->getInstr()->print(oss);
  return oss.str();
}

/// Return the basic block label. It is not necessarilly unique because a block
/// contains multiple scheduling regions. But it is fine for visualization.
std::string ScheduleDAGInstrs::getDAGName() const {
  return "dag." + BB->getFullName();
}

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

  void backtrack() { DFSStack.pop_back(); }

  const SUnit *getCurr() const { return DFSStack.back().first; }

  SUnit::const_pred_iterator getPred() const { return DFSStack.back().second; }

  SUnit::const_pred_iterator getPredEnd() const {
    return getCurr()->Preds.end();
  }
};
} // anonymous

void ScheduleDAGILP::resize(unsigned NumSUnits) {
  ILPValues.resize(NumSUnits);
}

ILPValue ScheduleDAGILP::getILP(const SUnit *SU) {
  return ILPValues[SU->NodeNum];
}

// A leaf node has an ILP of 1/1.
static ILPValue initILP(const SUnit *SU) {
  unsigned Cnt = SU->getInstr()->isTransient() ? 0 : 1;
  return ILPValue(Cnt, 1 + SU->getDepth());
}

/// Compute an ILP metric for all nodes in the subDAG reachable via depth-first
/// search from this root.
void ScheduleDAGILP::computeILP(const SUnit *Root) {
  if (!IsBottomUp)
    llvm_unreachable("Top-down ILP metric is unimplemnted");

  SchedDAGReverseDFS DFS;
  // Mark a node visited by validating it.
  ILPValues[Root->NodeNum] = initILP(Root);
  DFS.follow(Root);
  for (;;) {
    // Traverse the leftmost path as far as possible.
    while (DFS.getPred() != DFS.getPredEnd()) {
      const SUnit *PredSU = DFS.getPred()->getSUnit();
      DFS.advance();
      // If the pred is already valid, skip it.
      if (ILPValues[PredSU->NodeNum].isValid())
        continue;
      ILPValues[PredSU->NodeNum] = initILP(PredSU);
      DFS.follow(PredSU);
    }
    // Visit the top of the stack in postorder and backtrack.
    unsigned PredCount = ILPValues[DFS.getCurr()->NodeNum].InstrCount;
    DFS.backtrack();
    if (DFS.isComplete())
      break;
    // Add the recently finished predecessor's bottom-up descendent count.
    ILPValues[DFS.getCurr()->NodeNum].InstrCount += PredCount;
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void ILPValue::print(raw_ostream &OS) const {
  if (!isValid())
    OS << "BADILP";
  OS << InstrCount << " / " << Cycles << " = "
     << format("%g", ((double)InstrCount / Cycles));
}

void ILPValue::dump() const {
  dbgs() << *this << '\n';
}

namespace llvm {

raw_ostream &operator<<(raw_ostream &OS, const ILPValue &Val) {
  Val.print(OS);
  return OS;
}

} // namespace llvm
#endif // !NDEBUG || LLVM_ENABLE_DUMP
