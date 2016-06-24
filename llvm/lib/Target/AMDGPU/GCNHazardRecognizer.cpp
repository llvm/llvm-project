//===-- GCNHazardRecognizers.cpp - GCN Hazard Recognizer Impls ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements hazard recognizers for scheduling on GCN processors.
//
//===----------------------------------------------------------------------===//

#include "GCNHazardRecognizer.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Hazard Recoginizer Implementation
//===----------------------------------------------------------------------===//

GCNHazardRecognizer::GCNHazardRecognizer(const MachineFunction &MF) :
  CurrCycleInstr(nullptr),
  MF(MF),
  ST(MF.getSubtarget<SISubtarget>()) {
  MaxLookAhead = 5;
}

void GCNHazardRecognizer::EmitInstruction(SUnit *SU) {
  EmitInstruction(SU->getInstr());
}

void GCNHazardRecognizer::EmitInstruction(MachineInstr *MI) {
  CurrCycleInstr = MI;
}

ScheduleHazardRecognizer::HazardType
GCNHazardRecognizer::getHazardType(SUnit *SU, int Stalls) {
  MachineInstr *MI = SU->getInstr();

  if (SIInstrInfo::isSMRD(*MI) && checkSMRDHazards(MI) > 0)
    return NoopHazard;

  if (SIInstrInfo::isVMEM(*MI) && checkVMEMHazards(MI) > 0)
    return NoopHazard;

  if (SIInstrInfo::isDPP(*MI) && checkDPPHazards(MI) > 0)
    return NoopHazard;

  return NoHazard;
}

unsigned GCNHazardRecognizer::PreEmitNoops(SUnit *SU) {
  return PreEmitNoops(SU->getInstr());
}

unsigned GCNHazardRecognizer::PreEmitNoops(MachineInstr *MI) {
  if (SIInstrInfo::isSMRD(*MI))
    return std::max(0, checkSMRDHazards(MI));

  if (SIInstrInfo::isVMEM(*MI))
    return std::max(0, checkVMEMHazards(MI));

  if (SIInstrInfo::isDPP(*MI))
    return std::max(0, checkDPPHazards(MI));

  return 0;
}

void GCNHazardRecognizer::EmitNoop() {
  EmittedInstrs.push_front(nullptr);
}

void GCNHazardRecognizer::AdvanceCycle() {

  // When the scheduler detects a stall, it will call AdvanceCycle() without
  // emitting any instructions.
  if (!CurrCycleInstr)
    return;

  const SIInstrInfo *TII = ST.getInstrInfo();
  unsigned NumWaitStates = TII->getNumWaitStates(*CurrCycleInstr);

  // Keep track of emitted instructions
  EmittedInstrs.push_front(CurrCycleInstr);

  // Add a nullptr for each additional wait state after the first.  Make sure
  // not to add more than getMaxLookAhead() items to the list, since we
  // truncate the list to that size right after this loop.
  for (unsigned i = 1, e = std::min(NumWaitStates, getMaxLookAhead());
       i < e; ++i) {
    EmittedInstrs.push_front(nullptr);
  }

  // getMaxLookahead() is the largest number of wait states we will ever need
  // to insert, so there is no point in keeping track of more than that many
  // wait states.
  EmittedInstrs.resize(getMaxLookAhead());

  CurrCycleInstr = nullptr;
}

void GCNHazardRecognizer::RecedeCycle() {
  llvm_unreachable("hazard recognizer does not support bottom-up scheduling.");
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

int GCNHazardRecognizer::getWaitStatesSinceDef(
    unsigned Reg, function_ref<bool(MachineInstr *)> IsHazardDef) {
  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  int WaitStates = -1;
  for (MachineInstr *MI : EmittedInstrs) {
    ++WaitStates;
    if (!MI || !IsHazardDef(MI))
      continue;
    if (MI->modifiesRegister(Reg, TRI))
      return WaitStates;
  }
  return std::numeric_limits<int>::max();
}

//===----------------------------------------------------------------------===//
// No-op Hazard Detection
//===----------------------------------------------------------------------===//

static void addRegsToSet(iterator_range<MachineInstr::const_mop_iterator> Ops,
                         std::set<unsigned> &Set) {
  for (const MachineOperand &Op : Ops) {
    if (Op.isReg())
      Set.insert(Op.getReg());
  }
}

int GCNHazardRecognizer::checkSMEMSoftClauseHazards(MachineInstr *SMEM) {
  // SMEM soft clause are only present on VI+
  if (ST.getGeneration() < SISubtarget::VOLCANIC_ISLANDS)
    return 0;

  // A soft-clause is any group of consecutive SMEM instructions.  The
  // instructions in this group may return out of order and/or may be
  // replayed (i.e. the same instruction issued more than once).
  //
  // In order to handle these situations correctly we need to make sure
  // that when a clause has more than one instruction, no instruction in the
  // clause writes to a register that is read another instruction in the clause
  // (including itself). If we encounter this situaion, we need to break the
  // clause by inserting a non SMEM instruction.

  std::set<unsigned> ClauseDefs;
  std::set<unsigned> ClauseUses;

  for (MachineInstr *MI : EmittedInstrs) {

    // When we hit a non-SMEM instruction then we have passed the start of the
    // clause and we can stop.
    if (!MI || !SIInstrInfo::isSMRD(*MI))
      break;

    addRegsToSet(MI->defs(), ClauseDefs);
    addRegsToSet(MI->uses(), ClauseUses);
  }

  if (ClauseDefs.empty())
    return 0;

  // FIXME: When we support stores, we need to make sure not to put loads and
  // stores in the same clause if they use the same address.  For now, just
  // start a new clause whenever we see a store.
  if (SMEM->mayStore())
    return 1;

  addRegsToSet(SMEM->defs(), ClauseDefs);
  addRegsToSet(SMEM->uses(), ClauseUses);

  std::vector<unsigned> Result(std::max(ClauseDefs.size(), ClauseUses.size()));
  std::vector<unsigned>::iterator End;

  End = std::set_intersection(ClauseDefs.begin(), ClauseDefs.end(),
                              ClauseUses.begin(), ClauseUses.end(), Result.begin());

  // If the set of defs and uses intersect then we cannot add this instruction
  // to the clause, so we have a hazard.
  if (End != Result.begin())
    return 1;

  return 0;
}

int GCNHazardRecognizer::checkSMRDHazards(MachineInstr *SMRD) {
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  int WaitStatesNeeded = 0;

  WaitStatesNeeded = checkSMEMSoftClauseHazards(SMRD);

  // This SMRD hazard only affects SI.
  if (ST.getGeneration() != SISubtarget::SOUTHERN_ISLANDS)
    return WaitStatesNeeded;

  // A read of an SGPR by SMRD instruction requires 4 wait states when the
  // SGPR was written by a VALU instruction.
  int SmrdSgprWaitStates = 4;
  auto IsHazardDefFn = [TII] (MachineInstr *MI) { return TII->isVALU(*MI); };

  for (const MachineOperand &Use : SMRD->uses()) {
    if (!Use.isReg())
      continue;
    int WaitStatesNeededForUse =
        SmrdSgprWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefFn);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);
  }
  return WaitStatesNeeded;
}

int GCNHazardRecognizer::checkVMEMHazards(MachineInstr* VMEM) {
  const SIInstrInfo *TII = ST.getInstrInfo();

  if (ST.getGeneration() < SISubtarget::VOLCANIC_ISLANDS)
    return 0;

  const SIRegisterInfo &TRI = TII->getRegisterInfo();

  // A read of an SGPR by a VMEM instruction requires 5 wait states when the
  // SGPR was written by a VALU Instruction.
  int VmemSgprWaitStates = 5;
  int WaitStatesNeeded = 0;
  auto IsHazardDefFn = [TII] (MachineInstr *MI) { return TII->isVALU(*MI); };

  for (const MachineOperand &Use : VMEM->uses()) {
    if (!Use.isReg() || TRI.isVGPR(MF.getRegInfo(), Use.getReg()))
      continue;

    int WaitStatesNeededForUse =
        VmemSgprWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefFn);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);
  }
  return WaitStatesNeeded;
}

int GCNHazardRecognizer::checkDPPHazards(MachineInstr *DPP) {
  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  // Check for DPP VGPR read after VALU VGPR write.
  int DppVgprWaitStates = 2;
  int WaitStatesNeeded = 0;

  for (const MachineOperand &Use : DPP->uses()) {
    if (!Use.isReg() || !TRI->isVGPR(MF.getRegInfo(), Use.getReg()))
      continue;
    int WaitStatesNeededForUse =
        DppVgprWaitStates - getWaitStatesSinceDef(Use.getReg());
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);
  }

  return WaitStatesNeeded;
}
