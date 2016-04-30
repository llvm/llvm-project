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
  MF(MF) {
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
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo*>(MF.getSubtarget().getInstrInfo());
  MachineInstr *MI = SU->getInstr();

  if (TII->isSMRD(*MI) && checkSMRDHazards(MI) > 0)
    return NoopHazard;

  if (TII->isVMEM(*MI) && checkVMEMHazards(MI) > 0)
    return NoopHazard;

  return NoHazard;
}

unsigned GCNHazardRecognizer::PreEmitNoops(SUnit *SU) {
  return PreEmitNoops(SU->getInstr());
}

unsigned GCNHazardRecognizer::PreEmitNoops(MachineInstr *MI) {
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo*>(MF.getSubtarget().getInstrInfo());

  if (TII->isSMRD(*MI))
    return std::max(0, checkSMRDHazards(MI));

  if (TII->isVMEM(*MI))
    return std::max(0, checkVMEMHazards(MI));

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

  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo*>(MF.getSubtarget().getInstrInfo());
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

int GCNHazardRecognizer::getWaitStatesSinceDef(unsigned Reg,
                              std::function<bool(MachineInstr*)> IsHazardDef ) {
  const TargetRegisterInfo *TRI =
      MF.getSubtarget<AMDGPUSubtarget>().getRegisterInfo();

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

int GCNHazardRecognizer::checkSMRDHazards(MachineInstr *SMRD) {
  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();
  const SIInstrInfo *TII = static_cast<const SIInstrInfo*>(ST.getInstrInfo());

  // This SMRD hazard only affects SI.
  if (ST.getGeneration() != AMDGPUSubtarget::SOUTHERN_ISLANDS)
    return 0;

  // A read of an SGPR by SMRD instruction requires 4 wait states when the
  // SGPR was written by a VALU instruction.
  int SmrdSgprWaitStates = 4;
  int WaitStatesNeeded = 0;
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
  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();
  const SIInstrInfo *TII = static_cast<const SIInstrInfo*>(ST.getInstrInfo());

  if (ST.getGeneration() < AMDGPUSubtarget::VOLCANIC_ISLANDS)
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
