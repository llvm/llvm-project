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

#include "AMDGPUSubtarget.h"
#include "GCNHazardRecognizer.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <set>
#include <vector>

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

static bool isDivFMas(unsigned Opcode) {
  return Opcode == AMDGPU::V_DIV_FMAS_F32 || Opcode == AMDGPU::V_DIV_FMAS_F64;
}

static bool isSGetReg(unsigned Opcode) {
  return Opcode == AMDGPU::S_GETREG_B32;
}

static bool isSSetReg(unsigned Opcode) {
  return Opcode == AMDGPU::S_SETREG_B32 || Opcode == AMDGPU::S_SETREG_IMM32_B32;
}

static bool isRWLane(unsigned Opcode) {
  return Opcode == AMDGPU::V_READLANE_B32 || Opcode == AMDGPU::V_WRITELANE_B32;
}

static bool isRFE(unsigned Opcode) {
  return Opcode == AMDGPU::S_RFE_B64;
}

static unsigned getHWReg(const SIInstrInfo *TII, const MachineInstr &RegInstr) {
  const MachineOperand *RegOp = TII->getNamedOperand(RegInstr,
                                                     AMDGPU::OpName::simm16);
  return RegOp->getImm() & AMDGPU::Hwreg::ID_MASK_;
}

ScheduleHazardRecognizer::HazardType
GCNHazardRecognizer::getHazardType(SUnit *SU, int Stalls) {
  MachineInstr *MI = SU->getInstr();

  if (SIInstrInfo::isSMRD(*MI) && checkSMRDHazards(MI) > 0)
    return NoopHazard;

  if (SIInstrInfo::isVMEM(*MI) && checkVMEMHazards(MI) > 0)
    return NoopHazard;

  if (SIInstrInfo::isVALU(*MI) && checkVALUHazards(MI) > 0)
    return NoopHazard;

  if (SIInstrInfo::isDPP(*MI) && checkDPPHazards(MI) > 0)
    return NoopHazard;

  if (isDivFMas(MI->getOpcode()) && checkDivFMasHazards(MI) > 0)
    return NoopHazard;

  if (isRWLane(MI->getOpcode()) && checkRWLaneHazards(MI) > 0)
    return NoopHazard;

  if (isSGetReg(MI->getOpcode()) && checkGetRegHazards(MI) > 0)
    return NoopHazard;

  if (isSSetReg(MI->getOpcode()) && checkSetRegHazards(MI) > 0)
    return NoopHazard;

  if (isRFE(MI->getOpcode()) && checkRFEHazards(MI) > 0)
    return NoopHazard;

  return NoHazard;
}

unsigned GCNHazardRecognizer::PreEmitNoops(SUnit *SU) {
  return PreEmitNoops(SU->getInstr());
}

unsigned GCNHazardRecognizer::PreEmitNoops(MachineInstr *MI) {
  if (SIInstrInfo::isSMRD(*MI))
    return std::max(0, checkSMRDHazards(MI));

  if (SIInstrInfo::isVALU(*MI)) {
    int WaitStates = std::max(0, checkVALUHazards(MI));

    if (SIInstrInfo::isVMEM(*MI))
      WaitStates = std::max(WaitStates, checkVMEMHazards(MI));

    if (SIInstrInfo::isDPP(*MI))
      WaitStates = std::max(WaitStates, checkDPPHazards(MI));

    if (isDivFMas(MI->getOpcode()))
      WaitStates = std::max(WaitStates, checkDivFMasHazards(MI));

    if (isRWLane(MI->getOpcode()))
      WaitStates = std::max(WaitStates, checkRWLaneHazards(MI));

    return WaitStates;
  }

  if (isSGetReg(MI->getOpcode()))
    return std::max(0, checkGetRegHazards(MI));

  if (isSSetReg(MI->getOpcode()))
    return std::max(0, checkSetRegHazards(MI));

  if (isRFE(MI->getOpcode()))
    return std::max(0, checkRFEHazards(MI));

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

int GCNHazardRecognizer::getWaitStatesSince(
    function_ref<bool(MachineInstr *)> IsHazard) {
  int WaitStates = -1;
  for (MachineInstr *MI : EmittedInstrs) {
    ++WaitStates;
    if (!MI || !IsHazard(MI))
      continue;
    return WaitStates;
  }
  return std::numeric_limits<int>::max();
}

int GCNHazardRecognizer::getWaitStatesSinceDef(
    unsigned Reg, function_ref<bool(MachineInstr *)> IsHazardDef) {
  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  auto IsHazardFn = [IsHazardDef, TRI, Reg] (MachineInstr *MI) {
    return IsHazardDef(MI) && MI->modifiesRegister(Reg, TRI);
  };

  return getWaitStatesSince(IsHazardFn);
}

int GCNHazardRecognizer::getWaitStatesSinceSetReg(
    function_ref<bool(MachineInstr *)> IsHazard) {
  auto IsHazardFn = [IsHazard] (MachineInstr *MI) {
    return isSSetReg(MI->getOpcode()) && IsHazard(MI);
  };

  return getWaitStatesSince(IsHazardFn);
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

int GCNHazardRecognizer::checkDivFMasHazards(MachineInstr *DivFMas) {
  const SIInstrInfo *TII = ST.getInstrInfo();

  // v_div_fmas requires 4 wait states after a write to vcc from a VALU
  // instruction.
  const int DivFMasWaitStates = 4;
  auto IsHazardDefFn = [TII] (MachineInstr *MI) { return TII->isVALU(*MI); };
  int WaitStatesNeeded = getWaitStatesSinceDef(AMDGPU::VCC, IsHazardDefFn);

  return DivFMasWaitStates - WaitStatesNeeded;
}

int GCNHazardRecognizer::checkGetRegHazards(MachineInstr *GetRegInstr) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  unsigned GetRegHWReg = getHWReg(TII, *GetRegInstr);

  const int GetRegWaitStates = 2;
  auto IsHazardFn = [TII, GetRegHWReg] (MachineInstr *MI) {
    return GetRegHWReg == getHWReg(TII, *MI);
  };
  int WaitStatesNeeded = getWaitStatesSinceSetReg(IsHazardFn);

  return GetRegWaitStates - WaitStatesNeeded;
}

int GCNHazardRecognizer::checkSetRegHazards(MachineInstr *SetRegInstr) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  unsigned HWReg = getHWReg(TII, *SetRegInstr);

  const int SetRegWaitStates =
      ST.getGeneration() <= AMDGPUSubtarget::SEA_ISLANDS ? 1 : 2;
  auto IsHazardFn = [TII, HWReg] (MachineInstr *MI) {
    return HWReg == getHWReg(TII, *MI);
  };
  int WaitStatesNeeded = getWaitStatesSinceSetReg(IsHazardFn);
  return SetRegWaitStates - WaitStatesNeeded;
}

int GCNHazardRecognizer::createsVALUHazard(const MachineInstr &MI) {
  if (!MI.mayStore())
    return -1;

  const SIInstrInfo *TII = ST.getInstrInfo();
  unsigned Opcode = MI.getOpcode();
  const MCInstrDesc &Desc = MI.getDesc();

  int VDataIdx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::vdata);
  int VDataRCID = -1;
  if (VDataIdx != -1)
    VDataRCID = Desc.OpInfo[VDataIdx].RegClass;

  if (TII->isMUBUF(MI) || TII->isMTBUF(MI)) {
    // There is no hazard if the instruction does not use vector regs
    // (like wbinvl1)
    if (VDataIdx == -1)
      return -1;
    // For MUBUF/MTBUF instructions this hazard only exists if the
    // instruction is not using a register in the soffset field.
    const MachineOperand *SOffset =
        TII->getNamedOperand(MI, AMDGPU::OpName::soffset);
    // If we have no soffset operand, then assume this field has been
    // hardcoded to zero.
    if (AMDGPU::getRegBitWidth(VDataRCID) > 64 &&
        (!SOffset || !SOffset->isReg()))
      return VDataIdx;
  }

  // MIMG instructions create a hazard if they don't use a 256-bit T# and
  // the store size is greater than 8 bytes and they have more than two bits
  // of their dmask set.
  // All our MIMG definitions use a 256-bit T#, so we can skip checking for them.
  if (TII->isMIMG(MI)) {
    int SRsrcIdx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::srsrc);
    assert(SRsrcIdx != -1 &&
           AMDGPU::getRegBitWidth(Desc.OpInfo[SRsrcIdx].RegClass) == 256);
    (void)SRsrcIdx;
  }

  if (TII->isFLAT(MI)) {
    int DataIdx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::vdata);
    if (AMDGPU::getRegBitWidth(Desc.OpInfo[DataIdx].RegClass) > 64)
      return DataIdx;
  }

  return -1;
}

int GCNHazardRecognizer::checkVALUHazards(MachineInstr *VALU) {
  // This checks for the hazard where VMEM instructions that store more than
  // 8 bytes can have there store data over written by the next instruction.
  if (!ST.has12DWordStoreHazard())
    return 0;

  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const MachineRegisterInfo &MRI = VALU->getParent()->getParent()->getRegInfo();

  const int VALUWaitStates = 1;
  int WaitStatesNeeded = 0;

  for (const MachineOperand &Def : VALU->defs()) {
    if (!TRI->isVGPR(MRI, Def.getReg()))
      continue;
    unsigned Reg = Def.getReg();
    auto IsHazardFn = [this, Reg, TRI] (MachineInstr *MI) {
      int DataIdx = createsVALUHazard(*MI);
      return DataIdx >= 0 &&
             TRI->regsOverlap(MI->getOperand(DataIdx).getReg(), Reg);
    };
    int WaitStatesNeededForDef =
        VALUWaitStates - getWaitStatesSince(IsHazardFn);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForDef);
  }
  return WaitStatesNeeded;
}

int GCNHazardRecognizer::checkRWLaneHazards(MachineInstr *RWLane) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const MachineRegisterInfo &MRI =
      RWLane->getParent()->getParent()->getRegInfo();

  const MachineOperand *LaneSelectOp =
      TII->getNamedOperand(*RWLane, AMDGPU::OpName::src1);

  if (!LaneSelectOp->isReg() || !TRI->isSGPRReg(MRI, LaneSelectOp->getReg()))
    return 0;

  unsigned LaneSelectReg = LaneSelectOp->getReg();
  auto IsHazardFn = [TII] (MachineInstr *MI) {
    return TII->isVALU(*MI);
  };

  const int RWLaneWaitStates = 4;
  int WaitStatesSince = getWaitStatesSinceDef(LaneSelectReg, IsHazardFn);
  return RWLaneWaitStates - WaitStatesSince;
}

int GCNHazardRecognizer::checkRFEHazards(MachineInstr *RFE) {
  if (ST.getGeneration() < AMDGPUSubtarget::VOLCANIC_ISLANDS)
    return 0;

  const SIInstrInfo *TII = ST.getInstrInfo();

  const int RFEWaitStates = 1;

  auto IsHazardFn = [TII] (MachineInstr *MI) {
    return getHWReg(TII, *MI) == AMDGPU::Hwreg::ID_TRAPSTS;
  };
  int WaitStatesNeeded = getWaitStatesSinceSetReg(IsHazardFn);
  return RFEWaitStates - WaitStatesNeeded;
}
