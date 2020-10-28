//===-- GCNHazardRecognizers.cpp - GCN Hazard Recognizer Impls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements hazard recognizers for scheduling on GCN processors.
//
//===----------------------------------------------------------------------===//

#include "GCNHazardRecognizer.h"
#include "AMDGPUSubtarget.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
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
  IsHazardRecognizerMode(false),
  CurrCycleInstr(nullptr),
  MF(MF),
  ST(MF.getSubtarget<GCNSubtarget>()),
  TII(*ST.getInstrInfo()),
  TRI(TII.getRegisterInfo()),
  ClauseUses(TRI.getNumRegUnits()),
  ClauseDefs(TRI.getNumRegUnits()) {
  MaxLookAhead = MF.getRegInfo().isPhysRegUsed(AMDGPU::AGPR0) ? 18 : 5;
  TSchedModel.init(&ST);
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

static bool isSMovRel(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::S_MOVRELS_B32:
  case AMDGPU::S_MOVRELS_B64:
  case AMDGPU::S_MOVRELD_B32:
  case AMDGPU::S_MOVRELD_B64:
    return true;
  default:
    return false;
  }
}

static bool isSendMsgTraceDataOrGDS(const SIInstrInfo &TII,
                                    const MachineInstr &MI) {
  if (TII.isAlwaysGDS(MI.getOpcode()))
    return true;

  switch (MI.getOpcode()) {
  case AMDGPU::S_SENDMSG:
  case AMDGPU::S_SENDMSGHALT:
  case AMDGPU::S_TTRACEDATA:
    return true;
  // These DS opcodes don't support GDS.
  case AMDGPU::DS_NOP:
  case AMDGPU::DS_PERMUTE_B32:
  case AMDGPU::DS_BPERMUTE_B32:
    return false;
  default:
    if (TII.isDS(MI.getOpcode())) {
      int GDS = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                           AMDGPU::OpName::gds);
      if (MI.getOperand(GDS).getImm())
        return true;
    }
    return false;
  }
}

static bool isPermlane(const MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  return Opcode == AMDGPU::V_PERMLANE16_B32 ||
         Opcode == AMDGPU::V_PERMLANEX16_B32;
}

static unsigned getHWReg(const SIInstrInfo *TII, const MachineInstr &RegInstr) {
  const MachineOperand *RegOp = TII->getNamedOperand(RegInstr,
                                                     AMDGPU::OpName::simm16);
  return RegOp->getImm() & AMDGPU::Hwreg::ID_MASK_;
}

ScheduleHazardRecognizer::HazardType
GCNHazardRecognizer::getHazardType(SUnit *SU, int Stalls) {
  MachineInstr *MI = SU->getInstr();
  if (MI->isBundle())
   return NoHazard;

  if (SIInstrInfo::isSMRD(*MI) && checkSMRDHazards(MI) > 0)
    return NoopHazard;

  // FIXME: Should flat be considered vmem?
  if ((SIInstrInfo::isVMEM(*MI) ||
       SIInstrInfo::isFLAT(*MI))
      && checkVMEMHazards(MI) > 0)
    return NoopHazard;

  if (ST.hasNSAtoVMEMBug() && checkNSAtoVMEMHazard(MI) > 0)
    return NoopHazard;

  if (checkFPAtomicToDenormModeHazard(MI) > 0)
    return NoopHazard;

  if (ST.hasNoDataDepHazard())
    return NoHazard;

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

  if (ST.hasReadM0MovRelInterpHazard() &&
      (TII.isVINTRP(*MI) || isSMovRel(MI->getOpcode())) &&
      checkReadM0Hazards(MI) > 0)
    return NoopHazard;

  if (ST.hasReadM0SendMsgHazard() && isSendMsgTraceDataOrGDS(TII, *MI) &&
      checkReadM0Hazards(MI) > 0)
    return NoopHazard;

  if (SIInstrInfo::isMAI(*MI) && checkMAIHazards(MI) > 0)
    return NoopHazard;

  if (MI->mayLoadOrStore() && checkMAILdStHazards(MI) > 0)
    return NoopHazard;

  if (MI->isInlineAsm() && checkInlineAsmHazards(MI) > 0)
    return NoopHazard;

  return NoHazard;
}

static void insertNoopInBundle(MachineInstr *MI, const SIInstrInfo &TII) {
  BuildMI(*MI->getParent(), MI, MI->getDebugLoc(), TII.get(AMDGPU::S_NOP))
      .addImm(0);
}

void GCNHazardRecognizer::processBundle() {
  MachineBasicBlock::instr_iterator MI = std::next(CurrCycleInstr->getIterator());
  MachineBasicBlock::instr_iterator E = CurrCycleInstr->getParent()->instr_end();
  // Check bundled MachineInstr's for hazards.
  for (; MI != E && MI->isInsideBundle(); ++MI) {
    CurrCycleInstr = &*MI;
    unsigned WaitStates = PreEmitNoopsCommon(CurrCycleInstr);

    if (IsHazardRecognizerMode)
      fixHazards(CurrCycleInstr);

    for (unsigned i = 0; i < WaitStates; ++i)
      insertNoopInBundle(CurrCycleInstr, TII);

    // It’s unnecessary to track more than MaxLookAhead instructions. Since we
    // include the bundled MI directly after, only add a maximum of
    // (MaxLookAhead - 1) noops to EmittedInstrs.
    for (unsigned i = 0, e = std::min(WaitStates, MaxLookAhead - 1); i < e; ++i)
      EmittedInstrs.push_front(nullptr);

    EmittedInstrs.push_front(CurrCycleInstr);
    EmittedInstrs.resize(MaxLookAhead);
  }
  CurrCycleInstr = nullptr;
}

unsigned GCNHazardRecognizer::PreEmitNoops(MachineInstr *MI) {
  IsHazardRecognizerMode = true;
  CurrCycleInstr = MI;
  unsigned W = PreEmitNoopsCommon(MI);
  fixHazards(MI);
  CurrCycleInstr = nullptr;
  return W;
}

unsigned GCNHazardRecognizer::PreEmitNoopsCommon(MachineInstr *MI) {
  if (MI->isBundle())
    return 0;

  int WaitStates = 0;

  if (SIInstrInfo::isSMRD(*MI))
    return std::max(WaitStates, checkSMRDHazards(MI));

  if (SIInstrInfo::isVMEM(*MI) || SIInstrInfo::isFLAT(*MI))
    WaitStates = std::max(WaitStates, checkVMEMHazards(MI));

  if (ST.hasNSAtoVMEMBug())
    WaitStates = std::max(WaitStates, checkNSAtoVMEMHazard(MI));

  WaitStates = std::max(WaitStates, checkFPAtomicToDenormModeHazard(MI));

  if (ST.hasNoDataDepHazard())
    return WaitStates;

  if (SIInstrInfo::isVALU(*MI))
    WaitStates = std::max(WaitStates, checkVALUHazards(MI));

  if (SIInstrInfo::isDPP(*MI))
    WaitStates = std::max(WaitStates, checkDPPHazards(MI));

  if (isDivFMas(MI->getOpcode()))
    WaitStates = std::max(WaitStates, checkDivFMasHazards(MI));

  if (isRWLane(MI->getOpcode()))
    WaitStates = std::max(WaitStates, checkRWLaneHazards(MI));

  if (MI->isInlineAsm())
    return std::max(WaitStates, checkInlineAsmHazards(MI));

  if (isSGetReg(MI->getOpcode()))
    return std::max(WaitStates, checkGetRegHazards(MI));

  if (isSSetReg(MI->getOpcode()))
    return std::max(WaitStates, checkSetRegHazards(MI));

  if (isRFE(MI->getOpcode()))
    return std::max(WaitStates, checkRFEHazards(MI));

  if (ST.hasReadM0MovRelInterpHazard() && (TII.isVINTRP(*MI) ||
                                           isSMovRel(MI->getOpcode())))
    return std::max(WaitStates, checkReadM0Hazards(MI));

  if (ST.hasReadM0SendMsgHazard() && isSendMsgTraceDataOrGDS(TII, *MI))
    return std::max(WaitStates, checkReadM0Hazards(MI));

  if (SIInstrInfo::isMAI(*MI))
    return std::max(WaitStates, checkMAIHazards(MI));

  if (MI->mayLoadOrStore())
    return std::max(WaitStates, checkMAILdStHazards(MI));

  return WaitStates;
}

void GCNHazardRecognizer::EmitNoop() {
  EmittedInstrs.push_front(nullptr);
}

void GCNHazardRecognizer::AdvanceCycle() {
  // When the scheduler detects a stall, it will call AdvanceCycle() without
  // emitting any instructions.
  if (!CurrCycleInstr)
    return;

  // Do not track non-instructions which do not affect the wait states.
  // If included, these instructions can lead to buffer overflow such that
  // detectable hazards are missed.
  if (CurrCycleInstr->isImplicitDef() || CurrCycleInstr->isDebugInstr() ||
      CurrCycleInstr->isKill())
    return;

  if (CurrCycleInstr->isBundle()) {
    processBundle();
    return;
  }

  unsigned NumWaitStates = TII.getNumWaitStates(*CurrCycleInstr);

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

typedef function_ref<bool(MachineInstr *, int WaitStates)> IsExpiredFn;

// Returns a minimum wait states since \p I walking all predecessors.
// Only scans until \p IsExpired does not return true.
// Can only be run in a hazard recognizer mode.
static int getWaitStatesSince(GCNHazardRecognizer::IsHazardFn IsHazard,
                              MachineBasicBlock *MBB,
                              MachineBasicBlock::reverse_instr_iterator I,
                              int WaitStates,
                              IsExpiredFn IsExpired,
                              DenseSet<const MachineBasicBlock *> &Visited) {
  for (auto E = MBB->instr_rend(); I != E; ++I) {
    // Don't add WaitStates for parent BUNDLE instructions.
    if (I->isBundle())
      continue;

    if (IsHazard(&*I))
      return WaitStates;

    if (I->isInlineAsm() || I->isImplicitDef() || I->isDebugInstr())
      continue;

    WaitStates += SIInstrInfo::getNumWaitStates(*I);

    if (IsExpired(&*I, WaitStates))
      return std::numeric_limits<int>::max();
  }

  int MinWaitStates = WaitStates;
  bool Found = false;
  for (MachineBasicBlock *Pred : MBB->predecessors()) {
    if (!Visited.insert(Pred).second)
      continue;

    int W = getWaitStatesSince(IsHazard, Pred, Pred->instr_rbegin(),
                               WaitStates, IsExpired, Visited);

    if (W == std::numeric_limits<int>::max())
      continue;

    MinWaitStates = Found ? std::min(MinWaitStates, W) : W;
    if (IsExpired(nullptr, MinWaitStates))
      return MinWaitStates;

    Found = true;
  }

  if (Found)
    return MinWaitStates;

  return std::numeric_limits<int>::max();
}

static int getWaitStatesSince(GCNHazardRecognizer::IsHazardFn IsHazard,
                              MachineInstr *MI,
                              IsExpiredFn IsExpired) {
  DenseSet<const MachineBasicBlock *> Visited;
  return getWaitStatesSince(IsHazard, MI->getParent(),
                            std::next(MI->getReverseIterator()),
                            0, IsExpired, Visited);
}

int GCNHazardRecognizer::getWaitStatesSince(IsHazardFn IsHazard, int Limit) {
  if (IsHazardRecognizerMode) {
    auto IsExpiredFn = [Limit] (MachineInstr *, int WaitStates) {
      return WaitStates >= Limit;
    };
    return ::getWaitStatesSince(IsHazard, CurrCycleInstr, IsExpiredFn);
  }

  int WaitStates = 0;
  for (MachineInstr *MI : EmittedInstrs) {
    if (MI) {
      if (IsHazard(MI))
        return WaitStates;

      if (MI->isInlineAsm())
        continue;
    }
    ++WaitStates;

    if (WaitStates >= Limit)
      break;
  }
  return std::numeric_limits<int>::max();
}

int GCNHazardRecognizer::getWaitStatesSinceDef(unsigned Reg,
                                               IsHazardFn IsHazardDef,
                                               int Limit) {
  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  auto IsHazardFn = [IsHazardDef, TRI, Reg] (MachineInstr *MI) {
    return IsHazardDef(MI) && MI->modifiesRegister(Reg, TRI);
  };

  return getWaitStatesSince(IsHazardFn, Limit);
}

int GCNHazardRecognizer::getWaitStatesSinceSetReg(IsHazardFn IsHazard,
                                                  int Limit) {
  auto IsHazardFn = [IsHazard] (MachineInstr *MI) {
    return isSSetReg(MI->getOpcode()) && IsHazard(MI);
  };

  return getWaitStatesSince(IsHazardFn, Limit);
}

//===----------------------------------------------------------------------===//
// No-op Hazard Detection
//===----------------------------------------------------------------------===//

static void addRegUnits(const SIRegisterInfo &TRI,
                        BitVector &BV, unsigned Reg) {
  for (MCRegUnitIterator RUI(Reg, &TRI); RUI.isValid(); ++RUI)
    BV.set(*RUI);
}

static void addRegsToSet(const SIRegisterInfo &TRI,
                         iterator_range<MachineInstr::const_mop_iterator> Ops,
                         BitVector &Set) {
  for (const MachineOperand &Op : Ops) {
    if (Op.isReg())
      addRegUnits(TRI, Set, Op.getReg());
  }
}

void GCNHazardRecognizer::addClauseInst(const MachineInstr &MI) {
  // XXX: Do we need to worry about implicit operands
  addRegsToSet(TRI, MI.defs(), ClauseDefs);
  addRegsToSet(TRI, MI.uses(), ClauseUses);
}

static bool breaksSMEMSoftClause(MachineInstr *MI) {
  return !SIInstrInfo::isSMRD(*MI);
}

static bool breaksVMEMSoftClause(MachineInstr *MI) {
  return !SIInstrInfo::isVMEM(*MI) && !SIInstrInfo::isFLAT(*MI);
}

int GCNHazardRecognizer::checkSoftClauseHazards(MachineInstr *MEM) {
  // SMEM soft clause are only present on VI+, and only matter if xnack is
  // enabled.
  if (!ST.isXNACKEnabled())
    return 0;

  bool IsSMRD = TII.isSMRD(*MEM);

  resetClause();

  // A soft-clause is any group of consecutive SMEM instructions.  The
  // instructions in this group may return out of order and/or may be
  // replayed (i.e. the same instruction issued more than once).
  //
  // In order to handle these situations correctly we need to make sure that
  // when a clause has more than one instruction, no instruction in the clause
  // writes to a register that is read by another instruction in the clause
  // (including itself). If we encounter this situaion, we need to break the
  // clause by inserting a non SMEM instruction.

  for (MachineInstr *MI : EmittedInstrs) {
    // When we hit a non-SMEM instruction then we have passed the start of the
    // clause and we can stop.
    if (!MI)
      break;

    if (IsSMRD ? breaksSMEMSoftClause(MI) : breaksVMEMSoftClause(MI))
      break;

    addClauseInst(*MI);
  }

  if (ClauseDefs.none())
    return 0;

  // We need to make sure not to put loads and stores in the same clause if they
  // use the same address. For now, just start a new clause whenever we see a
  // store.
  if (MEM->mayStore())
    return 1;

  addClauseInst(*MEM);

  // If the set of defs and uses intersect then we cannot add this instruction
  // to the clause, so we have a hazard.
  return ClauseDefs.anyCommon(ClauseUses) ? 1 : 0;
}

int GCNHazardRecognizer::checkSMRDHazards(MachineInstr *SMRD) {
  int WaitStatesNeeded = 0;

  WaitStatesNeeded = checkSoftClauseHazards(SMRD);

  // This SMRD hazard only affects SI.
  if (!ST.hasSMRDReadVALUDefHazard())
    return WaitStatesNeeded;

  // A read of an SGPR by SMRD instruction requires 4 wait states when the
  // SGPR was written by a VALU instruction.
  int SmrdSgprWaitStates = 4;
  auto IsHazardDefFn = [this] (MachineInstr *MI) { return TII.isVALU(*MI); };
  auto IsBufferHazardDefFn = [this] (MachineInstr *MI) { return TII.isSALU(*MI); };

  bool IsBufferSMRD = TII.isBufferSMRD(*SMRD);

  for (const MachineOperand &Use : SMRD->uses()) {
    if (!Use.isReg())
      continue;
    int WaitStatesNeededForUse =
        SmrdSgprWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefFn,
                                                   SmrdSgprWaitStates);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);

    // This fixes what appears to be undocumented hardware behavior in SI where
    // s_mov writing a descriptor and s_buffer_load_dword reading the descriptor
    // needs some number of nops in between. We don't know how many we need, but
    // let's use 4. This wasn't discovered before probably because the only
    // case when this happens is when we expand a 64-bit pointer into a full
    // descriptor and use s_buffer_load_dword instead of s_load_dword, which was
    // probably never encountered in the closed-source land.
    if (IsBufferSMRD) {
      int WaitStatesNeededForUse =
        SmrdSgprWaitStates - getWaitStatesSinceDef(Use.getReg(),
                                                   IsBufferHazardDefFn,
                                                   SmrdSgprWaitStates);
      WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);
    }
  }

  return WaitStatesNeeded;
}

int GCNHazardRecognizer::checkVMEMHazards(MachineInstr* VMEM) {
  if (!ST.hasVMEMReadSGPRVALUDefHazard())
    return 0;

  int WaitStatesNeeded = checkSoftClauseHazards(VMEM);

  // A read of an SGPR by a VMEM instruction requires 5 wait states when the
  // SGPR was written by a VALU Instruction.
  const int VmemSgprWaitStates = 5;
  auto IsHazardDefFn = [this] (MachineInstr *MI) { return TII.isVALU(*MI); };
  for (const MachineOperand &Use : VMEM->uses()) {
    if (!Use.isReg() || TRI.isVGPR(MF.getRegInfo(), Use.getReg()))
      continue;

    int WaitStatesNeededForUse =
        VmemSgprWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefFn,
                                                   VmemSgprWaitStates);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);
  }
  return WaitStatesNeeded;
}

int GCNHazardRecognizer::checkDPPHazards(MachineInstr *DPP) {
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();

  // Check for DPP VGPR read after VALU VGPR write and EXEC write.
  int DppVgprWaitStates = 2;
  int DppExecWaitStates = 5;
  int WaitStatesNeeded = 0;
  auto IsHazardDefFn = [TII] (MachineInstr *MI) { return TII->isVALU(*MI); };

  for (const MachineOperand &Use : DPP->uses()) {
    if (!Use.isReg() || !TRI->isVGPR(MF.getRegInfo(), Use.getReg()))
      continue;
    int WaitStatesNeededForUse =
        DppVgprWaitStates - getWaitStatesSinceDef(Use.getReg(),
                              [](MachineInstr *) { return true; },
                              DppVgprWaitStates);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);
  }

  WaitStatesNeeded = std::max(
      WaitStatesNeeded,
      DppExecWaitStates - getWaitStatesSinceDef(AMDGPU::EXEC, IsHazardDefFn,
                                                DppExecWaitStates));

  return WaitStatesNeeded;
}

int GCNHazardRecognizer::checkDivFMasHazards(MachineInstr *DivFMas) {
  const SIInstrInfo *TII = ST.getInstrInfo();

  // v_div_fmas requires 4 wait states after a write to vcc from a VALU
  // instruction.
  const int DivFMasWaitStates = 4;
  auto IsHazardDefFn = [TII] (MachineInstr *MI) { return TII->isVALU(*MI); };
  int WaitStatesNeeded = getWaitStatesSinceDef(AMDGPU::VCC, IsHazardDefFn,
                                               DivFMasWaitStates);

  return DivFMasWaitStates - WaitStatesNeeded;
}

int GCNHazardRecognizer::checkGetRegHazards(MachineInstr *GetRegInstr) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  unsigned GetRegHWReg = getHWReg(TII, *GetRegInstr);

  const int GetRegWaitStates = 2;
  auto IsHazardFn = [TII, GetRegHWReg] (MachineInstr *MI) {
    return GetRegHWReg == getHWReg(TII, *MI);
  };
  int WaitStatesNeeded = getWaitStatesSinceSetReg(IsHazardFn, GetRegWaitStates);

  return GetRegWaitStates - WaitStatesNeeded;
}

int GCNHazardRecognizer::checkSetRegHazards(MachineInstr *SetRegInstr) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  unsigned HWReg = getHWReg(TII, *SetRegInstr);

  const int SetRegWaitStates = ST.getSetRegWaitStates();
  auto IsHazardFn = [TII, HWReg] (MachineInstr *MI) {
    return HWReg == getHWReg(TII, *MI);
  };
  int WaitStatesNeeded = getWaitStatesSinceSetReg(IsHazardFn, SetRegWaitStates);
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

int
GCNHazardRecognizer::checkVALUHazardsHelper(const MachineOperand &Def,
                                            const MachineRegisterInfo &MRI) {
  // Helper to check for the hazard where VMEM instructions that store more than
  // 8 bytes can have there store data over written by the next instruction.
  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  const int VALUWaitStates = 1;
  int WaitStatesNeeded = 0;

  if (!TRI->isVGPR(MRI, Def.getReg()))
    return WaitStatesNeeded;
  Register Reg = Def.getReg();
  auto IsHazardFn = [this, Reg, TRI] (MachineInstr *MI) {
    int DataIdx = createsVALUHazard(*MI);
    return DataIdx >= 0 &&
    TRI->regsOverlap(MI->getOperand(DataIdx).getReg(), Reg);
  };
  int WaitStatesNeededForDef =
    VALUWaitStates - getWaitStatesSince(IsHazardFn, VALUWaitStates);
  WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForDef);

  return WaitStatesNeeded;
}

int GCNHazardRecognizer::checkVALUHazards(MachineInstr *VALU) {
  // This checks for the hazard where VMEM instructions that store more than
  // 8 bytes can have there store data over written by the next instruction.
  if (!ST.has12DWordStoreHazard())
    return 0;

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  int WaitStatesNeeded = 0;

  for (const MachineOperand &Def : VALU->defs()) {
    WaitStatesNeeded = std::max(WaitStatesNeeded, checkVALUHazardsHelper(Def, MRI));
  }

  return WaitStatesNeeded;
}

int GCNHazardRecognizer::checkInlineAsmHazards(MachineInstr *IA) {
  // This checks for hazards associated with inline asm statements.
  // Since inline asms can contain just about anything, we use this
  // to call/leverage other check*Hazard routines. Note that
  // this function doesn't attempt to address all possible inline asm
  // hazards (good luck), but is a collection of what has been
  // problematic thus far.

  // see checkVALUHazards()
  if (!ST.has12DWordStoreHazard())
    return 0;

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  int WaitStatesNeeded = 0;

  for (unsigned I = InlineAsm::MIOp_FirstOperand, E = IA->getNumOperands();
       I != E; ++I) {
    const MachineOperand &Op = IA->getOperand(I);
    if (Op.isReg() && Op.isDef()) {
      WaitStatesNeeded = std::max(WaitStatesNeeded, checkVALUHazardsHelper(Op, MRI));
    }
  }

  return WaitStatesNeeded;
}

int GCNHazardRecognizer::checkRWLaneHazards(MachineInstr *RWLane) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  const MachineOperand *LaneSelectOp =
      TII->getNamedOperand(*RWLane, AMDGPU::OpName::src1);

  if (!LaneSelectOp->isReg() || !TRI->isSGPRReg(MRI, LaneSelectOp->getReg()))
    return 0;

  Register LaneSelectReg = LaneSelectOp->getReg();
  auto IsHazardFn = [TII] (MachineInstr *MI) {
    return TII->isVALU(*MI);
  };

  const int RWLaneWaitStates = 4;
  int WaitStatesSince = getWaitStatesSinceDef(LaneSelectReg, IsHazardFn,
                                              RWLaneWaitStates);
  return RWLaneWaitStates - WaitStatesSince;
}

int GCNHazardRecognizer::checkRFEHazards(MachineInstr *RFE) {
  if (!ST.hasRFEHazards())
    return 0;

  const SIInstrInfo *TII = ST.getInstrInfo();

  const int RFEWaitStates = 1;

  auto IsHazardFn = [TII] (MachineInstr *MI) {
    return getHWReg(TII, *MI) == AMDGPU::Hwreg::ID_TRAPSTS;
  };
  int WaitStatesNeeded = getWaitStatesSinceSetReg(IsHazardFn, RFEWaitStates);
  return RFEWaitStates - WaitStatesNeeded;
}

int GCNHazardRecognizer::checkReadM0Hazards(MachineInstr *MI) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  const int SMovRelWaitStates = 1;
  auto IsHazardFn = [TII] (MachineInstr *MI) {
    return TII->isSALU(*MI);
  };
  return SMovRelWaitStates - getWaitStatesSinceDef(AMDGPU::M0, IsHazardFn,
                                                   SMovRelWaitStates);
}

void GCNHazardRecognizer::fixHazards(MachineInstr *MI) {
  fixVMEMtoScalarWriteHazards(MI);
  fixVcmpxPermlaneHazards(MI);
  fixSMEMtoVectorWriteHazards(MI);
  fixVcmpxExecWARHazard(MI);
  fixLdsBranchVmemWARHazard(MI);
}

bool GCNHazardRecognizer::fixVcmpxPermlaneHazards(MachineInstr *MI) {
  if (!ST.hasVcmpxPermlaneHazard() || !isPermlane(*MI))
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();
  auto IsHazardFn = [TII] (MachineInstr *MI) {
    return TII->isVOPC(*MI);
  };

  auto IsExpiredFn = [] (MachineInstr *MI, int) {
    if (!MI)
      return false;
    unsigned Opc = MI->getOpcode();
    return SIInstrInfo::isVALU(*MI) &&
           Opc != AMDGPU::V_NOP_e32 &&
           Opc != AMDGPU::V_NOP_e64 &&
           Opc != AMDGPU::V_NOP_sdwa;
  };

  if (::getWaitStatesSince(IsHazardFn, MI, IsExpiredFn) ==
      std::numeric_limits<int>::max())
    return false;

  // V_NOP will be discarded by SQ.
  // Use V_MOB_B32 v?, v?. Register must be alive so use src0 of V_PERMLANE*
  // which is always a VGPR and available.
  auto *Src0 = TII->getNamedOperand(*MI, AMDGPU::OpName::src0);
  Register Reg = Src0->getReg();
  bool IsUndef = Src0->isUndef();
  BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
          TII->get(AMDGPU::V_MOV_B32_e32))
    .addReg(Reg, RegState::Define | (IsUndef ? RegState::Dead : 0))
    .addReg(Reg, IsUndef ? RegState::Undef : RegState::Kill);

  return true;
}

bool GCNHazardRecognizer::fixVMEMtoScalarWriteHazards(MachineInstr *MI) {
  if (!ST.hasVMEMtoScalarWriteHazard())
    return false;

  if (!SIInstrInfo::isSALU(*MI) && !SIInstrInfo::isSMRD(*MI))
    return false;

  if (MI->getNumDefs() == 0)
    return false;

  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  auto IsHazardFn = [TRI, MI] (MachineInstr *I) {
    if (!SIInstrInfo::isVMEM(*I) && !SIInstrInfo::isDS(*I) &&
        !SIInstrInfo::isFLAT(*I))
      return false;

    for (const MachineOperand &Def : MI->defs()) {
      MachineOperand *Op = I->findRegisterUseOperand(Def.getReg(), false, TRI);
      if (!Op)
        continue;
      return true;
    }
    return false;
  };

  auto IsExpiredFn = [](MachineInstr *MI, int) {
    return MI && (SIInstrInfo::isVALU(*MI) ||
                  (MI->getOpcode() == AMDGPU::S_WAITCNT &&
                   !MI->getOperand(0).getImm()) ||
                  (MI->getOpcode() == AMDGPU::S_WAITCNT_DEPCTR &&
                   MI->getOperand(0).getImm() == 0xffe3));
  };

  if (::getWaitStatesSince(IsHazardFn, MI, IsExpiredFn) ==
      std::numeric_limits<int>::max())
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();
  BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
          TII->get(AMDGPU::S_WAITCNT_DEPCTR))
      .addImm(0xffe3);
  return true;
}

bool GCNHazardRecognizer::fixSMEMtoVectorWriteHazards(MachineInstr *MI) {
  if (!ST.hasSMEMtoVectorWriteHazard())
    return false;

  if (!SIInstrInfo::isVALU(*MI))
    return false;

  unsigned SDSTName;
  switch (MI->getOpcode()) {
  case AMDGPU::V_READLANE_B32:
  case AMDGPU::V_READLANE_B32_gfx10:
  case AMDGPU::V_READFIRSTLANE_B32:
    SDSTName = AMDGPU::OpName::vdst;
    break;
  default:
    SDSTName = AMDGPU::OpName::sdst;
    break;
  }

  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST.getCPU());
  const MachineOperand *SDST = TII->getNamedOperand(*MI, SDSTName);
  if (!SDST) {
    for (const auto &MO : MI->implicit_operands()) {
      if (MO.isDef() && TRI->isSGPRClass(TRI->getPhysRegClass(MO.getReg()))) {
        SDST = &MO;
        break;
      }
    }
  }

  if (!SDST)
    return false;

  const Register SDSTReg = SDST->getReg();
  auto IsHazardFn = [SDSTReg, TRI] (MachineInstr *I) {
    return SIInstrInfo::isSMRD(*I) && I->readsRegister(SDSTReg, TRI);
  };

  auto IsExpiredFn = [TII, IV] (MachineInstr *MI, int) {
    if (MI) {
      if (TII->isSALU(*MI)) {
        switch (MI->getOpcode()) {
        case AMDGPU::S_SETVSKIP:
        case AMDGPU::S_VERSION:
        case AMDGPU::S_WAITCNT_VSCNT:
        case AMDGPU::S_WAITCNT_VMCNT:
        case AMDGPU::S_WAITCNT_EXPCNT:
          // These instructions cannot not mitigate the hazard.
          return false;
        case AMDGPU::S_WAITCNT_LGKMCNT:
          // Reducing lgkmcnt count to 0 always mitigates the hazard.
          return (MI->getOperand(1).getImm() == 0) &&
                 (MI->getOperand(0).getReg() == AMDGPU::SGPR_NULL);
        case AMDGPU::S_WAITCNT: {
          const int64_t Imm = MI->getOperand(0).getImm();
          AMDGPU::Waitcnt Decoded = AMDGPU::decodeWaitcnt(IV, Imm);
          return (Decoded.LgkmCnt == 0);
        }
        default:
          // SOPP instructions cannot mitigate the hazard.
          if (TII->isSOPP(*MI))
            return false;
          // At this point the SALU can be assumed to mitigate the hazard
          // because either:
          // (a) it is independent of the at risk SMEM (breaking chain),
          // or
          // (b) it is dependent on the SMEM, in which case an appropriate
          //     s_waitcnt lgkmcnt _must_ exist between it and the at risk
          //     SMEM instruction.
          return true;
        }
      }
    }
    return false;
  };

  if (::getWaitStatesSince(IsHazardFn, MI, IsExpiredFn) ==
      std::numeric_limits<int>::max())
    return false;

  BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
          TII->get(AMDGPU::S_MOV_B32), AMDGPU::SGPR_NULL)
      .addImm(0);
  return true;
}

bool GCNHazardRecognizer::fixVcmpxExecWARHazard(MachineInstr *MI) {
  if (!ST.hasVcmpxExecWARHazard() || !SIInstrInfo::isVALU(*MI))
    return false;

  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  if (!MI->modifiesRegister(AMDGPU::EXEC, TRI))
    return false;

  auto IsHazardFn = [TRI] (MachineInstr *I) {
    if (SIInstrInfo::isVALU(*I))
      return false;
    return I->readsRegister(AMDGPU::EXEC, TRI);
  };

  const SIInstrInfo *TII = ST.getInstrInfo();
  auto IsExpiredFn = [TII, TRI] (MachineInstr *MI, int) {
    if (!MI)
      return false;
    if (SIInstrInfo::isVALU(*MI)) {
      if (TII->getNamedOperand(*MI, AMDGPU::OpName::sdst))
        return true;
      for (auto MO : MI->implicit_operands())
        if (MO.isDef() && TRI->isSGPRClass(TRI->getPhysRegClass(MO.getReg())))
          return true;
    }
    if (MI->getOpcode() == AMDGPU::S_WAITCNT_DEPCTR &&
        (MI->getOperand(0).getImm() & 0xfffe) == 0xfffe)
      return true;
    return false;
  };

  if (::getWaitStatesSince(IsHazardFn, MI, IsExpiredFn) ==
      std::numeric_limits<int>::max())
    return false;

  BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
          TII->get(AMDGPU::S_WAITCNT_DEPCTR))
    .addImm(0xfffe);
  return true;
}

bool GCNHazardRecognizer::fixLdsBranchVmemWARHazard(MachineInstr *MI) {
  if (!ST.hasLdsBranchVmemWARHazard())
    return false;

  auto IsHazardInst = [] (const MachineInstr *MI) {
    if (SIInstrInfo::isDS(*MI))
      return 1;
    if (SIInstrInfo::isVMEM(*MI) || SIInstrInfo::isSegmentSpecificFLAT(*MI))
      return 2;
    return 0;
  };

  auto InstType = IsHazardInst(MI);
  if (!InstType)
    return false;

  auto IsExpiredFn = [&IsHazardInst] (MachineInstr *I, int) {
    return I && (IsHazardInst(I) ||
                 (I->getOpcode() == AMDGPU::S_WAITCNT_VSCNT &&
                  I->getOperand(0).getReg() == AMDGPU::SGPR_NULL &&
                  !I->getOperand(1).getImm()));
  };

  auto IsHazardFn = [InstType, &IsHazardInst] (MachineInstr *I) {
    if (!I->isBranch())
      return false;

    auto IsHazardFn = [InstType, IsHazardInst] (MachineInstr *I) {
      auto InstType2 = IsHazardInst(I);
      return InstType2 && InstType != InstType2;
    };

    auto IsExpiredFn = [InstType, &IsHazardInst] (MachineInstr *I, int) {
      if (!I)
        return false;

      auto InstType2 = IsHazardInst(I);
      if (InstType == InstType2)
        return true;

      return I->getOpcode() == AMDGPU::S_WAITCNT_VSCNT &&
             I->getOperand(0).getReg() == AMDGPU::SGPR_NULL &&
             !I->getOperand(1).getImm();
    };

    return ::getWaitStatesSince(IsHazardFn, I, IsExpiredFn) !=
           std::numeric_limits<int>::max();
  };

  if (::getWaitStatesSince(IsHazardFn, MI, IsExpiredFn) ==
      std::numeric_limits<int>::max())
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();
  BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
          TII->get(AMDGPU::S_WAITCNT_VSCNT))
    .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
    .addImm(0);

  return true;
}

int GCNHazardRecognizer::checkNSAtoVMEMHazard(MachineInstr *MI) {
  int NSAtoVMEMWaitStates = 1;

  if (!ST.hasNSAtoVMEMBug())
    return 0;

  if (!SIInstrInfo::isMUBUF(*MI) && !SIInstrInfo::isMTBUF(*MI))
    return 0;

  const SIInstrInfo *TII = ST.getInstrInfo();
  const auto *Offset = TII->getNamedOperand(*MI, AMDGPU::OpName::offset);
  if (!Offset || (Offset->getImm() & 6) == 0)
    return 0;

  auto IsHazardFn = [TII] (MachineInstr *I) {
    if (!SIInstrInfo::isMIMG(*I))
      return false;
    const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(I->getOpcode());
    return Info->MIMGEncoding == AMDGPU::MIMGEncGfx10NSA &&
           TII->getInstSizeInBytes(*I) >= 16;
  };

  return NSAtoVMEMWaitStates - getWaitStatesSince(IsHazardFn, 1);
}

int GCNHazardRecognizer::checkFPAtomicToDenormModeHazard(MachineInstr *MI) {
  int FPAtomicToDenormModeWaitStates = 3;

  if (MI->getOpcode() != AMDGPU::S_DENORM_MODE)
    return 0;

  auto IsHazardFn = [] (MachineInstr *I) {
    if (!SIInstrInfo::isVMEM(*I) && !SIInstrInfo::isFLAT(*I))
      return false;
    return SIInstrInfo::isFPAtomic(*I);
  };

  auto IsExpiredFn = [] (MachineInstr *MI, int WaitStates) {
    if (WaitStates >= 3 || SIInstrInfo::isVALU(*MI))
      return true;

    switch (MI->getOpcode()) {
    case AMDGPU::S_WAITCNT:
    case AMDGPU::S_WAITCNT_VSCNT:
    case AMDGPU::S_WAITCNT_VMCNT:
    case AMDGPU::S_WAITCNT_EXPCNT:
    case AMDGPU::S_WAITCNT_LGKMCNT:
    case AMDGPU::S_WAITCNT_IDLE:
      return true;
    default:
      break;
    }

    return false;
  };


  return FPAtomicToDenormModeWaitStates -
         ::getWaitStatesSince(IsHazardFn, MI, IsExpiredFn);
}

int GCNHazardRecognizer::checkMAIHazards(MachineInstr *MI) {
  assert(SIInstrInfo::isMAI(*MI));

  int WaitStatesNeeded = 0;
  unsigned Opc = MI->getOpcode();

  auto IsVALUFn = [] (MachineInstr *MI) {
    return SIInstrInfo::isVALU(*MI);
  };

  if (Opc != AMDGPU::V_ACCVGPR_READ_B32) { // MFMA or v_accvgpr_write
    const int LegacyVALUWritesVGPRWaitStates = 2;
    const int VALUWritesExecWaitStates = 4;
    const int MaxWaitStates = 4;

    int WaitStatesNeededForUse = VALUWritesExecWaitStates -
      getWaitStatesSinceDef(AMDGPU::EXEC, IsVALUFn, MaxWaitStates);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);

    if (WaitStatesNeeded < MaxWaitStates) {
      for (const MachineOperand &Use : MI->explicit_uses()) {
        const int MaxWaitStates = 2;

        if (!Use.isReg() || !TRI.isVGPR(MF.getRegInfo(), Use.getReg()))
          continue;

        int WaitStatesNeededForUse = LegacyVALUWritesVGPRWaitStates -
          getWaitStatesSinceDef(Use.getReg(), IsVALUFn, MaxWaitStates);
        WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);

        if (WaitStatesNeeded == MaxWaitStates)
          break;
      }
    }
  }

  auto IsMFMAFn = [] (MachineInstr *MI) {
    return SIInstrInfo::isMAI(*MI) &&
           MI->getOpcode() != AMDGPU::V_ACCVGPR_WRITE_B32 &&
           MI->getOpcode() != AMDGPU::V_ACCVGPR_READ_B32;
  };

  for (const MachineOperand &Op : MI->explicit_operands()) {
    if (!Op.isReg() || !TRI.isAGPR(MF.getRegInfo(), Op.getReg()))
      continue;

    if (Op.isDef() && Opc != AMDGPU::V_ACCVGPR_WRITE_B32)
      continue;

    const int MFMAWritesAGPROverlappedSrcABWaitStates = 4;
    const int MFMAWritesAGPROverlappedSrcCWaitStates = 2;
    const int MFMA4x4WritesAGPRAccVgprReadWaitStates = 4;
    const int MFMA16x16WritesAGPRAccVgprReadWaitStates = 10;
    const int MFMA32x32WritesAGPRAccVgprReadWaitStates = 18;
    const int MFMA4x4WritesAGPRAccVgprWriteWaitStates = 1;
    const int MFMA16x16WritesAGPRAccVgprWriteWaitStates = 7;
    const int MFMA32x32WritesAGPRAccVgprWriteWaitStates = 15;
    const int MaxWaitStates = 18;
    Register Reg = Op.getReg();
    unsigned HazardDefLatency = 0;

    auto IsOverlappedMFMAFn = [Reg, &IsMFMAFn, &HazardDefLatency, this]
                              (MachineInstr *MI) {
      if (!IsMFMAFn(MI))
        return false;
      Register DstReg = MI->getOperand(0).getReg();
      if (DstReg == Reg)
        return false;
      HazardDefLatency = std::max(HazardDefLatency,
                                  TSchedModel.computeInstrLatency(MI));
      return TRI.regsOverlap(DstReg, Reg);
    };

    int WaitStatesSinceDef = getWaitStatesSinceDef(Reg, IsOverlappedMFMAFn,
                                                   MaxWaitStates);
    int NeedWaitStates = MFMAWritesAGPROverlappedSrcABWaitStates;
    int SrcCIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src2);
    int OpNo = MI->getOperandNo(&Op);
    if (OpNo == SrcCIdx) {
      NeedWaitStates = MFMAWritesAGPROverlappedSrcCWaitStates;
    } else if (Opc == AMDGPU::V_ACCVGPR_READ_B32) {
      switch (HazardDefLatency) {
      case 2:  NeedWaitStates = MFMA4x4WritesAGPRAccVgprReadWaitStates;
               break;
      case 8:  NeedWaitStates = MFMA16x16WritesAGPRAccVgprReadWaitStates;
               break;
      case 16: LLVM_FALLTHROUGH;
      default: NeedWaitStates = MFMA32x32WritesAGPRAccVgprReadWaitStates;
               break;
      }
    } else if (Opc == AMDGPU::V_ACCVGPR_WRITE_B32) {
      switch (HazardDefLatency) {
      case 2:  NeedWaitStates = MFMA4x4WritesAGPRAccVgprWriteWaitStates;
               break;
      case 8:  NeedWaitStates = MFMA16x16WritesAGPRAccVgprWriteWaitStates;
               break;
      case 16: LLVM_FALLTHROUGH;
      default: NeedWaitStates = MFMA32x32WritesAGPRAccVgprWriteWaitStates;
               break;
      }
    }

    int WaitStatesNeededForUse = NeedWaitStates - WaitStatesSinceDef;
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);

    if (WaitStatesNeeded == MaxWaitStates)
      return WaitStatesNeeded; // Early exit.

    auto IsAccVgprWriteFn = [Reg, this] (MachineInstr *MI) {
      if (MI->getOpcode() != AMDGPU::V_ACCVGPR_WRITE_B32)
        return false;
      Register DstReg = MI->getOperand(0).getReg();
      return TRI.regsOverlap(Reg, DstReg);
    };

    const int AccVGPRWriteMFMAReadSrcCWaitStates = 1;
    const int AccVGPRWriteMFMAReadSrcABWaitStates = 3;
    const int AccVGPRWriteAccVgprReadWaitStates = 3;
    NeedWaitStates = AccVGPRWriteMFMAReadSrcABWaitStates;
    if (OpNo == SrcCIdx)
      NeedWaitStates = AccVGPRWriteMFMAReadSrcCWaitStates;
    else if (Opc == AMDGPU::V_ACCVGPR_READ_B32)
      NeedWaitStates = AccVGPRWriteAccVgprReadWaitStates;

    WaitStatesNeededForUse = NeedWaitStates -
      getWaitStatesSinceDef(Reg, IsAccVgprWriteFn, MaxWaitStates);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);

    if (WaitStatesNeeded == MaxWaitStates)
      return WaitStatesNeeded; // Early exit.
  }

  if (Opc == AMDGPU::V_ACCVGPR_WRITE_B32) {
    const int MFMA4x4ReadSrcCAccVgprWriteWaitStates = 0;
    const int MFMA16x16ReadSrcCAccVgprWriteWaitStates = 5;
    const int MFMA32x32ReadSrcCAccVgprWriteWaitStates = 13;
    const int MaxWaitStates = 13;
    Register DstReg = MI->getOperand(0).getReg();
    unsigned HazardDefLatency = 0;

    auto IsSrcCMFMAFn = [DstReg, &IsMFMAFn, &HazardDefLatency, this]
                         (MachineInstr *MI) {
      if (!IsMFMAFn(MI))
        return false;
      Register Reg = TII.getNamedOperand(*MI, AMDGPU::OpName::src2)->getReg();
      HazardDefLatency = std::max(HazardDefLatency,
                                  TSchedModel.computeInstrLatency(MI));
      return TRI.regsOverlap(Reg, DstReg);
    };

    int WaitStatesSince = getWaitStatesSince(IsSrcCMFMAFn, MaxWaitStates);
    int NeedWaitStates;
    switch (HazardDefLatency) {
    case 2:  NeedWaitStates = MFMA4x4ReadSrcCAccVgprWriteWaitStates;
             break;
    case 8:  NeedWaitStates = MFMA16x16ReadSrcCAccVgprWriteWaitStates;
             break;
    case 16: LLVM_FALLTHROUGH;
    default: NeedWaitStates = MFMA32x32ReadSrcCAccVgprWriteWaitStates;
             break;
    }

    int WaitStatesNeededForUse = NeedWaitStates - WaitStatesSince;
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);
  }

  return WaitStatesNeeded;
}

int GCNHazardRecognizer::checkMAILdStHazards(MachineInstr *MI) {
  if (!ST.hasMAIInsts())
    return 0;

  int WaitStatesNeeded = 0;

  auto IsAccVgprReadFn = [] (MachineInstr *MI) {
    return MI->getOpcode() == AMDGPU::V_ACCVGPR_READ_B32;
  };

  for (const MachineOperand &Op : MI->explicit_uses()) {
    if (!Op.isReg() || !TRI.isVGPR(MF.getRegInfo(), Op.getReg()))
      continue;

    Register Reg = Op.getReg();

    const int AccVgprReadLdStWaitStates = 2;
    const int VALUWriteAccVgprReadLdStDepVALUWaitStates = 1;
    const int MaxWaitStates = 2;

    int WaitStatesNeededForUse = AccVgprReadLdStWaitStates -
      getWaitStatesSinceDef(Reg, IsAccVgprReadFn, MaxWaitStates);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);

    if (WaitStatesNeeded == MaxWaitStates)
      return WaitStatesNeeded; // Early exit.

    auto IsVALUAccVgprReadCheckFn = [Reg, this] (MachineInstr *MI) {
      if (MI->getOpcode() != AMDGPU::V_ACCVGPR_READ_B32)
        return false;
      auto IsVALUFn = [] (MachineInstr *MI) {
        return SIInstrInfo::isVALU(*MI) && !SIInstrInfo::isMAI(*MI);
      };
      return getWaitStatesSinceDef(Reg, IsVALUFn, 2 /*MaxWaitStates*/) <
             std::numeric_limits<int>::max();
    };

    WaitStatesNeededForUse = VALUWriteAccVgprReadLdStDepVALUWaitStates -
      getWaitStatesSince(IsVALUAccVgprReadCheckFn, MaxWaitStates);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);
  }

  return WaitStatesNeeded;
}

bool GCNHazardRecognizer::ShouldPreferAnother(SUnit *SU) {
  if (!SU->isInstr())
    return false;

  MachineInstr *MAI = nullptr;
  auto IsMFMAFn = [&MAI] (MachineInstr *MI) {
    MAI = nullptr;
    if (SIInstrInfo::isMAI(*MI) &&
        MI->getOpcode() != AMDGPU::V_ACCVGPR_WRITE_B32 &&
        MI->getOpcode() != AMDGPU::V_ACCVGPR_READ_B32)
      MAI = MI;
    return MAI != nullptr;
  };

  MachineInstr *MI = SU->getInstr();
  if (IsMFMAFn(MI)) {
    int W = getWaitStatesSince(IsMFMAFn, 16);
    if (MAI)
      return W < (int)TSchedModel.computeInstrLatency(MAI);
  }

  return false;
}
