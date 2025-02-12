//===-- ARMHazardRecognizer.cpp - ARM postra hazard recognizer ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ARMHazardRecognizer.h"
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMInstrInfo.h"
#include "ARMSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>

using namespace llvm;

static cl::opt<int> DataBankMask("arm-data-bank-mask", cl::init(-1),
                                 cl::Hidden);
static cl::opt<bool> AssumeITCMConflict("arm-assume-itcm-bankconflict",
                                        cl::init(false), cl::Hidden);

static bool hasRAWHazard(MachineInstr *DefMI, MachineInstr *MI,
                         const TargetRegisterInfo &TRI) {
  // FIXME: Detect integer instructions properly.
  const MCInstrDesc &MCID = MI->getDesc();
  unsigned Domain = MCID.TSFlags & ARMII::DomainMask;
  if (MI->mayStore())
    return false;
  unsigned Opcode = MCID.getOpcode();
  if (Opcode == ARM::VMOVRS || Opcode == ARM::VMOVRRD)
    return false;
  if ((Domain & ARMII::DomainVFP) || (Domain & ARMII::DomainNEON))
    return MI->readsRegister(DefMI->getOperand(0).getReg(), &TRI);
  return false;
}

ScheduleHazardRecognizer::HazardType
ARMHazardRecognizerFPMLx::getHazardType(SUnit *SU, int Stalls) {
  assert(Stalls == 0 && "ARM hazards don't support scoreboard lookahead");

  MachineInstr *MI = SU->getInstr();

  if (!MI->isDebugInstr()) {
    // Look for special VMLA / VMLS hazards. A VMUL / VADD / VSUB following
    // a VMLA / VMLS will cause 4 cycle stall.
    const MCInstrDesc &MCID = MI->getDesc();
    if (LastMI && (MCID.TSFlags & ARMII::DomainMask) != ARMII::DomainGeneral) {
      MachineInstr *DefMI = LastMI;
      const MCInstrDesc &LastMCID = LastMI->getDesc();
      const MachineFunction *MF = MI->getParent()->getParent();
      const ARMBaseInstrInfo &TII = *static_cast<const ARMBaseInstrInfo *>(
                                        MF->getSubtarget().getInstrInfo());

      // Skip over one non-VFP / NEON instruction.
      if (!LastMI->isBarrier() &&
          !(TII.getSubtarget().hasMuxedUnits() && LastMI->mayLoadOrStore()) &&
          (LastMCID.TSFlags & ARMII::DomainMask) == ARMII::DomainGeneral) {
        MachineBasicBlock::iterator I = LastMI;
        if (I != LastMI->getParent()->begin()) {
          I = std::prev(I);
          DefMI = &*I;
        }
      }

      if (TII.isFpMLxInstruction(DefMI->getOpcode()) &&
          (TII.canCauseFpMLxStall(MI->getOpcode()) ||
           hasRAWHazard(DefMI, MI, TII.getRegisterInfo()))) {
        // Try to schedule another instruction for the next 4 cycles.
        if (FpMLxStalls == 0)
          FpMLxStalls = 4;
        return Hazard;
      }
    }
  }
  return NoHazard;
}

void ARMHazardRecognizerFPMLx::Reset() {
  LastMI = nullptr;
  FpMLxStalls = 0;
}

void ARMHazardRecognizerFPMLx::EmitInstruction(SUnit *SU) {
  MachineInstr *MI = SU->getInstr();
  if (!MI->isDebugInstr()) {
    LastMI = MI;
    FpMLxStalls = 0;
  }
}

void ARMHazardRecognizerFPMLx::AdvanceCycle() {
  if (FpMLxStalls && --FpMLxStalls == 0)
    // Stalled for 4 cycles but still can't schedule any other instructions.
    LastMI = nullptr;
}

void ARMHazardRecognizerFPMLx::RecedeCycle() {
  llvm_unreachable("reverse ARM hazard checking unsupported");
}

///////// Bank conflicts handled as hazards //////////////

static bool getBaseOffset(const MachineInstr &MI, const MachineOperand *&BaseOp,
                          int64_t &Offset) {

  uint64_t TSFlags = MI.getDesc().TSFlags;
  unsigned AddrMode = (TSFlags & ARMII::AddrModeMask);
  unsigned IndexMode =
      (TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift;

  // Address mode tells us what we want to know about operands for T2
  // instructions (but not size).  It tells us size (but not about operands)
  // for T1 instructions.
  switch (AddrMode) {
  default:
    return false;
  case ARMII::AddrModeT2_i8:
    // t2LDRBT, t2LDRB_POST, t2LDRB_PRE, t2LDRBi8,
    // t2LDRHT, t2LDRH_POST, t2LDRH_PRE, t2LDRHi8,
    // t2LDRSBT, t2LDRSB_POST, t2LDRSB_PRE, t2LDRSBi8,
    // t2LDRSHT, t2LDRSH_POST, t2LDRSH_PRE, t2LDRSHi8,
    // t2LDRT, t2LDR_POST, t2LDR_PRE, t2LDRi8
    BaseOp = &MI.getOperand(1);
    Offset = (IndexMode == ARMII::IndexModePost)
                 ? 0
                 : (IndexMode == ARMII::IndexModePre ||
                    IndexMode == ARMII::IndexModeUpd)
                       ? MI.getOperand(3).getImm()
                       : MI.getOperand(2).getImm();
    return true;
  case ARMII::AddrModeT2_i12:
    // t2LDRBi12, t2LDRHi12
    // t2LDRSBi12, t2LDRSHi12
    // t2LDRi12
    BaseOp = &MI.getOperand(1);
    Offset = MI.getOperand(2).getImm();
    return true;
  case ARMII::AddrModeT2_i8s4:
    // t2LDRD_POST, t2LDRD_PRE, t2LDRDi8
    BaseOp = &MI.getOperand(2);
    Offset = (IndexMode == ARMII::IndexModePost)
                 ? 0
                 : (IndexMode == ARMII::IndexModePre ||
                    IndexMode == ARMII::IndexModeUpd)
                       ? MI.getOperand(4).getImm()
                       : MI.getOperand(3).getImm();
    return true;
  case ARMII::AddrModeT1_1:
    // tLDRBi, tLDRBr (watch out!), TLDRSB
  case ARMII::AddrModeT1_2:
    // tLDRHi, tLDRHr (watch out!), TLDRSH
  case ARMII::AddrModeT1_4:
    // tLDRi, tLDRr (watch out!)
    BaseOp = &MI.getOperand(1);
    Offset = MI.getOperand(2).isImm() ? MI.getOperand(2).getImm() : 0;
    return MI.getOperand(2).isImm();
  }
  return false;
}

ARMBankConflictHazardRecognizer::ARMBankConflictHazardRecognizer(
    const ScheduleDAG *DAG, int64_t CPUBankMask, bool CPUAssumeITCMConflict)
    : MF(DAG->MF), DL(DAG->MF.getDataLayout()),
      DataMask(DataBankMask.getNumOccurrences() ? int64_t(DataBankMask)
                                                : CPUBankMask),
      AssumeITCMBankConflict(AssumeITCMConflict.getNumOccurrences()
                                 ? AssumeITCMConflict
                                 : CPUAssumeITCMConflict) {
  MaxLookAhead = 1;
}

ScheduleHazardRecognizer::HazardType
ARMBankConflictHazardRecognizer::CheckOffsets(unsigned O0, unsigned O1) {
  return (((O0 ^ O1) & DataMask) != 0) ? NoHazard : Hazard;
}

ScheduleHazardRecognizer::HazardType
ARMBankConflictHazardRecognizer::getHazardType(SUnit *SU, int Stalls) {
  MachineInstr &L0 = *SU->getInstr();
  if (!L0.mayLoad() || L0.mayStore() || L0.getNumMemOperands() != 1)
    return NoHazard;

  auto MO0 = *L0.memoperands().begin();
  auto BaseVal0 = MO0->getValue();
  auto BasePseudoVal0 = MO0->getPseudoValue();
  int64_t Offset0 = 0;

  if (!MO0->getSize().hasValue() || MO0->getSize().getValue() > 4)
    return NoHazard;

  bool SPvalid = false;
  const MachineOperand *SP = nullptr;
  int64_t SPOffset0 = 0;

  for (auto L1 : Accesses) {
    auto MO1 = *L1->memoperands().begin();
    auto BaseVal1 = MO1->getValue();
    auto BasePseudoVal1 = MO1->getPseudoValue();
    int64_t Offset1 = 0;

    // Pointers to the same object
    if (BaseVal0 && BaseVal1) {
      const Value *Ptr0, *Ptr1;
      Ptr0 = GetPointerBaseWithConstantOffset(BaseVal0, Offset0, DL, true);
      Ptr1 = GetPointerBaseWithConstantOffset(BaseVal1, Offset1, DL, true);
      if (Ptr0 == Ptr1 && Ptr0)
        return CheckOffsets(Offset0, Offset1);
    }

    if (BasePseudoVal0 && BasePseudoVal1 &&
        BasePseudoVal0->kind() == BasePseudoVal1->kind() &&
        BasePseudoVal0->kind() == PseudoSourceValue::FixedStack) {
      // Spills/fills
      auto FS0 = cast<FixedStackPseudoSourceValue>(BasePseudoVal0);
      auto FS1 = cast<FixedStackPseudoSourceValue>(BasePseudoVal1);
      Offset0 = MF.getFrameInfo().getObjectOffset(FS0->getFrameIndex());
      Offset1 = MF.getFrameInfo().getObjectOffset(FS1->getFrameIndex());
      return CheckOffsets(Offset0, Offset1);
    }

    // Constant pools (likely in ITCM)
    if (BasePseudoVal0 && BasePseudoVal1 &&
        BasePseudoVal0->kind() == BasePseudoVal1->kind() &&
        BasePseudoVal0->isConstantPool() && AssumeITCMBankConflict)
      return Hazard;

    // Is this a stack pointer-relative access?  We could in general try to
    // use "is this the same register and is it unchanged?", but the
    // memory operand tracking is highly likely to have already found that.
    // What we're after here is bank conflicts between different objects in
    // the stack frame.
    if (!SPvalid) { // set up SP
      if (!getBaseOffset(L0, SP, SPOffset0) || SP->getReg().id() != ARM::SP)
        SP = nullptr;
      SPvalid = true;
    }
    if (SP) {
      int64_t SPOffset1;
      const MachineOperand *SP1;
      if (getBaseOffset(*L1, SP1, SPOffset1) && SP1->getReg().id() == ARM::SP)
        return CheckOffsets(SPOffset0, SPOffset1);
    }
  }

  return NoHazard;
}

void ARMBankConflictHazardRecognizer::Reset() { Accesses.clear(); }

void ARMBankConflictHazardRecognizer::EmitInstruction(SUnit *SU) {
  MachineInstr &MI = *SU->getInstr();
  if (!MI.mayLoad() || MI.mayStore() || MI.getNumMemOperands() != 1)
    return;

  auto MO = *MI.memoperands().begin();
  LocationSize Size1 = MO->getSize();
  if (Size1.hasValue() && Size1.getValue() > 4)
    return;
  Accesses.push_back(&MI);
}

void ARMBankConflictHazardRecognizer::AdvanceCycle() { Accesses.clear(); }

void ARMBankConflictHazardRecognizer::RecedeCycle() { Accesses.clear(); }

#define DEBUG_TYPE "cortex-m4-alignment-hazard-rec"

STATISTIC(NumNoops, "Number of noops inserted");

static cl::opt<bool> LoopsOnly(DEBUG_TYPE "-loops-only", cl::Hidden,
                               cl::init(true),
                               cl::desc("Emit nops only in loops"));

static cl::opt<bool>
    InnermostLoopsOnly(DEBUG_TYPE "-innermost-loops-only", cl::Hidden,
                       cl::init(true),
                       cl::desc("Emit noops only in innermost loops"));

void ARMCortexM4AlignmentHazardRecognizer::Reset() { Offset = 0; }

ARMCortexM4AlignmentHazardRecognizer::ARMCortexM4AlignmentHazardRecognizer(
    const MCSubtargetInfo &STI)
    : STI(STI), MBB(nullptr), MF(nullptr), Offset(0), Advanced(false),
      EmittingNoop(false) {
  MaxLookAhead = 1;
}

void ARMCortexM4AlignmentHazardRecognizer::EmitInstruction(SUnit *SU) {
  if (!SU->isInstr())
    return;

  MachineInstr *MI = SU->getInstr();
  assert(MI);
  return EmitInstruction(MI);
}

void ARMCortexM4AlignmentHazardRecognizer::EmitInstruction(MachineInstr *MI) {
  if (MI->isDebugInstr())
    return;

  unsigned Size = MI->getDesc().getSize();
  Offset += Size;

  // If the previous instruction had a hazard, then we're inserting a nop. Mark
  // it with an AsmPrinter comment.
  if (EmittingNoop)
    if (MachineInstr *Prev = MI->getPrevNode())
      Prev->setAsmPrinterFlag(ARM::M4F_ALIGNMENT_HAZARD);

  EmittingNoop = false;
}

ScheduleHazardRecognizer::HazardType
ARMCortexM4AlignmentHazardRecognizer::getHazardType(SUnit *SU,
                                                    int /*Ignored*/) {
  if (!SU->isInstr())
    return HazardType::NoHazard;

  MachineInstr *MI = SU->getInstr();
  assert(MI);
  return getHazardTypeAssumingOffset(MI, Offset);
}

ScheduleHazardRecognizer::HazardType
ARMCortexM4AlignmentHazardRecognizer::getHazardTypeAssumingOffset(
    MachineInstr *MI, size_t AssumedOffset) {
  if (Advanced) {
    Advanced = false;
    return HazardType::NoHazard;
  }

  if (AssumedOffset % 4 == 0)
    return HazardType::NoHazard;

  const MCSchedModel &SCModel = STI.getSchedModel();
  const MachineFunction *MF = MI->getParent()->getParent();
  const ARMBaseInstrInfo &TII =
      *static_cast<const ARMBaseInstrInfo *>(MF->getSubtarget().getInstrInfo());
  int Latency = SCModel.computeInstrLatency<MCSubtargetInfo, MCInstrInfo,
                                            InstrItineraryData, MachineInstr>(
      STI, TII, *MI);
  if (!Latency)
    return HazardType::NoHazard;

  const MCInstrDesc &MCID = MI->getDesc();
  unsigned Domain = MCID.TSFlags & ARMII::DomainMask;

  bool SingleCycleFP =
      Latency == 1 && (Domain & (ARMII::DomainNEON | ARMII::DomainVFP));
  if (SingleCycleFP)
    return HazardType::NoopHazard;

  if (MCID.getSize() == 4 && (MI->mayLoad() || MI->mayStore()))
    return HazardType::NoopHazard;

  return HazardType::NoHazard;
}

void ARMCortexM4AlignmentHazardRecognizer::AdvanceCycle() { Advanced = true; }
void ARMCortexM4AlignmentHazardRecognizer::RecedeCycle() {}

void ARMCortexM4AlignmentHazardRecognizer::EmitNoop() { Offset += 2; }

unsigned ARMCortexM4AlignmentHazardRecognizer::PreEmitNoops(SUnit *SU) {
  if (!SU->isInstr())
    return 0;

  MachineInstr *MI = SU->getInstr();
  assert(MI);
  return PreEmitNoops(MI);
}

unsigned ARMCortexM4AlignmentHazardRecognizer::PreEmitNoops(MachineInstr *MI) {
  const MachineBasicBlock *Parent = MI->getParent();
  if (Parent != MBB) {
    Offset = 0;
    MBB = Parent;
  }

  LLVM_DEBUG(MI->dump());

  if (LoopsOnly) {
    // This optimization is likely only critical in loops. Try to save code size
    // elsewhere by avoiding it when we're not in an innermost loop.
    if (const MachineLoop *Loop = getLoopFor(MI)) {
      if (InnermostLoopsOnly && !Loop->isInnermost()) {
        LLVM_DEBUG(dbgs() << "\toffset=0x" << utohexstr(Offset)
                          << "\n\tnot in an innermost loop\n");
        return 0;
      }
    } else if (LoopsOnly) {
      LLVM_DEBUG(dbgs() << "\toffset=0x" << utohexstr(Offset)
                        << "\n\tnot in a loop\n");
      return 0;
    }
  }

  if (HazardType::NoopHazard == getHazardTypeAssumingOffset(MI, Offset)) {
    EmittingNoop = true;
    NumNoops++;
    LLVM_DEBUG(dbgs() << "\toffset=0x" << utohexstr(Offset)
                      << "\n\thas an alignment hazard, and requires a noop\n");
    return 1;
  }

  return 0;
}

const MachineLoop *
ARMCortexM4AlignmentHazardRecognizer::getLoopFor(MachineInstr *MI) {
  // Calculate and cache the MachineLoopInfo.
  MachineFunction *ParentMF = MI->getParent()->getParent();
  if (MF != ParentMF) {
    MF = ParentMF;
    MDT = MachineDominatorTree(*MF);
    MLI.~MachineLoopInfo();
    new (&MLI) MachineLoopInfo(MDT);
  }
  return MLI.getLoopFor(MI->getParent());
}