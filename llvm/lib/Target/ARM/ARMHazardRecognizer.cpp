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
#include "ARMSubtarget.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
using namespace llvm;

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
