//===- GCNVOPDUtils.cpp - GCN VOPD Utils  ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the AMDGPU DAG scheduling
/// mutation to pair VOPD instructions back to back. It also contains
//  subroutines useful in the creation of VOPD instructions
//
//===----------------------------------------------------------------------===//

#include "GCNVOPDUtils.h"
#include "AMDGPUSubtarget.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/MC/MCInst.h"

using namespace llvm;

#define DEBUG_TYPE "gcn-vopd-utils"

bool llvm::checkVOPDRegConstraints(const SIInstrInfo &TII,
                                   const MachineInstr &FirstMI,
                                   const MachineInstr &SecondMI) {
  const MachineFunction *MF = FirstMI.getMF();
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = dyn_cast<SIRegisterInfo>(ST.getRegisterInfo());
  const MachineRegisterInfo &MRI = MF->getRegInfo();
  const unsigned NumVGPRBanks = 4;
  // Literals also count against scalar bus limit
  SmallVector<const MachineOperand *> UniqueLiterals;
  auto addLiteral = [&](const MachineOperand &Op) {
    for (auto &Literal : UniqueLiterals) {
      if (Literal->isIdenticalTo(Op))
        return;
    }
    UniqueLiterals.push_back(&Op);
  };
  SmallVector<Register> UniqueScalarRegs;
  assert([&]() -> bool {
    for (auto MII = MachineBasicBlock::const_iterator(&FirstMI);
         MII != FirstMI.getParent()->instr_end(); ++MII) {
      if (&*MII == &SecondMI)
        return true;
    }
    return false;
  }() && "Expected FirstMI to precede SecondMI");
  // Cannot pair dependent instructions
  for (const auto &Use : SecondMI.uses())
    if (Use.isReg() && FirstMI.modifiesRegister(Use.getReg()))
      return false;

  struct ComponentInfo {
    ComponentInfo(const MachineInstr &MI) : MI(MI) {}
    Register Dst, Reg0, Reg1, Reg2;
    const MachineInstr &MI;
  };
  ComponentInfo CInfo[] = {ComponentInfo(FirstMI), ComponentInfo(SecondMI)};

  for (ComponentInfo &Comp : CInfo) {
    switch (Comp.MI.getOpcode()) {
    case AMDGPU::V_FMAMK_F32:
      // cannot inline the fixed literal in fmamk
      addLiteral(Comp.MI.getOperand(2));
      Comp.Reg2 = Comp.MI.getOperand(3).getReg();
      break;
    case AMDGPU::V_FMAAK_F32:
      // cannot inline the fixed literal in fmaak
      addLiteral(Comp.MI.getOperand(3));
      Comp.Reg1 = Comp.MI.getOperand(2).getReg();
      break;
    case AMDGPU::V_FMAC_F32_e32:
    case AMDGPU::V_DOT2_F32_F16:
    case AMDGPU::V_DOT2_F32_BF16:
      Comp.Reg1 = Comp.MI.getOperand(2).getReg();
      Comp.Reg2 = Comp.MI.getOperand(0).getReg();
      break;
    case AMDGPU::V_CNDMASK_B32_e32:
      UniqueScalarRegs.push_back(AMDGPU::VCC_LO);
      Comp.Reg1 = Comp.MI.getOperand(2).getReg();
      break;
    case AMDGPU::V_MOV_B32_e32:
      break;
    default:
      Comp.Reg1 = Comp.MI.getOperand(2).getReg();
      break;
    }

    Comp.Dst = Comp.MI.getOperand(0).getReg();

    const MachineOperand &Op0 = Comp.MI.getOperand(1);
    if (Op0.isReg()) {
      if (!TRI->isVectorRegister(MRI, Op0.getReg())) {
        if (!is_contained(UniqueScalarRegs, Op0.getReg()))
          UniqueScalarRegs.push_back(Op0.getReg());
      } else
        Comp.Reg0 = Op0.getReg();
    } else {
      if (!TII.isInlineConstant(Comp.MI, 1))
        addLiteral(Op0);
    }
  }

  if (UniqueLiterals.size() > 1)
    return false;
  if ((UniqueLiterals.size() + UniqueScalarRegs.size()) > 2)
    return false;

  // check port 0
  if (CInfo[0].Reg0 && CInfo[1].Reg0 &&
      CInfo[0].Reg0 % NumVGPRBanks == CInfo[1].Reg0 % NumVGPRBanks)
    return false;
  // check port 1
  if (CInfo[0].Reg1 && CInfo[1].Reg1 &&
      CInfo[0].Reg1 % NumVGPRBanks == CInfo[1].Reg1 % NumVGPRBanks)
    return false;
  // check port 2
  if (CInfo[0].Reg2 && CInfo[1].Reg2 &&
      !((CInfo[0].Reg2 ^ CInfo[1].Reg2) & 0x1))
    return false;
  if (!((CInfo[0].Dst ^ CInfo[1].Dst) & 0x1))
    return false;

  LLVM_DEBUG(dbgs() << "VOPD Reg Constraints Passed\n\tX: " << FirstMI
                    << "\n\tY: " << SecondMI << "\n");
  return true;
}

/// Check if the instr pair, FirstMI and SecondMI, should be scheduled
/// together. Given SecondMI, when FirstMI is unspecified, then check if
/// SecondMI may be part of a fused pair at all.
static bool shouldScheduleVOPDAdjacent(const TargetInstrInfo &TII,
                                       const TargetSubtargetInfo &TSI,
                                       const MachineInstr *FirstMI,
                                       const MachineInstr &SecondMI) {
  const SIInstrInfo &STII = static_cast<const SIInstrInfo &>(TII);
  unsigned Opc2 = SecondMI.getOpcode();
  auto SecondCanBeVOPD = AMDGPU::getCanBeVOPD(Opc2);

  // One instruction case
  if (!FirstMI)
    return SecondCanBeVOPD.Y;

  unsigned Opc = FirstMI->getOpcode();
  auto FirstCanBeVOPD = AMDGPU::getCanBeVOPD(Opc);

  if (!((FirstCanBeVOPD.X && SecondCanBeVOPD.Y) ||
        (FirstCanBeVOPD.Y && SecondCanBeVOPD.X)))
    return false;

  return checkVOPDRegConstraints(STII, *FirstMI, SecondMI);
}

/// Adapts design from MacroFusion
/// Puts valid candidate instructions back-to-back so they can easily
/// be turned into VOPD instructions
/// Greedily pairs instruction candidates. O(n^2) algorithm.
struct VOPDPairingMutation : ScheduleDAGMutation {
  ShouldSchedulePredTy shouldScheduleAdjacent; // NOLINT: function pointer

  VOPDPairingMutation(
      ShouldSchedulePredTy shouldScheduleAdjacent) // NOLINT: function pointer
      : shouldScheduleAdjacent(shouldScheduleAdjacent) {}

  void apply(ScheduleDAGInstrs *DAG) override {
    const TargetInstrInfo &TII = *DAG->TII;
    const GCNSubtarget &ST = DAG->MF.getSubtarget<GCNSubtarget>();
    if (!AMDGPU::hasVOPD(ST) || !ST.isWave32()) {
      LLVM_DEBUG(dbgs() << "Target does not support VOPDPairingMutation\n");
      return;
    }

    std::vector<SUnit>::iterator ISUI, JSUI;
    for (ISUI = DAG->SUnits.begin(); ISUI != DAG->SUnits.end(); ++ISUI) {
      const MachineInstr *IMI = ISUI->getInstr();
      if (!shouldScheduleAdjacent(TII, ST, nullptr, *IMI))
        continue;
      if (!hasLessThanNumFused(*ISUI, 2))
        continue;

      for (JSUI = ISUI + 1; JSUI != DAG->SUnits.end(); ++JSUI) {
        if (JSUI->isBoundaryNode())
          continue;
        const MachineInstr *JMI = JSUI->getInstr();
        if (!hasLessThanNumFused(*JSUI, 2) ||
            !shouldScheduleAdjacent(TII, ST, IMI, *JMI))
          continue;
        if (fuseInstructionPair(*DAG, *ISUI, *JSUI))
          break;
      }
    }
    LLVM_DEBUG(dbgs() << "Completed VOPDPairingMutation\n");
  }
};

std::unique_ptr<ScheduleDAGMutation> llvm::createVOPDPairingMutation() {
  return std::make_unique<VOPDPairingMutation>(shouldScheduleVOPDAdjacent);
}
