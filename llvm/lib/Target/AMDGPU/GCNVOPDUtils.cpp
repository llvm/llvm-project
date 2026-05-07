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
#include "Utils/AMDGPUBaseInfo.h"
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
                                   const MachineInstr &MIX,
                                   const MachineInstr &MIY, bool IsVOPD3,
                                   bool AllowSameVGPR) {
  namespace VOPD = AMDGPU::VOPD;

  const MachineFunction *MF = MIX.getMF();
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();

  if (IsVOPD3 && !ST.hasVOPD3())
    return false;
  if (!IsVOPD3 && (TII.isVOP3(MIX) || TII.isVOP3(MIY)))
    return false;
  if (TII.isDPP(MIX) || TII.isDPP(MIY))
    return false;

  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const MachineRegisterInfo &MRI = MF->getRegInfo();
  // Literals also count against scalar bus limit
  SmallVector<const MachineOperand *> UniqueLiterals;
  auto addLiteral = [&](const MachineOperand &Op) {
    for (auto &Literal : UniqueLiterals) {
      if (Literal->isIdenticalTo(Op))
        return;
    }
    UniqueLiterals.push_back(&Op);
  };
  SmallSet<Register, 4> UniqueScalarRegs;

  auto getVRegIdx = [&](unsigned OpcodeIdx, unsigned OperandIdx) {
    const MachineInstr &MI = (OpcodeIdx == VOPD::X) ? MIX : MIY;
    const MachineOperand &Operand = MI.getOperand(OperandIdx);
    if (Operand.isReg() && TRI->isVectorRegister(MRI, Operand.getReg()))
      return Operand.getReg();
    return Register();
  };

  auto InstInfo = AMDGPU::getVOPDInstInfo(MIX.getDesc(), MIY.getDesc());

  for (auto CompIdx : VOPD::COMPONENTS) {
    const MachineInstr &MI = (CompIdx == VOPD::X) ? MIX : MIY;

    const MachineOperand &Src0 = *TII.getNamedOperand(MI, AMDGPU::OpName::src0);
    if (Src0.isReg()) {
      if (!TRI->isVectorRegister(MRI, Src0.getReg())) {
        UniqueScalarRegs.insert(Src0.getReg());
      }
    } else if (!TII.isInlineConstant(Src0)) {
      if (IsVOPD3)
        return false;
      addLiteral(Src0);
    }

    if (InstInfo[CompIdx].hasMandatoryLiteral()) {
      if (IsVOPD3)
        return false;

      auto CompOprIdx = InstInfo[CompIdx].getMandatoryLiteralCompOperandIndex();
      addLiteral(MI.getOperand(CompOprIdx));
    }
    if (MI.getDesc().hasImplicitUseOfPhysReg(AMDGPU::VCC))
      UniqueScalarRegs.insert(AMDGPU::VCC_LO);

    if (IsVOPD3) {
      for (auto OpName : {AMDGPU::OpName::src1, AMDGPU::OpName::src2}) {
        const MachineOperand *Src = TII.getNamedOperand(MI, OpName);
        if (!Src)
          continue;
        if (OpName == AMDGPU::OpName::src2) {
          if (AMDGPU::hasNamedOperand(MI.getOpcode(), AMDGPU::OpName::bitop3))
            continue;
          if (MI.getOpcode() == AMDGPU::V_CNDMASK_B32_e64) {
            UniqueScalarRegs.insert(Src->getReg());
            continue;
          }
        }
        if (!Src->isReg() || !TRI->isVGPR(MRI, Src->getReg()))
          return false;
      }

      for (auto OpName : {AMDGPU::OpName::clamp, AMDGPU::OpName::omod,
                          AMDGPU::OpName::op_sel}) {
        if (TII.hasModifiersSet(MI, OpName))
          return false;
      }

      // Neg is allowed, other modifiers are not. NB: even though sext has the
      // same value as neg, there are no combinable instructions with sext.
      for (auto OpName :
           {AMDGPU::OpName::src0_modifiers, AMDGPU::OpName::src1_modifiers,
            AMDGPU::OpName::src2_modifiers}) {
        const MachineOperand *Mods = TII.getNamedOperand(MI, OpName);
        if (Mods && (Mods->getImm() & ~SISrcMods::NEG))
          return false;
      }
    }
  }

  if (UniqueLiterals.size() > 1)
    return false;
  if ((UniqueLiterals.size() + UniqueScalarRegs.size()) > 2)
    return false;

  // On GFX1170+ if both OpX and OpY are V_MOV_B32 then OPY uses SRC2
  // source-cache.
  bool SkipSrc = (ST.hasGFX11_7Insts() || ST.hasGFX12Insts()) &&
                 MIX.getOpcode() == AMDGPU::V_MOV_B32_e32 &&
                 MIY.getOpcode() == AMDGPU::V_MOV_B32_e32;

  if (InstInfo.hasInvalidOperand(getVRegIdx, *TRI, SkipSrc, AllowSameVGPR,
                                 IsVOPD3))
    return false;

  if (IsVOPD3) {
    // BITOP3 can be converted to DUAL_BITOP2 only if src2 is zero.
    // MIX check is only relevant to scheduling?
    if (AMDGPU::hasNamedOperand(MIX.getOpcode(), AMDGPU::OpName::bitop3)) {
      const MachineOperand &Src2 =
          *TII.getNamedOperand(MIX, AMDGPU::OpName::src2);
      if (!Src2.isImm() || Src2.getImm())
        return false;
    }
    if (AMDGPU::hasNamedOperand(MIY.getOpcode(), AMDGPU::OpName::bitop3)) {
      const MachineOperand &Src2 =
          *TII.getNamedOperand(MIY, AMDGPU::OpName::src2);
      if (!Src2.isImm() || Src2.getImm())
        return false;
    }
  }

  LLVM_DEBUG(dbgs() << "VOPD Reg Constraints Passed\n\tX: " << MIX
                    << "\n\tY: " << MIY << "\n");
  return true;
}

/// Core pair-eligibility check for a single VOPD encoding variant (VOPD or
/// VOPD3).  Returns the X/Y assignment on success, or std::nullopt otherwise.
static std::optional<VOPDMatchInfo>
tryMatchVOPDPairVariant(const SIInstrInfo &TII, unsigned EncodingFamily,
                        MachineInstr &FirstMI, MachineInstr &SecondMI,
                        bool IsVOPD3) {
  unsigned Opc = FirstMI.getOpcode();
  unsigned Opc2 = SecondMI.getOpcode();
  AMDGPU::CanBeVOPD FirstCanBeVOPD =
      AMDGPU::getCanBeVOPD(Opc, EncodingFamily, IsVOPD3);
  AMDGPU::CanBeVOPD SecondCanBeVOPD =
      AMDGPU::getCanBeVOPD(Opc2, EncodingFamily, IsVOPD3);

  // If SecondMI depends on FirstMI they cannot execute at the same time.
  if (TII.hasRAWDependency(FirstMI, SecondMI))
    return std::nullopt;

  const GCNSubtarget &ST = TII.getSubtarget();
  bool AllowSameVGPR = ST.hasGFX1250Insts();

  if (FirstCanBeVOPD.X && SecondCanBeVOPD.Y) {
    if (checkVOPDRegConstraints(TII, FirstMI, SecondMI, IsVOPD3, AllowSameVGPR))
      return VOPDMatchInfo{&FirstMI, &SecondMI, IsVOPD3};
  }

  // AllowSameVGPR relaxes the VGPR bank overlap check for source operands.
  // Only enable it when there is no antidependency.
  bool IsAntiDep = TII.hasRAWDependency(SecondMI, FirstMI);
  AllowSameVGPR &= !IsAntiDep;

  if (FirstCanBeVOPD.Y && SecondCanBeVOPD.X) {
    if (IsAntiDep && !TII.isVOPDAntidependencyAllowed(SecondMI))
      return std::nullopt;
    if (checkVOPDRegConstraints(TII, SecondMI, FirstMI, IsVOPD3, AllowSameVGPR))
      return VOPDMatchInfo{&SecondMI, &FirstMI, IsVOPD3};
  }

  return std::nullopt;
}

std::optional<VOPDMatchInfo> llvm::tryMatchVOPDPair(const SIInstrInfo &TII,
                                                    MachineInstr &FirstMI,
                                                    MachineInstr &SecondMI) {
  const GCNSubtarget &ST = TII.getSubtarget();
  unsigned EncodingFamily = AMDGPU::getVOPDEncodingFamily(ST);
  if (auto Match = tryMatchVOPDPairVariant(TII, EncodingFamily, FirstMI,
                                           SecondMI, /*IsVOPD3=*/false))
    return Match;
  if (ST.hasVOPD3())
    return tryMatchVOPDPairVariant(TII, EncodingFamily, FirstMI, SecondMI,
                                   /*IsVOPD3=*/true);
  return std::nullopt;
}

/// Check if the instr pair, FirstMI and SecondMI, should be scheduled
/// together. Given SecondMI, when FirstMI is unspecified, then check if
/// SecondMI may be part of a fused pair at all.
static bool shouldScheduleVOPDAdjacent(const TargetInstrInfo &TII,
                                       const TargetSubtargetInfo &TSI,
                                       const MachineInstr *FirstMI,
                                       const MachineInstr &SecondMI) {
  const SIInstrInfo &STII = static_cast<const SIInstrInfo &>(TII);
  const GCNSubtarget &ST = STII.getSubtarget();

  // One instruction case: just check whether SecondMI is eligible at all.
  if (!FirstMI) {
    unsigned EncodingFamily = AMDGPU::getVOPDEncodingFamily(ST);
    unsigned Opc2 = SecondMI.getOpcode();
    auto checkCanBeVOPD = [&](bool VOPD3) {
      AMDGPU::CanBeVOPD CanBeVOPD =
          AMDGPU::getCanBeVOPD(Opc2, EncodingFamily, VOPD3);
      return CanBeVOPD.Y || CanBeVOPD.X;
    };
    return checkCanBeVOPD(false) || (ST.hasVOPD3() && checkCanBeVOPD(true));
  }

#ifdef EXPENSIVE_CHECKS
  assert([&]() -> bool {
    for (auto MII = MachineBasicBlock::const_iterator(FirstMI);
         MII != FirstMI->getParent()->instr_end(); ++MII) {
      if (&*MII == &SecondMI)
        return true;
    }
    return false;
  }() && "Expected FirstMI to precede SecondMI");
#endif

  return tryMatchVOPDPair(STII, *const_cast<MachineInstr *>(FirstMI),
                          const_cast<MachineInstr &>(SecondMI))
      .has_value();
}

namespace {
/// Adapts design from MacroFusion
/// Puts valid candidate instructions back-to-back so they can easily
/// be turned into VOPD instructions
/// Greedily pairs instruction candidates. O(n^2) algorithm.
struct VOPDPairingMutation : ScheduleDAGMutation {
  MacroFusionPredTy shouldScheduleAdjacent; // NOLINT: function pointer

  VOPDPairingMutation(
      MacroFusionPredTy shouldScheduleAdjacent) // NOLINT: function pointer
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
} // namespace

std::unique_ptr<ScheduleDAGMutation> llvm::createVOPDPairingMutation() {
  return std::make_unique<VOPDPairingMutation>(shouldScheduleVOPDAdjacent);
}
