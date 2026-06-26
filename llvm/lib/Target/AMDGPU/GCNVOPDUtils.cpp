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

    // V_FMAMK_F32 (src1) and V_FMAAK_F32 (src2) have a mandatory literal.
    // VOPD3 instructions don't set MandatoryLiteralIdx.
    if (InstInfo[CompIdx].hasMandatoryLiteral()) {
      auto CompOprIdx = InstInfo[CompIdx].getMandatoryLiteralCompOperandIndex();
      addLiteral(MI.getOperand(CompOprIdx));
    }

    // VOPD only. Affects V_CNDMASK_B32_e32.
    if (MI.getDesc().hasImplicitUseOfPhysReg(AMDGPU::VCC))
      UniqueScalarRegs.insert(AMDGPU::VCC_LO);

    if (IsVOPD3) {
      if (const MachineOperand *Src1 =
              TII.getNamedOperand(MI, AMDGPU::OpName::src1)) {
        if (!Src1->isReg() || !TRI->isVGPR(MRI, Src1->getReg()))
          return false;
      }

      if (const MachineOperand *Src2 =
              TII.getNamedOperand(MI, AMDGPU::OpName::src2)) {
        if (AMDGPU::hasNamedOperand(MI.getOpcode(), AMDGPU::OpName::bitop3)) {
          // BITOP3 can be converted to DUAL_BITOP2 when src2 is zero.
          if (!Src2->isImm() || Src2->getImm())
            return false;
        } else if (MI.getOpcode() == AMDGPU::V_CNDMASK_B32_e64) {
          UniqueScalarRegs.insert(Src2->getReg());
        } else if (!Src2->isReg() || !TRI->isVGPR(MRI, Src2->getReg())) {
          return false;
        }
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

  auto getVRegIdx = [&](unsigned OpcodeIdx, unsigned OperandIdx) {
    const MachineInstr &MI = (OpcodeIdx == VOPD::X) ? MIX : MIY;
    const MachineOperand &Operand = MI.getOperand(OperandIdx);
    if (Operand.isReg() && TRI->isVectorRegister(MRI, Operand.getReg()))
      return Operand.getReg();
    return Register();
  };

  // On GFX1170+ if both OpX and OpY are V_MOV_B32 then OPY uses SRC2
  // source-cache.
  bool SkipSrc = (ST.hasGFX11_7Insts() || ST.hasGFX12Insts()) &&
                 MIX.getOpcode() == AMDGPU::V_MOV_B32_e32 &&
                 MIY.getOpcode() == AMDGPU::V_MOV_B32_e32;

  // Check VGPR bank constraints for operand registers across both instructions.
  if (InstInfo.hasInvalidOperand(getVRegIdx, *TRI, SkipSrc, AllowSameVGPR,
                                 IsVOPD3))
    return false;

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

  if (!(FirstCanBeVOPD.X && SecondCanBeVOPD.Y) &&
      !(FirstCanBeVOPD.Y && SecondCanBeVOPD.X))
    return std::nullopt;

  // If SecondMI depends on FirstMI they cannot execute at the same time.
  if (TII.hasRAWDependency(FirstMI, SecondMI))
    return std::nullopt;

  const GCNSubtarget &ST = TII.getSubtarget();
  bool AllowSameVGPR = ST.hasGFX12Insts();

  if (FirstCanBeVOPD.X && SecondCanBeVOPD.Y) {
    if (checkVOPDRegConstraints(TII, FirstMI, SecondMI, IsVOPD3, AllowSameVGPR))
      return VOPDMatchInfo{&FirstMI, &SecondMI, IsVOPD3};
  }

  if (FirstCanBeVOPD.Y && SecondCanBeVOPD.X) {
    // AllowSameVGPR relaxes the VGPR bank overlap check for source operands.
    // Only enable it when there is no antidependency.
    bool IsAntiDep = TII.hasRAWDependency(SecondMI, FirstMI);
    AllowSameVGPR &= !IsAntiDep;
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

/// Collect all load (dependents if \p Forward else dependencies) that connect
/// to the \p Head SU.
/// \p Visited should allocate enough bits for the number of SUnits, but its
/// value can otherwise be uninitialized.
static void collectLoads(SmallPtrSet<SUnit *, 8> &Loads, BitVector &Visited,
                         SUnit &Head, bool Forward, bool StopAtLoads) {
  if (Head.isBoundaryNode())
    return;

  Visited.reset();

  SmallVector<SUnit *> Stack;
  Stack.push_back(&Head);
  while (!Stack.empty()) {
    SUnit *SU = Stack.pop_back_val();
    const SmallVector<SDep, 4> &Deps = Forward ? SU->Succs : SU->Preds;
    for (const SDep &Edge : Deps) {
      if (StopAtLoads && Edge.getKind() != SDep::Data)
        continue;
      SUnit *Dep = Edge.getSUnit();
      if (Dep->isBoundaryNode() || Visited.test(Dep->NodeNum))
        continue;
      Visited.set(Dep->NodeNum);

      if (Dep->isInstr() && Dep->getInstr()->mayLoad()) {
        Loads.insert(Dep);
        if (StopAtLoads)
          continue;
      }
      Stack.push_back(Dep);
    }
  }
}

/// Checks whether fusing SU \p I with SU \p J would force the loads preceding
/// \p J to complete before loads depending on \p I.
///
/// \p ILoadSuccs should hold all first load successors of \p I (via
/// collectLoads with StopAtLoads=true). For set bits in \p LoadPredsComputed,
/// the corresponding set in \p LoadPredsCache should hold all transitive load
/// dependencies (via collectLoads with StopAtLoads=false). The \p Scratch
/// bitvector should allocate enough bits for the number of SUnits.
static bool loadsMayOverlap(
    [[maybe_unused]] SUnit &I, const SmallPtrSet<SUnit *, 8> &ILoadSuccs,
    SUnit &J, BitVector &LoadPredsComputed,
    SmallVector<SmallPtrSet<SUnit *, 8>> &LoadPredsCache, BitVector &Scratch) {

  if (ILoadSuccs.empty())
    return false;

  SmallPtrSet<SUnit *, 8> &JLoadPreds = LoadPredsCache[J.NodeNum];
  if (!LoadPredsComputed.test(J.NodeNum)) {
    collectLoads(JLoadPreds, Scratch, J, /*Forward=*/false,
                 /*StopAtLoads=*/true);
    LoadPredsComputed.set(J.NodeNum);
  }
  if (JLoadPreds.empty())
    return false;

  for (SUnit *ILoad : ILoadSuccs) {
    SmallPtrSet<SUnit *, 8> &ILoadDeps = LoadPredsCache[ILoad->NodeNum];
    if (!LoadPredsComputed.test(ILoad->NodeNum)) {
      collectLoads(ILoadDeps, Scratch, *ILoad, /*Forward=*/false,
                   /*StopAtLoads=*/false);
      LoadPredsComputed.set(ILoad->NodeNum);
    }

    for (SUnit *JLoad : JLoadPreds) {
      if (ILoad == JLoad) {
        LLVM_DEBUG(
            dbgs() << "Will not pair SU(" << I.NodeNum << ") with SU("
                   << J.NodeNum << ")\n"
                   << "  Fusion would introduce a cyclic dependency with SU("
                   << ILoad->NodeNum << ")\n");
        return true;
      }

      if (!ILoadDeps.contains(JLoad)) {
        LLVM_DEBUG(dbgs() << "Will not pair SU(" << I.NodeNum << ") with SU("
                          << J.NodeNum << ")\n"
                          << "  Fusion may force SU(" << JLoad->NodeNum
                          << ") to complete its load before dispatching SU("
                          << ILoad->NodeNum << ")\n");
        return true;
      }
    }
  }
  return false;
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

    BitVector VOPDCapable(DAG->SUnits.size());
    unsigned IIdx = 0;
    // Pre-compute whether each individual instruction can be VOPD
    for (auto ISUI = DAG->SUnits.begin(), E = DAG->SUnits.end(); ISUI != E;
         ++ISUI, ++IIdx) {
      const MachineInstr *IMI = ISUI->getInstr();
      if (shouldScheduleAdjacent(TII, ST, nullptr, *IMI) &&
          hasLessThanNumFused(*ISUI, 2))
        VOPDCapable[IIdx] = true;
    }

    IIdx = 0;
    SmallPtrSet<SUnit *, 8> ILoadSuccs;

    // Cache collected load predecessors.
    // For VOPDCapable nodes, this caches collectLoads with StopAtLoads=true
    // For loads, this caches collectLoads with StopAtLoads=false
    BitVector LoadPredsComputed(DAG->SUnits.size());
    SmallVector<SmallPtrSet<SUnit *, 8>> LoadPredsCache(DAG->SUnits.size());

    BitVector Scratch(DAG->SUnits.size());
    for (auto ISUI = DAG->SUnits.begin(), E = DAG->SUnits.end(); ISUI != E;
         ++ISUI, ++IIdx) {
      if (!VOPDCapable[IIdx])
        continue;
      const MachineInstr *IMI = ISUI->getInstr();

      ILoadSuccs.clear();
      collectLoads(ILoadSuccs, Scratch, *ISUI, /*Forward=*/true,
                   /*StopAtLoads=*/true);

      unsigned JIdx = IIdx + 1;
      for (auto JSUI = ISUI + 1; JSUI != E; ++JSUI, ++JIdx) {
        if (!VOPDCapable[JIdx] || JSUI->isBoundaryNode())
          continue;
        const MachineInstr *JMI = JSUI->getInstr();
        if (!hasLessThanNumFused(*JSUI, 2) ||
            !shouldScheduleAdjacent(TII, ST, IMI, *JMI))
          continue;

        if (loadsMayOverlap(*ISUI, ILoadSuccs, *JSUI, LoadPredsComputed,
                            LoadPredsCache, Scratch))
          continue;

        if (fuseInstructionPair(*DAG, *ISUI, *JSUI)) {
          // Clear to prevent future checks/fusing
          VOPDCapable[JIdx] = false;
          break;
        }
      }
    }
    LLVM_DEBUG(dbgs() << "Completed VOPDPairingMutation\n");
  }
};
} // namespace

std::unique_ptr<ScheduleDAGMutation> llvm::createVOPDPairingMutation() {
  return std::make_unique<VOPDPairingMutation>(shouldScheduleVOPDAdjacent);
}
