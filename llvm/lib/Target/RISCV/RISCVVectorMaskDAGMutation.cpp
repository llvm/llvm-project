//===- RISCVVectorMaskDAGMutation.cpp - RISC-V Vector Mask DAGMutation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Schedule mutations for RISC-V vector masks.
//
// RISCVV0AliasDAGMutation preserves the order between physical V0 accesses and
// VMV0 virtual-register accesses. Every VMV0 virtual register is eventually
// allocated to V0, but the scheduler cannot infer that alias before register
// allocation.
//
// RISCVVectorMaskDAGMutation adds artificial dependencies between mask producer
// instructions and masked instructions to reduce overlapping mask live ranges.
//
// If there are multiple masks producers followed by multiple masked
// instructions, then at each masked instructions add dependency edges between
// every producer and masked instruction.
//
// The reason why we need to do this:
// 1. When tracking register pressure, we don't track physical registers.
// 2. We have a RegisterClass for mask register (which is `VMV0`), but we don't
//    use it by the time we reach scheduling. Instead, we use physical
//    register V0 directly and insert a `$v0 = COPY ...` before the use.
// 3. For mask producers, we are using VR RegisterClass (we can allocate V0-V31
//    to it). So if V0 is not available, there are still 31 available registers
//    out there.
//
// This means that the RegPressureTracker can't track the pressure of mask
// registers correctly.
//
// This schedule mutation is a workaround to fix this issue.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include "llvm/TargetParser/RISCVTargetParser.h"

#define DEBUG_TYPE "machine-scheduler"

namespace llvm {
namespace {

static bool aliasV0Accesses(unsigned First, unsigned Second) {
  // The register allocator resolves interference between virtual VMV0 values,
  // while the scheduler already models physical V0 dependencies. Add only
  // the otherwise-invisible physical/virtual aliases, and leave two reads
  // independently schedulable.
  return ((First & PhysV0Def) && (Second & (VirtualV0Use | VirtualV0Def))) ||
         ((First & PhysV0Use) && (Second & VirtualV0Def)) ||
         ((First & VirtualV0Def) && (Second & (PhysV0Use | PhysV0Def))) ||
         ((First & VirtualV0Use) && (Second & PhysV0Def));
}

class RISCVV0AliasDAGMutation : public ScheduleDAGMutation {
  const TargetRegisterInfo *TRI;

public:
  RISCVV0AliasDAGMutation(const TargetRegisterInfo *TRI) : TRI(TRI) {}

  void apply(ScheduleDAGInstrs *DAG) override {
    SmallVector<std::pair<SUnit *, unsigned>, 8> V0Accesses;

    for (SUnit &SU : DAG->SUnits) {
      const MachineInstr &MI = *SU.getInstr();
      unsigned Kind =
          RISCVRegisterInfo::getV0AccessKind(MI, DAG->MRI, DAG->TII, *TRI);
      if (Kind == NoV0Access)
        continue;

      // Keep all earlier accesses: skipping a read/read edge must not make a
      // later write forget either read. V0 access counts are normally small.
      for (auto [Pred, PredKind] : V0Accesses) {
        if (aliasV0Accesses(PredKind, Kind)) {
          bool Added = DAG->addEdge(&SU, SDep(Pred, SDep::Artificial));
          assert(Added && "unexpected cycle while ordering V0 accesses");
          (void)Added;
        }
      }
      V0Accesses.emplace_back(&SU, Kind);
    }
  }
};

static bool isCopyToV0(const MachineInstr &MI, const MachineRegisterInfo &MRI,
                       const TargetRegisterInfo *TRI) {
  if (!MI.isFullCopy() || !MI.getOperand(1).readsReg() ||
      !MI.getOperand(1).getReg().isVirtual())
    return false;
  Register Dst = MI.getOperand(0).getReg();
  if (Dst.isPhysical())
    return TRI->regsOverlap(Dst, RISCV::V0);
  return RISCVRegisterInfo::isV0OnlyRegClass(
      MRI.getRegClass(Dst), MI.getOperand(0).getSubReg(), *TRI);
}

static bool isMaskProducerForV0(const MachineInstr &MI,
                                const MachineRegisterInfo &MRI,
                                const TargetRegisterInfo *TRI) {
  if (MI.getNumExplicitDefs() != 1)
    return false;
  const MachineOperand &Def = MI.getOperand(0);
  return Def.isReg() && Def.getReg().isVirtual() &&
         RISCVRegisterInfo::isV0OnlyRegClass(MRI.getRegClass(Def.getReg()),
                                             Def.getSubReg(), *TRI) &&
         MRI.hasOneNonDBGUse(Def.getReg());
}

static bool isSoleUseCopyToV0(SUnit &SU, const MachineRegisterInfo &MRI,
                              const TargetRegisterInfo *TRI) {
  if (SU.Succs.size() != 1)
    return false;
  SDep &Dep = SU.Succs[0];
  // Ignore dependencies other than data or strong ordering.
  if (Dep.isWeak())
    return false;

  SUnit &DepSU = *Dep.getSUnit();
  if (DepSU.isBoundaryNode())
    return false;
  return isCopyToV0(*DepSU.getInstr(), MRI, TRI);
}

static bool hasV0ConstrainedUse(const MachineInstr &MI,
                                const MachineRegisterInfo &MRI,
                                const TargetInstrInfo *TII,
                                const TargetRegisterInfo *TRI) {
  for (const MachineOperand &MO : MI.uses()) {
    if (!MO.isReg() || !MO.readsReg() || !MO.getReg())
      continue;
    Register Reg = MO.getReg();
    if (Reg.isPhysical()) {
      if (TRI->regsOverlap(Reg, RISCV::V0))
        return true;
      continue;
    }
    if (RISCVRegisterInfo::isV0OnlyRegClass(MRI.getRegClass(Reg),
                                            MO.getSubReg(), *TRI))
      return true;
    const TargetRegisterClass *Constraint =
        MI.getRegClassConstraint(MO.getOperandNo(), TII, TRI);
    if (RISCVRegisterInfo::isV0OnlyRegClass(Constraint, /*SubReg=*/0, *TRI))
      return true;
  }
  return false;
}

class RISCVVectorMaskDAGMutation : public ScheduleDAGMutation {
private:
  const TargetRegisterInfo *TRI;

public:
  RISCVVectorMaskDAGMutation(const TargetRegisterInfo *TRI) : TRI(TRI) {}

  void apply(ScheduleDAGInstrs *DAG) override {
    SUnit *NearestUseV0SU = nullptr;
    SmallVector<SUnit *, 2> DefMask;
    for (SUnit &SU : DAG->SUnits) {
      const MachineInstr *MI = SU.getInstr();
      bool UseV0 = hasV0ConstrainedUse(*MI, DAG->MRI, DAG->TII, TRI);
      bool DefV0 = isMaskProducerForV0(*MI, DAG->MRI, TRI) ||
                   isSoleUseCopyToV0(SU, DAG->MRI, TRI);
      if (DefV0 && !UseV0)
        DefMask.push_back(&SU);

      if (UseV0) {
        NearestUseV0SU = &SU;

        // Copy may not be a real use, so skip it here.
        if (DefMask.size() > 1 && !MI->isCopy()) {
          for (SUnit *Def : DefMask)
            if (DAG->canAddEdge(Def, &SU))
              DAG->addEdge(Def, SDep(&SU, SDep::Artificial));
        }

        if (!DefMask.empty())
          DefMask.erase(DefMask.begin());
      }

      if (NearestUseV0SU && NearestUseV0SU != &SU && DefV0 &&
          // For LMUL=8 cases, there will be more possibilities to spill.
          // FIXME: We should use RegPressureTracker to do fine-grained
          // controls.
          RISCVII::getLMul(MI->getDesc().TSFlags) != RISCVVType::LMUL_8)
        DAG->addEdge(&SU, SDep(NearestUseV0SU, SDep::Artificial));
    }
  }
};

} // namespace

std::unique_ptr<ScheduleDAGMutation>
createRISCVV0AliasDAGMutation(const TargetRegisterInfo *TRI) {
  return std::make_unique<RISCVV0AliasDAGMutation>(TRI);
}

std::unique_ptr<ScheduleDAGMutation>
createRISCVVectorMaskDAGMutation(const TargetRegisterInfo *TRI) {
  return std::make_unique<RISCVVectorMaskDAGMutation>(TRI);
}

} // namespace llvm
