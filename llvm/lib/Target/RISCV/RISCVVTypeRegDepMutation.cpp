//===- RISCVVTypeRegDepMutation.cpp - RISC-V Vector Mask DAGMutation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCVRegisterInfo.h"
#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include "llvm/TargetParser/RISCVTargetParser.h"

#define DEBUG_TYPE "machine-scheduler"

namespace llvm {

static void collectPossibleVSetsForMI(
    MachineInstr *MI, int RequiredVL, unsigned PossibleVType,
    SmallVector<std::tuple<MachineInstr *, int>> &VSetInstrVLs,
    SmallVector<MachineInstr *> &PossibleVSets) {
  unsigned VTypeMask = unsigned(-1);
  if (MI->getOpcode() == RISCV::PseudoVLE8_V_MF4 ||
      MI->getOpcode() == RISCV::PseudoVLE8_V_MF8) {
    // TODO: actually, we care about the ratio lmul / sew
    VTypeMask = ~((0x7 << 3) | 0x7);
  }
  unsigned RequiredMaskedVType = PossibleVType & VTypeMask;
  for (auto &VSetInstrVL : VSetInstrVLs) {
    MachineInstr *VSetInstr = std::get<0>(VSetInstrVL);
    if ((std::get<1>(VSetInstrVL) == RequiredVL) &&
        ((VSetInstr->getOperand(2).getImm() & VTypeMask) ==
         RequiredMaskedVType)) {
      PossibleVSets.push_back(VSetInstr);
    }
  }
  return;
}

class RISCVVTypeRegDepMutation : public ScheduleDAGMutation {
private:
  const TargetRegisterInfo *TRI;

public:
  RISCVVTypeRegDepMutation(const TargetRegisterInfo *TRI) : TRI(TRI) {}
  void apply(ScheduleDAGInstrs *DAG) override;
};

// For each instruction I that reads vtype register:
//  (1) compute reaching definitions of vtype register.
//  (2) From all vtype reg definitions in the basic block collect the ones
//  compatible with those found in (1) (call this set Comp(I)).
//  (3) for each instruction vset in Comp(I) if there is data dependence on
//  vtype reg vset->I, remove it, also, remove all anti-depencies I->vset (4)
//  Choose by some heuristic an instruction best_vset from Comp(I) and insert
//  data  dependence on vtype: best_vset->I. NOTE: code below is not correct,
//  but by luck it doesn't cause any issues on SPEC.
void RISCVVTypeRegDepMutation::apply(ScheduleDAGInstrs *DAG) {
  SmallVector<std::tuple<MachineInstr *, int>> VSetInstrVLs;
  SmallVector<std::tuple<MachineInstr *, int, MachineInstr *>> MIVLVSetInstrs;
  int CurrentVL = -1;
  for (auto MBBI = DAG->begin(); MBBI != DAG->end(); ++MBBI) {
    switch (MBBI->getOpcode()) {
    default:
      continue;
    case RISCV::PseudoVSETIVLI: {
      CurrentVL = MBBI->getOperand(1).getImm();
      VSetInstrVLs.push_back({&*MBBI, CurrentVL});
      continue;
    }
    case RISCV::PseudoVSETVLI: {
      continue;
    }
    case RISCV::PseudoVSETVLIX0X0: {
      // TODO: the case when we're setting to VLMax
      // if (MBBI->getOperand(0).getReg() != RISCV::X0) {
      //   continue;
      // }
      VSetInstrVLs.push_back(std::make_pair(&*MBBI, CurrentVL));
      continue;
    }
    case RISCV::PseudoVLE8_V_MF8:
    case RISCV::PseudoVLE8_V_MF4: {
      if (!VSetInstrVLs.empty())
        MIVLVSetInstrs.push_back(
            {&*MBBI, CurrentVL, std::get<0>(VSetInstrVLs.back())});
      continue;
    }
    }
  }

  DenseMap<MachineInstr *,
           std::tuple<SmallVector<MachineInstr *>, MachineInstr *>>
      MIToPossibleVSetsMap;
  for (auto &MIVLVSetInstr : MIVLVSetInstrs) {
    MachineInstr *OrigVSetInstr = std::get<2>(MIVLVSetInstr);
    int RequiredVL = std::get<1>(MIVLVSetInstr);
    int PossibleVType = OrigVSetInstr->getOperand(2).getImm();
    SmallVector<MachineInstr *> PossibleVSets;
    MachineInstr *MI = std::get<0>(MIVLVSetInstr);
    collectPossibleVSetsForMI(MI, RequiredVL, PossibleVType, VSetInstrVLs,
                              PossibleVSets);
    MIToPossibleVSetsMap[MI] = {PossibleVSets, OrigVSetInstr};
  }

  DenseMap<MachineInstr *, SUnit *> VSetToSUMap;
  for (SUnit &SU : DAG->SUnits) {
    if (!SU.isInstr())
      continue;
    MachineInstr *MI = SU.getInstr();
    unsigned Opc = MI->getOpcode();
    if ((Opc == RISCV::PseudoVSETIVLI) || (Opc == RISCV::PseudoVSETVLI) ||
        (Opc == RISCV::PseudoVSETVLIX0))
      VSetToSUMap[MI] = &SU;
  }

  for (SUnit &SU : DAG->SUnits) {
    if (!SU.isInstr())
      continue;
    MachineInstr *MI = SU.getInstr();
    SmallVector<MachineInstr *> &PossibleVSets =
        std::get<0>(MIToPossibleVSetsMap[MI]);
    if (PossibleVSets.size() < 2)
      continue;
    // Choose the earliest (in the original program order) VSET insruction
    // satisfying the vtype requirements of MI.
    SUnit *NewVSetSU = VSetToSUMap[PossibleVSets[0]];
    SUnit *OldVSetSU = VSetToSUMap[std::get<1>(MIToPossibleVSetsMap[MI])];
    for (auto &D : SU.Succs) {
      if (D.getKind() != SDep::Kind::Anti)
        continue;
      unsigned Reg = D.getReg();
      if (Reg != RISCV::VL && Reg != RISCV::VTYPE)
        continue;
      SUnit &AntiDepSucc = *D.getSUnit();
      // TODO: we can only remove anti-depence to compatible vsets.
      for (auto &P : AntiDepSucc.Preds) {
        if (P.getSUnit() == &SU) {
          AntiDepSucc.removePred(P);
        }
      }
    }
    for (auto &D : SU.Preds) {
      if (D.getSUnit() != OldVSetSU)
        continue;
      if (D.getKind() != SDep::Kind::Data)
        continue;
      unsigned Reg = D.getReg();
      if (Reg != RISCV::VL && Reg != RISCV::VTYPE)
        continue;
      unsigned Latency = D.getLatency();
      SU.removePred(D);
      SDep NewSDep(NewVSetSU, SDep::Kind::Data, Reg);
      NewSDep.setLatency(Latency);
      SU.addPred(NewSDep);
    }
  }
  return;
}

std::unique_ptr<ScheduleDAGMutation>
createRISCVVTypeRegDepMutation(const TargetRegisterInfo *TRI) {
  return std::make_unique<RISCVVTypeRegDepMutation>(TRI);
}
} // namespace llvm
