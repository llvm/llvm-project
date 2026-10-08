//===- AMDGPUSim/MIRAdapter.cpp - MachineInstr Adapter --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implements MachineInstr classification and property queries through the
/// representation independent SimInstInfo interface, so the simulator core
/// does not depend on MachineInstr, SIInstrInfo, or SIRegisterInfo.
//
//===----------------------------------------------------------------------===//

#include "MIRAdapter.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "HWModel.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include <cmath>

namespace llvm {
namespace AMDGPUSim {

MachineInstrInfo::MachineInstrInfo(const SIInstrInfo &TII,
                                   const SIRegisterInfo &TRI)
    : TII(TII), TRI(TRI) {}

InstClass MachineInstrInfo::classifyInst(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();

  if (Opc == AMDGPU::S_DELAY_ALU)
    return InstClass::DELAY_ALU;

  if (Opc == AMDGPU::S_SET_VGPR_MSB)
    return InstClass::MSB_SET;

  if (Opc == AMDGPU::S_BARRIER_SIGNAL_M0 ||
      Opc == AMDGPU::S_BARRIER_SIGNAL_ISFIRST_M0 ||
      Opc == AMDGPU::S_BARRIER_SIGNAL_IMM ||
      Opc == AMDGPU::S_BARRIER_SIGNAL_ISFIRST_IMM)
    return InstClass::BARRIER_SIGNAL;
  if (Opc == AMDGPU::S_BARRIER_WAIT)
    return InstClass::BARRIER_WAIT;
  if (Opc == AMDGPU::S_BARRIER)
    return InstClass::BARRIER;

  if (TII.isWaitcnt(Opc) || Opc == AMDGPU::S_WAITCNT_DEPCTR ||
      Opc == AMDGPU::S_WAIT_XCNT || Opc == AMDGPU::S_WAIT_TENSORCNT)
    return InstClass::WAITCNT;

  if (MI.isBranch())
    return InstClass::BRANCH;

  if (TII.isXDLWMMA(MI))
    return InstClass::WMMA;

  if (Opc == AMDGPU::TENSOR_LOAD_TO_LDS_d4 ||
      Opc == AMDGPU::TENSOR_LOAD_TO_LDS_d2 ||
      Opc == AMDGPU::TENSOR_STORE_FROM_LDS_d4 ||
      Opc == AMDGPU::TENSOR_STORE_FROM_LDS_d2 ||
      Opc == AMDGPU::TENSOR_LOAD_TO_LDS_d4_gfx1250 ||
      Opc == AMDGPU::TENSOR_LOAD_TO_LDS_d2_gfx1250 ||
      AMDGPU::isTensorStore(Opc))
    return InstClass::TDM;

  if (SIInstrFlags::isDS(MI)) {
    if (MI.mayLoad())
      return InstClass::DS_READ;
    if (MI.mayStore())
      return InstClass::DS_WRITE;
    return InstClass::OTHER;
  }

  if (TII.isVMEM(MI)) {
    if (MI.mayLoad())
      return InstClass::VMEM_READ;
    if (MI.mayStore())
      return InstClass::VMEM_WRITE;
    return InstClass::OTHER;
  }

  if (TII.isSMRD(MI))
    return InstClass::SMEM;

  if (TII.isSALU(MI))
    return InstClass::SALU;

  if (SIInstrInfo::isTRANS(MI))
    return InstClass::TRANS;

  if (SIInstrFlags::isVALU(MI))
    return InstClass::VALU;

  return InstClass::OTHER;
}

unsigned MachineInstrInfo::getLatency(const MachineInstr &MI,
                                      InstClass IC) const {
  switch (IC) {
  case InstClass::DS_READ:
    return GFX1250::DSReadLatency;
  case InstClass::DS_WRITE:
    return GFX1250::DSWriteLatency;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    return GFX1250::VMEMLatency;
  case InstClass::SMEM:
    return GFX1250::SMEMLatency;
  case InstClass::BARRIER:
    return GFX1250::BarrierLatency;
  case InstClass::NOP:
  case InstClass::DELAY_ALU:
  case InstClass::WAITCNT:
  case InstClass::BRANCH:
  case InstClass::MSB_SET:
    return 1;
  default:
    break;
  }

  // Query scheduling model for latency.
  const TargetSchedModel &SchedModel = TII.getSchedModel();
  if (SchedModel.hasInstrSchedModel()) {
    unsigned Latency = SchedModel.computeInstrLatency(&MI);
    if (Latency > 0)
      return Latency;
  }

  // Use the class defaults.
  return getLatencyForClass(IC);
}

SimInst MachineInstrInfo::createSimInst(const MachineInstr &MI) const {
  InstClass IC = classifyInst(MI);
  return SimInst(const_cast<MachineInstr *>(&MI), IC, getLatency(MI, IC),
                 getUnitForClass(IC));
}

unsigned MachineInstrInfo::getRepeatRate(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  // VOPD issues as one pair regardless of repeat data on either component.
  if (AMDGPU::isVOPD(MI->getOpcode()))
    return 1;

  return std::max(TII.getBlockingCycles(*MI), TII.getRepeatRate(*MI));
}

bool MachineInstrInfo::isLOLVALU(const SimInst &SI) const {
  return SI.Class == InstClass::VALU && getRepeatRate(SI) > 1;
}

unsigned MachineInstrInfo::getResourceCycles(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  InstClass IC = SI.Class;

  // Use repeat rate for VALU and TRANS occupancy.
  if (IC == InstClass::VALU || IC == InstClass::TRANS) {
    unsigned RepeatRate = getRepeatRate(SI);
    if (RepeatRate > 0)
      return RepeatRate;
  }

  // Use fixed occupancy for DS and TDM.
  if (IC == InstClass::DS_READ || IC == InstClass::DS_WRITE)
    return 1;
  if (IC == InstClass::TDM)
    return GFX1250::TDMResourceCycles;

  // Query scheduling model for throughput.
  const TargetSchedModel &SchedModel = TII.getSchedModel();
  if (SchedModel.hasInstrSchedModel()) {
    double ReciprocalThroughput = SchedModel.computeReciprocalThroughput(MI);
    if (ReciprocalThroughput > 0.0) {
      unsigned Cycles =
          std::max(1u, static_cast<unsigned>(std::ceil(ReciprocalThroughput)));
      if (IC == InstClass::TRANS && Cycles < 2)
        return 2;
      return Cycles;
    }
  }

  // Use the class defaults.
  if (IC == InstClass::WMMA)
    return 8;
  if (IC == InstClass::TRANS)
    return 2;
  return 1;
}

unsigned MachineInstrInfo::getDelayAluImm(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  if (MI->getOpcode() == AMDGPU::S_DELAY_ALU && MI->getNumOperands() > 0 &&
      MI->getOperand(0).isImm())
    return MI->getOperand(0).getImm();
  return 0;
}

WaitRequirements MachineInstrInfo::getWaitInfo(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  unsigned WaitCount = 0;
  if (MI->getNumOperands() > 0 && MI->getOperand(0).isImm())
    WaitCount = MI->getOperand(0).getImm();

  WaitRequirements Waits;
  switch (MI->getOpcode()) {
  case AMDGPU::S_WAIT_DSCNT:
    Waits.emplace_back(WaitType::DS, WaitCount);
    break;
  case AMDGPU::S_WAIT_LOADCNT:
    Waits.emplace_back(WaitType::VMEMLoad, WaitCount);
    break;
  case AMDGPU::S_WAIT_LOADCNT_DSCNT: {
    AMDGPU::Waitcnt Decoded = AMDGPU::decodeLoadcntDscnt(
        AMDGPU::getIsaVersion(TII.getSubtarget().getCPU()), WaitCount);
    Waits.emplace_back(WaitType::VMEMLoad, Decoded.get(AMDGPU::LOAD_CNT));
    Waits.emplace_back(WaitType::DS, Decoded.get(AMDGPU::DS_CNT));
    break;
  }
  case AMDGPU::S_WAIT_STORECNT:
    Waits.emplace_back(WaitType::VMEMStore, WaitCount);
    break;
  case AMDGPU::S_WAIT_STORECNT_DSCNT: {
    AMDGPU::Waitcnt Decoded = AMDGPU::decodeStorecntDscnt(
        AMDGPU::getIsaVersion(TII.getSubtarget().getCPU()), WaitCount);
    Waits.emplace_back(WaitType::VMEMStore, Decoded.get(AMDGPU::STORE_CNT));
    Waits.emplace_back(WaitType::DS, Decoded.get(AMDGPU::DS_CNT));
    break;
  }
  case AMDGPU::S_WAIT_KMCNT:
    Waits.emplace_back(WaitType::SMEM, WaitCount);
    break;
  case AMDGPU::S_WAIT_TENSORCNT:
    Waits.emplace_back(WaitType::Tensor, WaitCount);
    break;
  case AMDGPU::S_WAIT_XCNT:
    Waits.emplace_back(WaitType::XCnt, WaitCount);
    break;
  case AMDGPU::S_WAITCNT_DEPCTR:
    Waits.emplace_back(WaitType::DepCtr, WaitCount);
    break;
  default:
    break;
  }
  return Waits;
}

unsigned MachineInstrInfo::getVaVdstTarget(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  if (MI->getOpcode() == AMDGPU::S_WAITCNT_DEPCTR && MI->getNumOperands() > 0 &&
      MI->getOperand(0).isImm())
    return AMDGPU::DepCtr::decodeFieldVaVdst(MI->getOperand(0).getImm());
  return 15;
}

WMMAProperties MachineInstrInfo::getWMMAProperties(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  return TII.getWMMAProperties(*MI);
}

bool MachineInstrInfo::hasSGPROperands(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  const MachineRegisterInfo &MRI = MI->getMF()->getRegInfo();

  for (const MachineOperand &MO : MI->explicit_uses()) {
    if (MO.isReg() && MO.getReg().isPhysical() &&
        TRI.isSGPRReg(MRI, MO.getReg()))
      return true;
  }
  return false;
}

unsigned MachineInstrInfo::getInstBytes(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  return TII.getInstSizeInBytes(*MI);
}

} // namespace AMDGPUSim
} // namespace llvm
