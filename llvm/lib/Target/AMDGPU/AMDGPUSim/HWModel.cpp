//===- AMDGPUSim/HWModel.cpp - Hardware model implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implements hardware timing and resource parameters for AMDGPU subtargets.
//
//===----------------------------------------------------------------------===//

#include "HWModel.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace AMDGPUSim {

// TODO: Refactor these latency defaults into target independent queries.
unsigned getLatencyForClass(InstClass IC) {
  switch (IC) {
  case InstClass::VALU:
    return GFX1250::VALULatency;
  case InstClass::SALU:
    return GFX1250::SALULatency;
  case InstClass::TRANS:
  case InstClass::WMMA:
    return GFX1250::TRANSLatency;
  case InstClass::DS_READ:
    return GFX1250::DSReadLatency;
  case InstClass::DS_WRITE:
    return GFX1250::DSWriteLatency;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    return GFX1250::VMEMLatency;
  case InstClass::SMEM:
    return GFX1250::SMEMLatency;
  case InstClass::TDM:
    return GFX1250::TDMLatency;
  case InstClass::BARRIER:
  case InstClass::BARRIER_SIGNAL:
  case InstClass::BARRIER_WAIT:
    return GFX1250::BarrierLatency;
  default:
    return 1;
  }
}

// Translate the simulator classification to the shared classification used by
// coexecution windows.
static AMDGPU::InstructionFlavor getInstructionFlavor(InstClass IC) {
  switch (IC) {
  case InstClass::VALU:
    return AMDGPU::InstructionFlavor::SingleCycleVALU;
  case InstClass::TRANS:
    return AMDGPU::InstructionFlavor::TRANS;
  case InstClass::WMMA:
    return AMDGPU::InstructionFlavor::WMMA;
  case InstClass::SALU:
  case InstClass::BRANCH:
  case InstClass::NOP:
    return AMDGPU::InstructionFlavor::SALU;
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
    return AMDGPU::InstructionFlavor::DS;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    return AMDGPU::InstructionFlavor::VMEM;
  case InstClass::SMEM:
    return AMDGPU::InstructionFlavor::SMEM;
  case InstClass::TDM:
    return AMDGPU::InstructionFlavor::DMA;
  case InstClass::BARRIER:
  case InstClass::BARRIER_SIGNAL:
  case InstClass::BARRIER_WAIT:
  case InstClass::WAITCNT:
    return AMDGPU::InstructionFlavor::Fence;
  case InstClass::DELAY_ALU:
  case InstClass::MSB_SET:
  case InstClass::OTHER:
    return AMDGPU::InstructionFlavor::Other;
  }
  llvm_unreachable("unknown instruction class");
}

AMDGPU::CoExecMaskT getCoExecMask(InstClass IC) {
  return AMDGPU::getCoExecMask(getInstructionFlavor(IC));
}

GFX1250HWModel::GFX1250HWModel() {
  MaxDSInFlight = GFX1250::MaxDSInFlight;
  MaxVMEMInFlight = GFX1250::MaxVMEMInFlight;
  MaxTDMInFlight = GFX1250::MaxTDMInFlight;
  VaVdstMultiplier = 4;
}

} // namespace AMDGPUSim
} // namespace llvm
