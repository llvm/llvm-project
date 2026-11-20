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
#define AMDGPUSIM_INST_CLASS(Name, Unit, Flavor)                               \
  case InstClass::Name:                                                        \
    return AMDGPU::InstructionFlavor::Flavor;
#include "SimEnums.def"
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
