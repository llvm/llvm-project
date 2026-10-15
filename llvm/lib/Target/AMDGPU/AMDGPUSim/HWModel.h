//===- AMDGPUSim/HWModel.h - Hardware model parameters ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines hardware timing and resource parameters for AMDGPU subtargets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_HWMODEL_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_HWMODEL_H

#include "AMDGPUCoExecInfo.h"
#include "SimInst.h"

namespace llvm {
namespace AMDGPUSim {

/// GPU targets with hardware models supported by AMDGPUSim.
enum class GPUTarget : uint8_t { GFX1250 = 0 };

/// GFX1250 latency, resource occupancy, and queue capacity parameters consumed
/// by the instruction adapter and simulator.
namespace GFX1250 {
constexpr unsigned VALULatency = 5;
constexpr unsigned SALULatency = 2;
constexpr unsigned TRANSLatency = 8;
constexpr unsigned DSReadLatency = 50;
constexpr unsigned DSWriteLatency = 8;
constexpr unsigned VMEMLatency = 300;
constexpr unsigned SMEMLatency = 20;
constexpr unsigned TDMLatency = 320;
constexpr unsigned TDMResourceCycles = 2;
constexpr unsigned BarrierLatency = 32;
constexpr unsigned XACKLatency = 36;

constexpr unsigned MaxDSInFlight = 10;
constexpr unsigned MaxVMEMInFlight = 16;
constexpr unsigned MaxTDMInFlight = 4;
} // namespace GFX1250

/// Return the default latency in cycles for \p IC.
/// Unmodeled classes return one cycle.
unsigned getLatencyForClass(InstClass IC);

/// Map \p IC to the mask used by shared coexecution windows.
AMDGPU::CoExecMaskT getCoExecMask(InstClass IC);

/// Target resource limits consumed by the static simulator.
class HWModel {
public:
  virtual ~HWModel() = default;
  virtual GPUTarget getTarget() const = 0;

  /// Maximum operations accepted before the corresponding FIFO stalls.
  /// The default value leaves the capacity unbounded.
  unsigned MaxDSInFlight = ~0u;
  unsigned MaxVMEMInFlight = ~0u;
  unsigned MaxTDMInFlight = ~0u;

  /// Scale factor from VALU latency to VaVdst write readiness.
  /// Concrete models must replace the unset default.
  unsigned VaVdstMultiplier = ~0u;

protected:
  HWModel() = default;
};

/// GFX1250 resource limits for single wave simulation.
class GFX1250HWModel final : public HWModel {
public:
  GFX1250HWModel();
  GPUTarget getTarget() const override { return GPUTarget::GFX1250; }
};

} // namespace AMDGPUSim
} // namespace llvm

#endif
