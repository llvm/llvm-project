//===- AMDGPUSim/SimInstInfo.h - Instruction query interface ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the instruction query interface between adapters and the simulator
/// core.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMINSTINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMINSTINFO_H

#include "SimInst.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
namespace AMDGPUSim {

using WaitRequirements = SmallVector<WaitRequirement, 2>;

/// Representation independent instruction queries used by the simulator.
///
/// Each SimInst passed to an implementation must originate from its matching
/// adapter.
class SimInstInfo {
public:
  virtual ~SimInstInfo() = default;

  /// Return execution unit occupancy in cycles for resource modeling and long
  /// latency VALU classification. Return zero when unavailable.
  virtual unsigned getRepeatRate(const SimInst &SI) const = 0;

  /// Return whether \p SI is a VALU with a repeat rate greater than one.
  virtual bool isLOLVALU(const SimInst &SI) const = 0;

  /// Return the number of cycles that \p SI occupies its functional unit.
  virtual unsigned getResourceCycles(const SimInst &SI) const = 0;

  /// Return the S_DELAY_ALU immediate, or zero when \p SI is not S_DELAY_ALU.
  virtual unsigned getDelayAluImm(const SimInst &SI) const = 0;

  /// Decode the counter thresholds imposed by \p SI.
  /// Return an empty list when the instruction does not wait or the adapter
  /// does not model its wait encoding.
  virtual WaitRequirements getWaitInfo(const SimInst &SI) const = 0;

  /// Return the VaVdst target in the range zero through 14.
  /// Return 15 when no VaVdst wait applies.
  virtual unsigned getVaVdstTarget(const SimInst &SI) const = 0;

  /// Return the properties that select the WMMA coexecution window.
  virtual WMMAProperties getWMMAProperties(const SimInst &SI) const = 0;

  /// Return whether \p SI has an explicit physical SGPR source.
  virtual bool hasSGPROperands(const SimInst &SI) const = 0;

  /// Return the encoded instruction size in bytes.
  virtual unsigned getInstBytes(const SimInst &SI) const = 0;
};

} // namespace AMDGPUSim
} // namespace llvm

#endif
