//===- AMDGPUSim/InstrInfo.h - Per-instruction result -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines instruction timing results and stall reporting metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_INSTRINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_INSTRINFO_H

#include "SimInst.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace AMDGPUSim {

/// Dominant cause of an instruction issue stall.
enum class StallReason : uint8_t {
#define AMDGPUSIM_STALL_REASON(Name, Label) Name,
#include "SimEnums.def"
};

/// Return a stable label for \p Reason, or null when Reason is NONE.
inline const char *getStallReasonString(StallReason Reason) {
  switch (Reason) {
#define AMDGPUSIM_STALL_REASON(Name, Label)                                    \
  case StallReason::Name:                                                      \
    return Label;
#include "SimEnums.def"
  }
  llvm_unreachable("unknown stall reason");
}

/// Stall sources measured in cycles while determining instruction issue.
///
/// Sources can overlap, so total() returns the effective issue delay instead
/// of summing the individual fields.
struct StallBreakdown {
  /// Effective pre-issue delay after overlapping sources are combined.
  unsigned EffectiveStall = 0;
  /// Structural unit or VALU resource availability delay.
  unsigned FU = 0;
  /// Delay until a legal scaled WMMA LD_SCALE slot.
  unsigned VALUSlot = 0;
  /// Delay until the active coexecution window admits the instruction.
  unsigned CoExec = 0;
  /// Copy of CoExec used for verbose WMMACoExecMiss reporting.
  unsigned CoExecFromEffective = 0;
  /// Delay from the current or deferred S_DELAY_ALU dependency.
  unsigned DelayAlu = 0;
  /// Delay until modeled wait counters other than VaVdst reach their
  /// thresholds.
  unsigned WaitCnt = 0;
  /// Delay until the DS, VMEM, or TDM issue queue has capacity.
  unsigned MemFIFO = 0;
  /// Delay that moves a long latency VALU past the active WMMA window.
  unsigned LongLatVALU = 0;
  /// Delay needed to keep TRANS and long latency VALU issues two cycles apart.
  unsigned LOLVALUTRANSHazard = 0;
  /// Delay before SALU issue while a prior VALU or WMMA SGPR read is active.
  unsigned SSRC = 0;
  /// Delay until the modeled VaVdst count reaches the requested DEPCTR target.
  unsigned VaVdst = 0;

  unsigned total() const { return EffectiveStall; }
};

/// Result and reporting metadata for one simulated instruction.
struct InstrSimInfo {
  unsigned StallCycles = 0;
  /// Dominant categorized source. This can remain NONE when StallCycles
  /// contains only a delay that has no reporting category.
  StallReason Reason = StallReason::NONE;
  StallBreakdown Breakdown;

  /// Whether this instruction was classified as WMMA.
  bool IsWMMA = false;
  /// Modeled coexecution window pattern, or empty when no window is modeled.
  StringRef WMMAPattern;

  // WMMA window snapshot captured during stall evaluation.
  bool InWMMAWindow = false;
  uint8_t WMMAStage = 0;
  uint8_t WMMATotalWindow = 0;
  AMDGPU::CoExecStageType StageType = AMDGPU::CoExecStageType::NONE;
  /// Whether stall evaluation in an active WMMA window found neither a
  /// coexecution delay nor a long latency VALU delay.
  bool CoExecuted = false;

  /// Set for S_DELAY_ALU and for S_SET_VGPR_MSB fused with its predecessor.
  bool WasFused = false;
  /// Whether S_SET_VGPR_MSB could not fuse with the preceding instruction.
  bool WasExposed = false;
  /// Whether an exposed S_SET_VGPR_MSB consumes no cycle because the next
  /// instruction already has a coexecution delay.
  bool WasMasked = false;

  /// Return the dominant stall reason name, or null when Reason is NONE.
  const char *getReasonString() const { return getStallReasonString(Reason); }
};

} // namespace AMDGPUSim
} // namespace llvm

#endif
