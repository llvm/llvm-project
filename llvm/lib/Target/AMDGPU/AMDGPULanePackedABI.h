//===-- AMDGPULanePackedABI.h - Lane-packed inreg arg ABI -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Custom calling convention analysis that packs overflow inreg arguments
/// into VGPR lanes using writelane/readlane instead of consuming one VGPR
/// per overflow argument.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPULANEPACKEDABI_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPULANEPACKEDABI_H

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/TargetCallingConv.h"
#include <functional>

namespace llvm {

/// Custom CC analysis that packs overflow inreg arguments into VGPR lanes.
/// Inreg args are first assigned to SGPRs. When SGPRs are exhausted, overflow
/// inreg args share VGPR lanes (up to 32 for wave32, 64 for wave64 per VGPR).
/// Non-inreg args are assigned to individual VGPRs normally.
///
/// Lane-packed entries are marked with needsCustom() == true.
template <typename ArgT>
void analyzeArgsWithLanePacking(CCState &State,
                                const SmallVectorImpl<ArgT> &Args,
                                bool IsWave32);

/// Post-process CC results: repack overflow inreg VGPR assignments into
/// lane-packed entries. IsInReg returns true if the arg at the given index
/// has the inreg flag. This is used by GlobalISel which doesn't use
/// InputArg/OutputArg directly.
void packOverflowInRegToVGPRLanes(SmallVectorImpl<CCValAssign> &ArgLocs,
                                  const std::function<bool(unsigned)> &IsInReg,
                                  bool IsWave32);

/// Compute the lane index for a lane-packed CCValAssign entry.
/// The lane index is the count of prior custom entries with the same VGPR.
unsigned getLaneIndexForPackedArg(const SmallVectorImpl<CCValAssign> &Locs,
                                  unsigned Idx);

/// Returns true if the lane-packing feature flag is enabled.
bool isInRegVGPRLanePackingEnabled();

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPULANEPACKEDABI_H
