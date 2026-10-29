//===- AMDGPUAsyncStages.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Shared AMDGPU asyncmark stage definitions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AMDGPUASYNCSTAGES_H
#define LLVM_SUPPORT_AMDGPUASYNCSTAGES_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>

namespace llvm::AMDGPU {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// Bit mask of the async stages tracked by the asyncmark / wait_asyncmark
/// intrinsics. Each stage has its own independent sequence of marks. A single
/// stage is a mask with exactly one bit set.
///
/// The bit of each stage is part of the IR, so do not renumber them. Some
/// values are RESERVED for later use.
enum class AsyncStages : uint32_t {
  NONE = 0,
  // Tensor loads to LDS and tensor stores from LDS.
  TENSOR = 1u << 0,
  // Asynchronous global loads to LDS.
  GLOBAL_LOAD_ASYNC_TO_LDS = 1u << 1,
  // Asynchronous multicast (cluster) global loads to LDS.
  GLOBAL_LOAD_ASYNC_TO_LDS_MCAST = 1u << 2,
  // Asynchronous global stores from LDS.
  GLOBAL_STORE_ASYNC_FROM_LDS = 1u << 3,
  // Held for downstream use.
  RESERVED_4 = 1u << 4,
  // Buffer loads to LDS and pre-gfx1250 global loads to LDS.
  BUFFER_GLOBAL_LOAD = 1u << 5,
  RESERVED_6 = 1u << 6,
  RESERVED_7 = 1u << 7,
  RESERVED_8 = 1u << 8,
  RESERVED_9 = 1u << 9,
  RESERVED_10 = 1u << 10,

  RESERVED = RESERVED_4 | RESERVED_6 | RESERVED_7 | RESERVED_8 | RESERVED_9 |
             RESERVED_10,

  // Bits that a mask may legally set. Reserved stages are included: naming a
  // stage whose operations do not exist yet is harmless, and accepting the bit
  // keeps masks portable as stages are filled in.
  ALL = (1u << 11) - 1,

  LLVM_MARK_AS_BITMASK_ENUM(RESERVED_10)
};

/// The number of stages, which is also the number of bits in a stage mask.
inline constexpr unsigned NumAsyncStages = BitWidth<AsyncStages>;

/// The name of each stage, indexed by the position of its bit.
inline constexpr StringLiteral AsyncStageNames[NumAsyncStages] = {
    "TENSOR",
    "GLOBAL_LOAD_ASYNC_TO_LDS",
    "GLOBAL_LOAD_ASYNC_TO_LDS_MCAST",
    "GLOBAL_STORE_ASYNC_FROM_LDS",
    "RESERVED_4",
    "BUFFER_GLOBAL_LOAD",
    "RESERVED_6",
    "RESERVED_7",
    "RESERVED_8",
    "RESERVED_9",
    "RESERVED_10",
};

/// \returns true if \p Mask is a legal stage mask operand of the asyncmark /
/// wait_asyncmark intrinsics and pseudos.
constexpr bool isValidAsyncStageMask(int64_t Mask) {
  return Mask >= 0 && (Mask & ~int64_t(to_underlying(AsyncStages::ALL))) == 0;
}

/// \returns the stages named by the stage mask operand \p Mask of the
/// asyncmark / wait_asyncmark intrinsics and pseudos. The zero mask is the one
/// exception to a set bit naming a stage: it names every stage rather than
/// none.
constexpr AsyncStages asyncStagesFromMaskOperand(int64_t Mask) {
  assert(isValidAsyncStageMask(Mask) && "Invalid async stage mask");
  return Mask ? static_cast<AsyncStages>(Mask) : AsyncStages::ALL;
}

/// \returns the single stages named by \p Mask, from the least significant bit
/// up.
inline auto asyncStages(AsyncStages Mask) {
  return make_filter_range(
      map_range(seq(0u, NumAsyncStages),
                [](unsigned I) { return static_cast<AsyncStages>(1u << I); }),
      [Mask](AsyncStages Stage) { return any(Mask & Stage); });
}

/// \returns the position of the bit of the single stage \p Stage, for indexing
/// per-stage data.
inline unsigned getAsyncStageIndex(AsyncStages Stage) {
  assert(has_single_bit(to_underlying(Stage)) && "Expected a single stage");
  return countr_zero(to_underlying(Stage));
}

/// \returns the name of the single stage \p Stage.
inline StringRef getAsyncStageName(AsyncStages Stage) {
  return AsyncStageNames[getAsyncStageIndex(Stage)];
}

/// Print the stages named by \p Mask as a '|'-separated list, or "all" if it
/// names every stage.
inline raw_ostream &operator<<(raw_ostream &OS, AsyncStages Mask) {
  if (Mask == AsyncStages::ALL)
    return OS << "all";
  if (Mask == AsyncStages::NONE)
    return OS << "none";
  interleave(
      asyncStages(Mask), OS,
      [&OS](AsyncStages Stage) { OS << getAsyncStageName(Stage); }, "|");
  return OS;
}

} // namespace llvm::AMDGPU

#endif // LLVM_SUPPORT_AMDGPUASYNCSTAGES_H
