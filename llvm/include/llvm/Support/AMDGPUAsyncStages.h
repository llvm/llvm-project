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

#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <string>

namespace llvm {
namespace AMDGPU {
namespace AsyncStage {

// Async stages tracked by the asyncmark / wait_asyncmark intrinsics. Each stage
// has its own independent sequence of marks.
//
// The intrinsics do not name a stage directly. They take a bitmask in which a
// set bit means "do not participate": asyncmark omits the stages it names, and
// wait_asyncmark ignores them. A mask of 0 therefore covers every stage, which
// is the behavior of the original stage-less intrinsics.
//
// Do not renumber. Some values are RESERVED for later use.
enum Stage : uint32_t {
  // Tensor loads to LDS and tensor stores from LDS.
  TENSOR = 0,
  // Asynchronous global loads to LDS.
  GLOBAL_LOAD_ASYNC_TO_LDS = 1,
  // Asynchronous multicast (cluster) global loads to LDS.
  GLOBAL_LOAD_ASYNC_TO_LDS_MCAST = 2,
  // Asynchronous global stores from LDS.
  ASYNC_LDS_STORE = 3,
  RESERVED_4 = 4,
  // Buffer loads to LDS and pre-gfx1250 global loads to LDS.
  BUFFER_GLOBAL_LOAD = 5,
  RESERVED_6 = 6,
  RESERVED_7 = 7,
  RESERVED_8 = 8,
  RESERVED_9 = 9,
  RESERVED_10 = 10,
  STAGE_LAST = RESERVED_10,

  NUM_STAGES = STAGE_LAST + 1
};

// Bits that a mask may legally set. Reserved stages are included: omitting a
// stage whose operations do not exist yet is harmless, and accepting the bit
// keeps masks portable as stages are filled in.
constexpr uint32_t MaskAllStages = (uint32_t(1) << NUM_STAGES) - 1;

constexpr bool isValidMask(uint32_t Mask) {
  return (Mask & ~MaskAllStages) == 0;
}

// A stage participates in an operation unless the mask names it.
constexpr bool participates(uint32_t Mask, uint32_t S) {
  return !(Mask & (uint32_t(1) << S));
}

constexpr bool isReservedStage(uint32_t S) {
  switch (S) {
  case RESERVED_4:
  case RESERVED_6:
  case RESERVED_7:
  case RESERVED_8:
  case RESERVED_9:
  case RESERVED_10:
    return true;
  case TENSOR:
  case GLOBAL_LOAD_ASYNC_TO_LDS:
  case GLOBAL_LOAD_ASYNC_TO_LDS_MCAST:
  case ASYNC_LDS_STORE:
  case BUFFER_GLOBAL_LOAD:
    return false;
  }
  llvm_unreachable("Unhandled stage");
}

constexpr const char *getStageName(uint32_t S) {
  switch (S) {
  case TENSOR:
    return "TENSOR";
  case GLOBAL_LOAD_ASYNC_TO_LDS:
    return "GLOBAL_LOAD_ASYNC_TO_LDS";
  case GLOBAL_LOAD_ASYNC_TO_LDS_MCAST:
    return "GLOBAL_LOAD_ASYNC_TO_LDS_MCAST";
  case ASYNC_LDS_STORE:
    return "ASYNC_LDS_STORE";
  case RESERVED_4:
    return "RESERVED_4";
  case BUFFER_GLOBAL_LOAD:
    return "BUFFER_GLOBAL_LOAD";
  case RESERVED_6:
    return "RESERVED_6";
  case RESERVED_7:
    return "RESERVED_7";
  case RESERVED_8:
    return "RESERVED_8";
  case RESERVED_9:
    return "RESERVED_9";
  case RESERVED_10:
    return "RESERVED_10";
  }
  llvm_unreachable("Unhandled stage");
}

// Render the stages a mask leaves in as a '|'-separated list. Masks usually
// name many more stages than they leave out, so listing the stages that
// participate keeps the rendering short and says what actually happens.
inline std::string getCoveredStagesString(uint32_t Mask) {
  if (!(Mask & MaskAllStages))
    return "all";
  if ((Mask & MaskAllStages) == MaskAllStages)
    return "none";
  std::string Result;
  for (uint32_t S = 0; S != NUM_STAGES; ++S) {
    if (!participates(Mask, S))
      continue;
    if (!Result.empty())
      Result += '|';
    Result += getStageName(S);
  }
  return Result;
}

} // namespace AsyncStage
} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_SUPPORT_AMDGPUASYNCSTAGES_H
