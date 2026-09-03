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

#include <cstdint>

namespace llvm {
namespace AMDGPU {
namespace AsyncStage {

// Stage selector for the asyncmark / wait_asyncmark intrinsics.
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
  /// Non-format buffer loads to LDS and legacy global loads to LDS.
  UNFORMATTED_BUFFER_GLOBAL_LOAD = 5,
  RESERVED_6 = 6,
  RESERVED_7 = 7,
  RESERVED_8 = 8,
  RESERVED_9 = 9,
  RESERVED_10 = 10,
  STAGE_LAST = RESERVED_10,

  ALL = 16,
};

/// Number of distinct non-ALL stages. ALL is not part of this range.
constexpr unsigned NumStages = STAGE_LAST + 1;

/// Number of slots needed to index every stage, including ALL. The
/// slots between STAGE_LAST and ALL are unused.
constexpr unsigned NumStageSlots = ALL + 1;

constexpr bool isValidStage(uint32_t S) { return S <= STAGE_LAST || S == ALL; }

constexpr bool isReservedStage(uint32_t S) {
  switch (S) {
  case RESERVED_4:
  case RESERVED_6:
  case RESERVED_7:
  case RESERVED_8:
  case RESERVED_9:
  case RESERVED_10:
    return true;
  default:
    return false;
  }
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
  case UNFORMATTED_BUFFER_GLOBAL_LOAD:
    return "UNFORMATTED_BUFFER_GLOBAL_LOAD";
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
  case ALL:
    return "ALL";
  default:
    return "invalid";
  }
}

} // namespace AsyncStage
} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_SUPPORT_AMDGPUASYNCSTAGES_H
