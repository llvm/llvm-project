//===- MemProfCommon.h - MemProf support ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common types used by different parts of the MemProf code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_MEMPROFCOMMON_H
#define LLVM_PROFILEDATA_MEMPROFCOMMON_H

#include <cstdint>

namespace llvm {

// For optional hinted size reporting, holds a pair of the full stack id
// (pre-trimming, from the full context in the profile), and the associated
// total profiled size.
struct ContextTotalSize {
  uint64_t FullStackId;
  uint64_t TotalSize;
};

// Allocation type assigned to an allocation reached by a given context.
// More can be added, now this is cold, notcold and hot.
// Values should be powers of two so that they can be ORed, in particular to
// track allocations that have different behavior with different calling
// contexts.
enum class AllocationType : uint8_t {
  None = 0,
  NotCold = 1,
  Cold = 2,
  Hot = 4,
  All = 7 // This should always be set to the OR of all values.
};

} // namespace llvm

#endif // LLVM_PROFILEDATA_MEMPROFCOMMON_H
