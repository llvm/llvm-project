//===------ FlattenAlgo.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main algorithm of the FlattenSchedulePass. This is a separate file to avoid
// the unittest for this requiring linking against LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_FLATTENALGO_H
#define POLLY_FLATTENALGO_H

#include "polly/Support/GICHelper.h"

namespace polly {
/// Recursively flatten a schedule.
///
/// Reduce the number of scatter dimensions as much as possible without changing
/// the relative order of instances in a schedule. Ideally, this results in a
/// single scatter dimension, but it may not always be possible to combine
/// dimensions, eg. if a dimension is unbounded. In worst case, the original
/// schedule is returned.
///
/// Schedules with fewer dimensions may be easier to understand for humans, but
/// it should make no difference to the computer.
///
/// @param Schedule The input schedule.
///
/// @return The flattened schedule.
isl::union_map flattenSchedule(isl::union_map Schedule);
} // namespace polly

#endif /* POLLY_FLATTENALGO_H */
