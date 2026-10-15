//===- AMDGPUPrefetchOrdering.h - Order prefetches before loads -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUPREFETCHORDERING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUPREFETCHORDERING_H

#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include <memory>

namespace llvm {

/// Orders every prefetch before the loads that follow it. Add it before the
/// IGroupLP mutation, so that sched_group_barrier pipelines keep the order.
std::unique_ptr<ScheduleDAGMutation> createAMDGPUPrefetchOrderingDAGMutation();

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUPREFETCHORDERING_H
