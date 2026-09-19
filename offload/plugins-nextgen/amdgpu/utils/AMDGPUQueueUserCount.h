//===- AMDGPUQueueUserCount.h - HSA queue stream user count -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_UTILS_AMDGPUQUEUEUSERCOUNT_H
#define OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_UTILS_AMDGPUQUEUEUSERCOUNT_H

#include <cstdint>

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Tracks how many streams currently use an HSA queue.
struct AMDGPUQueueUserCount {
  uint32_t getUserCount() const { return NumUsers; }
  void addUser() { ++NumUsers; }
  void removeUser() { --NumUsers; }

private:
  uint32_t NumUsers = 0;
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_UTILS_AMDGPUQUEUEUSERCOUNT_H
