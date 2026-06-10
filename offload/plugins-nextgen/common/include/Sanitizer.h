//===-- Sanitizer.h - Host-side GPU sanitizer reporting ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_COMMON_SANITIZER_H
#define OFFLOAD_PLUGINS_NEXTGEN_COMMON_SANITIZER_H

#include "sanitizer/gpu_sanitizer.h"

#include "llvm/ADT/DenseSet.h"

#include <cstdint>
#include <mutex>
#include <utility>

namespace llvm::omp::target {
namespace plugin {
struct GenericDeviceTy;
class DeviceImageTy;
} // namespace plugin

/// Deduplication tables for GPU sanitizer diagnostics.
struct SanitizerTables {
  /// Returns true the first time this conflicting PC pair and kind are seen.
  bool isNewRace(uint64_t PC, uint64_t PeerPC, unsigned Kind);

private:
  std::mutex Mtx;
  DenseSet<std::pair<uint64_t, uint64_t>> Races;
};

/// Report a concurrency sanitizer hit on the given device with deduplication.
void reportGPUCSanRace(plugin::GenericDeviceTy &Device, SanitizerTables &Tables,
                       const __tsan_gpu_race &Race);

} // namespace llvm::omp::target

#endif // OFFLOAD_PLUGINS_NEXTGEN_COMMON_SANITIZER_H
