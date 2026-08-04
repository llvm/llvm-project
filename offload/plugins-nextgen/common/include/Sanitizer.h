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

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <mutex>

namespace llvm::omp::target {
namespace plugin {
struct GenericDeviceTy;
} // namespace plugin

/// Deduplication tables for GPU sanitizer diagnostics.
struct SanitizerTables {
  /// Records a minimal-runtime (UBSan) diagnostic, returning true the first
  /// time this program counter and kind are seen.
  bool isNewReport(uint64_t PC, StringRef Kind);

private:
  std::mutex Mtx;
  StringSet<> Reports;
};

/// Symbolize and print one GPU UndefinedBehaviorSanitizer diagnostic against
/// \p Device's loaded images, in a format reminiscent of the host sanitizers.
/// \p Tables deduplicates repeated diagnostics.
void reportGPUUBSan(plugin::GenericDeviceTy &Device, SanitizerTables &Tables,
                    const __ubsan_gpu_report &Report);

} // namespace llvm::omp::target

#endif // OFFLOAD_PLUGINS_NEXTGEN_COMMON_SANITIZER_H
