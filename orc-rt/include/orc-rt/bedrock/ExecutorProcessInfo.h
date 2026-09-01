//===---- ExecutorProcessInfo.h - Executor Process Info APIs ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// APIs to provide information about the host process in which the executor
// is running.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_EXECUTORPROCESSINFO_H
#define ORC_RT_BEDROCK_EXECUTORPROCESSINFO_H

#include "orc-rt/support/Error.h"

#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace orc_rt {

/// Provides information about the host process in which the ORC runtime
/// executor is running.
class ExecutorProcessInfo {
public:
  /// Create an ExecutorProcessInfo from the given values.
  ExecutorProcessInfo(std::string Triple, size_t PageSize,
                      std::string CPUFeatures) noexcept;

  /// Create an ExecutorProcessInfo, auto-detecting values.
  static Expected<ExecutorProcessInfo> Detect() noexcept;

  /// Returns a string that is usable in SubtargetFeatures for the host process.
  const std::string &targetCPUFeatures() const noexcept { return CPUFeatures; }

  /// Returns a target triple string for the host process.
  const std::string &targetTriple() const noexcept { return Triple; }

  /// Returns the host process's page size.
  size_t pageSize() const noexcept { return PageSize; }

  /// This will return a string that can be forwarded to SubtargetFeatures
  /// It will only return "+" turning off features, is left to caller.
  /// This calls syscalls so result is cached
  static std::string detectCPUFeatures() noexcept;
  /// This calls syscalls so result is cached
  static std::string detectTargetTriple() noexcept;

  static Expected<size_t> detectPageSize() noexcept;

private:
  friend struct ExecutorProcessInfoTestAccess;

  // Storage of string_views is static, so will last the lifetime of the runtime
  static std::vector<std::string_view> detectTargetCPUFeatures();

  /// Formats vector of feature names as an SubtargetFeatures valid string, e.g.
  /// "+avx,+avx2".
  static std::string
  formatCPUFeatures(const std::vector<std::string_view> &Features);

  static std::string
  makeTargetTriple(std::initializer_list<std::string_view> Components);

  std::string Triple;
  size_t PageSize;
  std::string CPUFeatures;
};

} // namespace orc_rt

#endif // ORC_RT_BEDROCK_EXECUTORPROCESSINFO_H
