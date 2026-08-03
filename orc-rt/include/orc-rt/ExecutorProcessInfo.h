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

#ifndef ORC_RT_EXECUTORPROCESSINFO_H
#define ORC_RT_EXECUTORPROCESSINFO_H

#include "orc-rt/Error.h"
#include <string>

namespace orc_rt {

/// Provides information about the host process in which the ORC runtime
/// executor is running.
class ExecutorProcessInfo {
public:
  /// Create an ExecutorProcessInfo from the given values.
  ExecutorProcessInfo(std::string Triple, size_t PageSize) noexcept;

  /// Create an ExecutorProcessInfo, auto-detecting values.
  static Expected<ExecutorProcessInfo> Detect() noexcept;

  /// Returns a target triple string for the host process.
  const std::string &targetTriple() const noexcept { return Triple; }

  /// Returns the host process's page size.
  size_t pageSize() const noexcept { return PageSize; }

  static std::string detectTargetTriple() noexcept;
  static Expected<size_t> detectPageSize() noexcept;

private:
  std::string Triple;
  size_t PageSize;
};

} // namespace orc_rt

#endif // ORC_RT_EXECUTORPROCESSINFO_H
