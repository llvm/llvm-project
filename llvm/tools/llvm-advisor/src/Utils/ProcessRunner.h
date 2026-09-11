//===------------------- ProcessRunner.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Safe, allowlisted subprocess execution for external_fallback capabilities.
// The only file in the entire codebase that may spawn child processes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#include <string>

namespace llvm::advisor {

/// Result of an external process invocation.
struct ProcessResult {
  int ExitCode = -1;
  std::string Stdout;
  std::string Stderr;
  uint64_t WallTimeNs = 0;
};

/// Runs external tools with an explicit allowlist.
///
/// Programs and their arguments must be registered before execution.
/// Any argument not in the allowlist causes the run to fail.
class ProcessRunner {
public:
  /// Allow all arguments for a given program.
  void allow(StringRef Program);

  /// Allow only specific flags for a given program.
  /// Overrides any previous allow-all setting.
  void allow(StringRef Program, ArrayRef<StringRef> Flags);

  /// Execute a program with the given arguments.
  ///
  /// Fails if the program or any argument is not allowlisted.
  /// If TimeoutSeconds is zero, no timeout is applied.
  Expected<ProcessResult> run(StringRef Program,
                              ArrayRef<std::string> Arguments,
                              unsigned TimeoutSeconds = 0) const;

private:
  struct ToolPolicy {
    StringSet<> Flags;
    bool AllowAll = false;
  };

  bool isAllowedArg(StringRef Program, StringRef Arg) const;
  StringMap<ToolPolicy> AllowList;
};

} // namespace llvm::advisor
