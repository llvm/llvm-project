//===------------------- ProcessRunner.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the ProcessRunner code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ADVISOR_PROCESS_RUNNER_H
#define LLVM_ADVISOR_PROCESS_RUNNER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <string>

namespace llvm {
namespace advisor {

class ProcessRunner {
public:
  struct ProcessResult {
    int exitCode;
    std::string stdout;
    std::string stderr;
    double executionTime;
  };

  static Expected<ProcessResult>
  run(llvm::StringRef program, const llvm::SmallVector<std::string, 8> &args,
      int timeoutSeconds = 60);

  static Expected<ProcessResult> runWithEnv(
      llvm::StringRef program, const llvm::SmallVector<std::string, 8> &args,
      const llvm::SmallVector<std::string, 8> &env, int timeoutSeconds = 60);
};

} // namespace advisor
} // namespace llvm

#endif
