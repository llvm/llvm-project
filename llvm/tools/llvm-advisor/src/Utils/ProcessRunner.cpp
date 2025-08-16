//===-------------------- ProcessRunner.cpp - LLVM Advisor ----------------===//
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

#include "ProcessRunner.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <vector>

namespace llvm {
namespace advisor {

Expected<ProcessRunner::ProcessResult>
ProcessRunner::run(llvm::StringRef program,
                   const llvm::SmallVector<std::string, 8> &args,
                   int timeoutSeconds) {

  auto programPath = sys::findProgramByName(program);
  if (!programPath) {
    return createStringError(programPath.getError(),
                             "Tool not found: " + program);
  }

  llvm::SmallVector<StringRef, 8> execArgs;
  execArgs.push_back(program);
  for (const auto &arg : args) {
    execArgs.push_back(arg);
  }

  SmallString<128> stdoutPath, stderrPath;
  sys::fs::createTemporaryFile("stdout", "tmp", stdoutPath);
  sys::fs::createTemporaryFile("stderr", "tmp", stderrPath);

  std::optional<StringRef> redirects[] = {
      std::nullopt,          // stdin
      StringRef(stdoutPath), // stdout
      StringRef(stderrPath)  // stderr
  };

  int exitCode = sys::ExecuteAndWait(*programPath, execArgs, std::nullopt,
                                     redirects, timeoutSeconds);

  ProcessResult result;
  result.exitCode = exitCode;
  // TODO: Collect information about compilation time
  result.executionTime = 0; // not tracking time

  auto stdoutBuffer = MemoryBuffer::getFile(stdoutPath);
  if (stdoutBuffer) {
    result.stdout = (*stdoutBuffer)->getBuffer().str();
  }

  auto stderrBuffer = MemoryBuffer::getFile(stderrPath);
  if (stderrBuffer) {
    result.stderr = (*stderrBuffer)->getBuffer().str();
  }

  sys::fs::remove(stdoutPath);
  sys::fs::remove(stderrPath);

  return result;
}

Expected<ProcessRunner::ProcessResult> ProcessRunner::runWithEnv(
    llvm::StringRef program, const llvm::SmallVector<std::string, 8> &args,
    const llvm::SmallVector<std::string, 8> &env, int timeoutSeconds) {

  // For simplicity, just use the regular run method
  // Environment variables can be added later if needed
  return run(program, args, timeoutSeconds);
}

} // namespace advisor
} // namespace llvm
