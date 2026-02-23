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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <optional>
#include <vector>

namespace llvm {
namespace advisor {

/// Run \p program with \p args, optionally overriding the child environment
/// with \p envOverride (pass std::nullopt to inherit the parent environment).
/// stdout and stderr are captured into the returned ProcessResult.
static llvm::Expected<ProcessRunner::ProcessResult>
runImpl(llvm::StringRef program, const llvm::SmallVector<std::string, 8> &args,
        std::optional<llvm::ArrayRef<llvm::StringRef>> envOverride,
        int timeoutSeconds) {
  auto programPath = llvm::sys::findProgramByName(program);
  if (!programPath)
    return llvm::createStringError(programPath.getError(),
                                   "Tool not found: " + program.str());

  // argv[0] is the resolved executable path; remaining entries are the args.
  llvm::SmallVector<llvm::StringRef, 16> execArgs;
  execArgs.push_back(*programPath);
  for (const auto &arg : args)
    execArgs.push_back(arg);

  llvm::SmallString<128> stdoutPath, stderrPath;
  if (auto EC =
          llvm::sys::fs::createTemporaryFile("advisor-out", "tmp", stdoutPath))
    return llvm::createStringError(EC,
                                   "Failed to create temporary stdout file");
  if (auto EC =
          llvm::sys::fs::createTemporaryFile("advisor-err", "tmp", stderrPath))
    return llvm::createStringError(EC,
                                   "Failed to create temporary stderr file");

  std::optional<llvm::StringRef> redirects[] = {
      std::nullopt,                // stdin  — inherit
      llvm::StringRef(stdoutPath), // stdout — captured
      llvm::StringRef(stderrPath)  // stderr — captured
  };

  int exitCode =
      llvm::sys::ExecuteAndWait(*programPath, execArgs, envOverride, redirects,
                                static_cast<unsigned>(timeoutSeconds));

  ProcessRunner::ProcessResult result;
  result.exitCode = exitCode;
  result.executionTime = 0.0;

  if (auto buf = llvm::MemoryBuffer::getFile(stdoutPath))
    result.stdout = (*buf)->getBuffer().str();
  if (auto buf = llvm::MemoryBuffer::getFile(stderrPath))
    result.stderr = (*buf)->getBuffer().str();

  llvm::sys::fs::remove(stdoutPath);
  llvm::sys::fs::remove(stderrPath);

  return result;
}

Expected<ProcessRunner::ProcessResult>
ProcessRunner::run(llvm::StringRef program,
                   const llvm::SmallVector<std::string, 8> &args,
                   int timeoutSeconds) {
  return runImpl(program, args, /*envOverride=*/std::nullopt, timeoutSeconds);
}

Expected<ProcessRunner::ProcessResult> ProcessRunner::runWithEnv(
    llvm::StringRef program, const llvm::SmallVector<std::string, 8> &args,
    const llvm::SmallVector<std::string, 8> &env, int timeoutSeconds) {
  // Convert the environment strings to StringRef so we can pass them directly
  // to ExecuteAndWait.  This sets the *child* environment without touching the
  // parent process, which is the correct and thread-safe approach.
  llvm::SmallVector<llvm::StringRef, 16> envRefs;
  envRefs.reserve(env.size());
  for (const auto &e : env)
    envRefs.push_back(e);

  return runImpl(program, args, llvm::ArrayRef<llvm::StringRef>(envRefs),
                 timeoutSeconds);
}

} // namespace advisor
} // namespace llvm
