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
runImpl(llvm::StringRef Program, const llvm::SmallVector<std::string, 8> &Args,
        std::optional<llvm::ArrayRef<llvm::StringRef>> EnvOverride,
        int TimeoutSeconds) {
  auto ProgramPath = llvm::sys::findProgramByName(Program);
  if (!ProgramPath)
    return llvm::createStringError(ProgramPath.getError(),
                                   "Tool not found: " + Program.str());

  // argv[0] is the resolved executable path; remaining entries are the args.
  llvm::SmallVector<llvm::StringRef, 16> ExecArgs;
  ExecArgs.push_back(*ProgramPath);
  for (const auto &Arg : Args)
    ExecArgs.push_back(Arg);

  llvm::SmallString<128> StdoutPath, StderrPath;
  if (auto EC =
          llvm::sys::fs::createTemporaryFile("advisor-out", "tmp", StdoutPath))
    return llvm::createStringError(EC,
                                   "Failed to create temporary stdout file");
  if (auto EC =
          llvm::sys::fs::createTemporaryFile("advisor-err", "tmp", StderrPath))
    return llvm::createStringError(EC,
                                   "Failed to create temporary stderr file");

  std::optional<llvm::StringRef> Redirects[] = {
      std::nullopt,                // stdin  — inherit
      llvm::StringRef(StdoutPath), // stdout — captured
      llvm::StringRef(StderrPath)  // stderr — captured
  };

  int ExitCode =
      llvm::sys::ExecuteAndWait(*ProgramPath, ExecArgs, EnvOverride, Redirects,
                                static_cast<unsigned>(TimeoutSeconds));

  ProcessRunner::ProcessResult Result;
  Result.exitCode = ExitCode;
  Result.executionTime = 0.0;

  if (auto Buf = llvm::MemoryBuffer::getFile(StdoutPath))
    Result.stdout = (*Buf)->getBuffer().str();
  if (auto Buf = llvm::MemoryBuffer::getFile(StderrPath))
    Result.stderr = (*Buf)->getBuffer().str();

  llvm::sys::fs::remove(StdoutPath);
  llvm::sys::fs::remove(StderrPath);

  return Result;
}

Expected<ProcessRunner::ProcessResult>
ProcessRunner::run(llvm::StringRef Program,
                   const llvm::SmallVector<std::string, 8> &Args,
                   int TimeoutSeconds) {
  return runImpl(Program, Args, /*envOverride=*/std::nullopt, TimeoutSeconds);
}

Expected<ProcessRunner::ProcessResult> ProcessRunner::runWithEnv(
    llvm::StringRef Program, const llvm::SmallVector<std::string, 8> &Args,
    const llvm::SmallVector<std::string, 8> &Env, int TimeoutSeconds) {
  // Convert the environment strings to StringRef so we can pass them directly
  // to ExecuteAndWait.  This sets the *child* environment without touching the
  // parent process, which is the correct and thread-safe approach.
  llvm::SmallVector<llvm::StringRef, 16> EnvRefs;
  EnvRefs.reserve(Env.size());
  for (const auto &E : Env)
    EnvRefs.push_back(E);

  return runImpl(Program, Args, llvm::ArrayRef<llvm::StringRef>(EnvRefs),
                 TimeoutSeconds);
}

} // namespace advisor
} // namespace llvm
