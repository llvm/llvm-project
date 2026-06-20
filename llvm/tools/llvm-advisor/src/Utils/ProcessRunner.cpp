//===------------------- ProcessRunner.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Safe, allowlisted subprocess execution for external_fallback capabilities.
//
//===----------------------------------------------------------------------===//

#include "Utils/ProcessRunner.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

#include <chrono>

using namespace llvm;
using namespace llvm::advisor;

namespace {

/// RAII guard that removes a temporary file on destruction.
struct TempFileGuard {
  StringRef Path;
  explicit TempFileGuard(StringRef Path) : Path(Path) {}
  ~TempFileGuard() { sys::fs::remove(Path); }
};

} // namespace

void ProcessRunner::allow(StringRef Program) {
  ToolPolicy &Policy = AllowList[Program];
  Policy.AllowAll = true;
  Policy.Flags.clear();
}

void ProcessRunner::allow(StringRef Program, ArrayRef<StringRef> Flags) {
  ToolPolicy &Policy = AllowList[Program];
  Policy.AllowAll = false;
  Policy.Flags.clear();
  for (StringRef Flag : Flags)
    Policy.Flags.insert(Flag);
}

bool ProcessRunner::isAllowedArg(StringRef Program, StringRef Arg) const {
  if (Arg.empty() || Arg.contains('\0') || Arg.contains('\n') ||
      Arg.contains('\r'))
    return false;
  if (!Arg.starts_with("-"))
    return true;

  StringMap<ToolPolicy>::const_iterator I = AllowList.find(Program);
  if (I == AllowList.end())
    return false;
  if (I->second.AllowAll)
    return true;
  if (I->second.Flags.empty())
    return false;
  if (I->second.Flags.contains(Arg))
    return true;

  StringRef Name = Arg.take_until([](char C) { return C == '='; });
  return I->second.Flags.contains(Name);
}

Expected<ProcessResult> ProcessRunner::run(StringRef Program,
                                           ArrayRef<std::string> Arguments,
                                           unsigned TimeoutSeconds) const {
  if (!AllowList.contains(Program))
    return createStringError(inconvertibleErrorCode(),
                             "program not allowlisted: %s",
                             Program.str().c_str());

  ErrorOr<std::string> Resolved = sys::findProgramByName(Program);
  if (!Resolved)
    return createStringError(Resolved.getError(), "cannot find program: %s",
                             Program.str().c_str());

  SmallVector<StringRef, 16> Args;
  Args.push_back(*Resolved);
  for (const auto &Arg : Arguments) {
    if (!isAllowedArg(Program, Arg))
      return createStringError(inconvertibleErrorCode(),
                               "argument not allowlisted: %s", Arg.c_str());
    Args.push_back(Arg);
  }

  SmallString<128> StdoutPath, StderrPath;
  if (auto EC = sys::fs::createTemporaryFile("advisor-out", "tmp", StdoutPath))
    return createStringError(EC, "failed to create stdout temp file");
  TempFileGuard StdoutGuard(StdoutPath);

  if (auto EC = sys::fs::createTemporaryFile("advisor-err", "tmp", StderrPath))
    return createStringError(EC, "failed to create stderr temp file");
  TempFileGuard StderrGuard(StderrPath);

  std::optional<StringRef> Redirects[] = {
      std::nullopt,          // stdin
      StringRef(StdoutPath), // stdout
      StringRef(StderrPath)  // stderr
  };

  auto StartTime = std::chrono::steady_clock::now();
  int ExitCode = sys::ExecuteAndWait(*Resolved, Args, std::nullopt, Redirects,
                                     TimeoutSeconds);
  auto EndTime = std::chrono::steady_clock::now();

  ProcessResult Result;
  Result.ExitCode = ExitCode;
  Result.WallTimeNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(EndTime - StartTime)
          .count();

  if (auto Buf = MemoryBuffer::getFile(StdoutPath))
    Result.Stdout = (*Buf)->getBuffer().str();
  if (auto Buf = MemoryBuffer::getFile(StderrPath))
    Result.Stderr = (*Buf)->getBuffer().str();

  return Result;
}
