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
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include <vector>

extern char **environ;

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

  // Prepare environment variables (current environment + our additions)
  SmallVector<StringRef> environment;

  // Get current environment variables
  char **envp = environ;
  while (*envp) {
    environment.emplace_back(*envp);
    ++envp;
  }

  // Add our additional environment variables
  for (const auto &var : env) {
    environment.emplace_back(var);
  }

  // Prepare arguments
  llvm::SmallVector<llvm::StringRef> argv;
  argv.push_back(program);
  for (const auto &arg : args)
    argv.push_back(arg);

  // Create temporary files for stdout and stderr
  SmallString<64> stdoutFile, stderrFile;
  sys::fs::createTemporaryFile("stdout", "tmp", stdoutFile);
  sys::fs::createTemporaryFile("stderr", "tmp", stderrFile);

  // Set up redirects
  SmallVector<std::optional<StringRef>, 3> redirects;
  redirects.push_back(std::nullopt);     // stdin
  redirects.push_back(stdoutFile.str()); // stdout
  redirects.push_back(stderrFile.str()); // stderr

  // Run the process with custom environment
  bool executionFailed = false;
  int status = sys::ExecuteAndWait(program, argv,
                                   environment,    // Custom environment
                                   redirects,      // Redirects
                                   timeoutSeconds, // Timeout
                                   0,              // Memory limit (no limit)
                                   nullptr, // Standard output (using file)
                                   &executionFailed, // Execution failed flag
                                   nullptr           // Process statistics
  );

  if (executionFailed) {
    return createStringError(std::errc::no_such_file_or_directory,
                             "Failed to execute process");
  }

  // Read stdout and stderr from temporary files
  std::string stdoutStr, stderrStr;

  auto stdoutBuffer = MemoryBuffer::getFile(stdoutFile);
  if (stdoutBuffer) {
    stdoutStr = stdoutBuffer.get()->getBuffer().str();
  }

  auto stderrBuffer = MemoryBuffer::getFile(stderrFile);
  if (stderrBuffer) {
    stderrStr = stderrBuffer.get()->getBuffer().str();
  }

  // Clean up temporary files
  sys::fs::remove(stdoutFile);
  sys::fs::remove(stderrFile);

  // Process result
  ProcessResult result;
  result.exitCode = status;
  result.stdout = stdoutStr;
  result.stderr = stderrStr;
  result.executionTime = 0; // Not tracking time
  return result;
}

} // namespace advisor
} // namespace llvm
