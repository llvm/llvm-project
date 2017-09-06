//===-- Program.cpp - Implement OS Program Concept --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the operating system Program concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Program.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include <system_error>
using namespace llvm;
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

static bool Execute(ProcessInfo &PI, StringRef Program, const char **Args,
                    const char **Env, const StringRef **Redirects,
                    unsigned MemoryLimit, std::string *ErrMsg);

int sys::ExecuteAndWait(StringRef Program, const char **Args, const char **Envp,
                        const StringRef **Redirects, unsigned SecondsToWait,
                        unsigned MemoryLimit, std::string *ErrMsg,
                        bool *ExecutionFailed) {
  ProcessInfo PI;
  if (Execute(PI, Program, Args, Envp, Redirects, MemoryLimit, ErrMsg)) {
    if (ExecutionFailed)
      *ExecutionFailed = false;
    ProcessInfo Result = Wait(
        PI, SecondsToWait, /*WaitUntilTerminates=*/SecondsToWait == 0, ErrMsg);
    return Result.ReturnCode;
  }

  if (ExecutionFailed)
    *ExecutionFailed = true;

  return -1;
}

ProcessInfo sys::ExecuteNoWait(StringRef Program, const char **Args,
                               const char **Envp, const StringRef **Redirects,
                               unsigned MemoryLimit, std::string *ErrMsg,
                               bool *ExecutionFailed) {
  ProcessInfo PI;
  if (ExecutionFailed)
    *ExecutionFailed = false;
  if (!Execute(PI, Program, Args, Envp, Redirects, MemoryLimit, ErrMsg))
    if (ExecutionFailed)
      *ExecutionFailed = true;

  return PI;
}

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Program.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Program.inc"
#endif
