//===- SystemUtils.cpp - Utilities for low-level system tasks -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions used to do a variety of low-level, often
// system-specific, tasks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Streams.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/System/Process.h"
#include "llvm/System/Program.h"
#include <ostream>
using namespace llvm;

bool llvm::CheckBitcodeOutputToConsole(std::ostream* stream_to_check,
                                       bool print_warning) {
  if (stream_to_check == cout.stream() &&
      sys::Process::StandardOutIsDisplayed()) {
    if (print_warning) {
      cerr << "WARNING: You're attempting to print out a bitcode file.\n"
           << "This is inadvisable as it may cause display problems. If\n"
           << "you REALLY want to taste LLVM bitcode first-hand, you\n"
           << "can force output with the `-f' option.\n\n";
    }
    return true;
  }
  return false;
}

/// FindExecutable - Find a named executable, giving the argv[0] of program
/// being executed. This allows us to find another LLVM tool if it is built
/// into the same directory, but that directory is neither the current
/// directory, nor in the PATH.  If the executable cannot be found, return an
/// empty string.
///
#undef FindExecutable   // needed on windows :(
sys::Path llvm::FindExecutable(const std::string &ExeName,
                               const std::string &ProgramPath) {
  // First check the directory that the calling program is in.  We can do this
  // if ProgramPath contains at least one / character, indicating that it is a
  // relative path to bugpoint itself.
  sys::Path Result ( ProgramPath );
  Result.eraseComponent();
  if (!Result.isEmpty()) {
    Result.appendComponent(ExeName);
    if (Result.canExecute())
      return Result;
  }

  return sys::Program::FindProgramByName(ExeName);
}
