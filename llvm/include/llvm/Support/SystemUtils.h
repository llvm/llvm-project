//===- SystemUtils.h - Utilities to do low-level system stuff ---*- C++ -*-===//
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

#ifndef LLVM_SUPPORT_SYSTEMUTILS_H
#define LLVM_SUPPORT_SYSTEMUTILS_H

#include "llvm/System/Program.h"

namespace llvm {

/// Determine if the ostream provided is connected to the std::cout and
/// displayed or not (to a console window). If so, generate a warning message
/// advising against display of bitcode and return true. Otherwise just return
/// false
/// @brief Check for output written to a console
bool CheckBitcodeOutputToConsole(
  std::ostream* stream_to_check, ///< The stream to be checked
  bool print_warning = true ///< Control whether warnings are printed
);

/// FindExecutable - Find a named executable, giving the argv[0] of program
/// being executed. This allows us to find another LLVM tool if it is built into
/// the same directory, but that directory is neither the current directory, nor
/// in the PATH.  If the executable cannot be found, return an empty string.
/// @brief Find a named executable.
sys::Path FindExecutable(const std::string &ExeName,
                         const std::string &ProgramPath);

} // End llvm namespace

#endif
