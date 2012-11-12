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

#include <string>

namespace llvm {
  class raw_ostream;
  namespace sys { class Path; }

/// Determine if the raw_ostream provided is connected to a terminal. If so,
/// generate a warning message to errs() advising against display of bitcode
/// and return true. Otherwise just return false.
/// @brief Check for output written to a console
bool CheckBitcodeOutputToConsole(
  raw_ostream &stream_to_check, ///< The stream to be checked
  bool print_warning = true     ///< Control whether warnings are printed
);

/// PrependMainExecutablePath - Prepend the path to the program being executed
/// to \p ExeName, given the value of argv[0] and the address of main()
/// itself. This allows us to find another LLVM tool if it is built in the same
/// directory. An empty string is returned on error; note that this function
/// just mainpulates the path and doesn't check for executability.
/// @brief Find a named executable.
sys::Path PrependMainExecutablePath(const std::string &ExeName,
                                    const char *Argv0, void *MainAddr);

} // End llvm namespace

#endif
