//===- llvm/System/Signals.h - Signal Handling support ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// unix signals occuring while your program is running.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_SIGNALS_H
#define LLVM_SYSTEM_SIGNALS_H

#include "llvm/System/Path.h"

namespace llvm {
namespace sys {

  /// This function registers signal handlers to ensure that if a signal gets
  /// delivered that the named file is removed.
  /// @brief Remove a file if a fatal signal occurs.
  bool RemoveFileOnSignal(const Path &Filename, std::string* ErrMsg = 0);

  /// This function registers a signal handler to ensure that if a fatal signal
  /// gets delivered to the process that the named directory and all its
  /// contents are removed.
  /// @brief Remove a directory if a fatal signal occurs.
  bool RemoveDirectoryOnSignal(const Path& path, std::string* ErrMsg = 0);

  /// When an error signal (such as SIBABRT or SIGSEGV) is delivered to the
  /// process, print a stack trace and then exit.
  /// @brief Print a stack trace if a fatal signal occurs.
  void PrintStackTraceOnErrorSignal();

  /// This function registers a function to be called when the user "interrupts"
  /// the program (typically by pressing ctrl-c).  When the user interrupts the
  /// program, the specified interrupt function is called instead of the program
  /// being killed, and the interrupt function automatically disabled.  Note
  /// that interrupt functions are not allowed to call any non-reentrant
  /// functions.  An null interrupt function pointer disables the current
  /// installed function.  Note also that the handler may be executed on a
  /// different thread on some platforms.
  /// @brief Register a function to be called when ctrl-c is pressed.
  void SetInterruptFunction(void (*IF)());
} // End sys namespace
} // End llvm namespace

#endif
