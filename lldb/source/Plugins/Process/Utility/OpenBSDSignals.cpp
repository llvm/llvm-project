//===-- OpenBSDSignals.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpenBSDSignals.h"

#ifdef __OpenBSD__
#include <csignal>

#define ADD_SIGCODE(signal_name, signal_value, code_name, code_value, ...)     \
  static_assert(signal_name == signal_value,                                   \
                "Value mismatch for signal number " #signal_name);             \
  static_assert(code_name == code_value,                                       \
                "Value mismatch for signal code " #code_name);                 \
  AddSignalCode(signal_value, code_value, __VA_ARGS__)
#else
#define ADD_SIGCODE(signal_name, signal_value, code_name, code_value, ...)     \
  AddSignalCode(signal_value, code_value, __VA_ARGS__)
#endif /* ifdef __OpenBSD */

using namespace lldb_private;

OpenBSDSignals::OpenBSDSignals() : UnixSignals() { Reset(); }

void OpenBSDSignals::Reset() {
  UnixSignals::Reset();

  // clang-format off
  // SIGILL
  ADD_SIGCODE(SIGILL, 4, ILL_ILLOPC, 1, "illegal opcode");
  ADD_SIGCODE(SIGILL, 4, ILL_ILLOPN, 2, "illegal operand");
  ADD_SIGCODE(SIGILL, 4, ILL_ILLADR, 3, "illegal addressing mode");
  ADD_SIGCODE(SIGILL, 4, ILL_ILLTRP, 4, "illegal trap");
  ADD_SIGCODE(SIGILL, 4, ILL_PRVOPC, 5, "privileged opcode");
  ADD_SIGCODE(SIGILL, 4, ILL_PRVREG, 6, "privileged register");
  ADD_SIGCODE(SIGILL, 4, ILL_COPROC, 7, "coprocessor error");
  ADD_SIGCODE(SIGILL, 4, ILL_BADSTK, 8, "internal stack error");
  ADD_SIGCODE(SIGILL, 4, ILL_BTCFI,  9, "IBT missing on indirect call");

  // SIGFPE
  ADD_SIGCODE(SIGFPE, 8, FPE_INTDIV, 1, "integer divide by zero");
  ADD_SIGCODE(SIGFPE, 8, FPE_INTOVF, 2, "integer overflow");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTDIV, 3, "floating point divide by zero");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTOVF, 4, "floating point overflow");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTUND, 5, "floating point underflow");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTRES, 6, "floating point inexact result");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTINV, 7, "invalid floating point operation");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTSUB, 8, "subscript out of range");

  // SIGBUS
  ADD_SIGCODE(SIGBUS, 10, BUS_ADRALN, 1, "invalid address alignment");
  ADD_SIGCODE(SIGBUS, 10, BUS_ADRERR, 2, "non-existent physical address");
  ADD_SIGCODE(SIGBUS, 10, BUS_OBJERR, 3, "object specific hardware error");

  // SIGSEGV
  ADD_SIGCODE(SIGSEGV, 11, SEGV_MAPERR, 1, "address not mapped to object",
                SignalCodePrintOption::Address);
  ADD_SIGCODE(SIGSEGV, 11, SEGV_ACCERR, 2, "invalid permissions for mapped object",
                SignalCodePrintOption::Address);

  //        SIGNO NAME           SUPPRESS STOP   NOTIFY DESCRIPTION
  //        ===== ============== ======== ====== ====== ========================
  AddSignal(32,   "SIGTHR",      false,   false, false, "thread library AST");
  // clang-format on
}
