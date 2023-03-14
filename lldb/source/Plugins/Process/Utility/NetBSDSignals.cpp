//===-- NetBSDSignals.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NetBSDSignals.h"

using namespace lldb_private;

NetBSDSignals::NetBSDSignals() : UnixSignals() { Reset(); }

void NetBSDSignals::Reset() {
  UnixSignals::Reset();

  // clang-format off
  // SIGILL
  AddSignalCode(4, 1 /*ILL_ILLOPC*/, "illegal opcode");
  AddSignalCode(4, 2 /*ILL_ILLOPN*/, "illegal operand");
  AddSignalCode(4, 3 /*ILL_ILLADR*/, "illegal addressing mode");
  AddSignalCode(4, 4 /*ILL_ILLTRP*/, "illegal trap");
  AddSignalCode(4, 5 /*ILL_PRVOPC*/, "privileged opcode");
  AddSignalCode(4, 6 /*ILL_PRVREG*/, "privileged register");
  AddSignalCode(4, 7 /*ILL_COPROC*/, "coprocessor error");
  AddSignalCode(4, 8 /*ILL_BADSTK*/, "internal stack error");

  // SIGFPE
  AddSignalCode(8, 1 /*FPE_INTDIV*/, "integer divide by zero");
  AddSignalCode(8, 2 /*FPE_INTOVF*/, "integer overflow");
  AddSignalCode(8, 3 /*FPE_FLTDIV*/, "floating point divide by zero");
  AddSignalCode(8, 4 /*FPE_FLTOVF*/, "floating point overflow");
  AddSignalCode(8, 5 /*FPE_FLTUND*/, "floating point underflow");
  AddSignalCode(8, 6 /*FPE_FLTRES*/, "floating point inexact result");
  AddSignalCode(8, 7 /*FPE_FLTINV*/, "invalid floating point operation");
  AddSignalCode(8, 8 /*FPE_FLTSUB*/, "subscript out of range");

  // SIGBUS
  AddSignalCode(10, 1 /*BUS_ADRALN*/, "invalid address alignment");
  AddSignalCode(10, 2 /*BUS_ADRERR*/, "non-existent physical address");
  AddSignalCode(10, 3 /*BUS_OBJERR*/, "object specific hardware error");

  // SIGSEGV
  AddSignalCode(11, 1 /*SEGV_MAPERR*/, "address not mapped to object",
                SignalCodePrintOption::Address);
  AddSignalCode(11, 2 /*SEGV_ACCERR*/, "invalid permissions for mapped object",
                SignalCodePrintOption::Address);

  //        SIGNO  NAME          SUPPRESS STOP   NOTIFY DESCRIPTION
  //        ===== ============== ======== ====== ====== ========================
  AddSignal(32,   "SIGPWR",      false,   true,  true,  "power fail/restart (not reset when caught)");
  AddSignal(33,   "SIGRTMIN",    false,   false, false, "real time signal 0");
  AddSignal(34,   "SIGRTMIN+1",  false,   false, false, "real time signal 1");
  AddSignal(35,   "SIGRTMIN+2",  false,   false, false, "real time signal 2");
  AddSignal(36,   "SIGRTMIN+3",  false,   false, false, "real time signal 3");
  AddSignal(37,   "SIGRTMIN+4",  false,   false, false, "real time signal 4");
  AddSignal(38,   "SIGRTMIN+5",  false,   false, false, "real time signal 5");
  AddSignal(39,   "SIGRTMIN+6",  false,   false, false, "real time signal 6");
  AddSignal(40,   "SIGRTMIN+7",  false,   false, false, "real time signal 7");
  AddSignal(41,   "SIGRTMIN+8",  false,   false, false, "real time signal 8");
  AddSignal(42,   "SIGRTMIN+9",  false,   false, false, "real time signal 9");
  AddSignal(43,   "SIGRTMIN+10", false,   false, false, "real time signal 10");
  AddSignal(44,   "SIGRTMIN+11", false,   false, false, "real time signal 11");
  AddSignal(45,   "SIGRTMIN+12", false,   false, false, "real time signal 12");
  AddSignal(46,   "SIGRTMIN+13", false,   false, false, "real time signal 13");
  AddSignal(47,   "SIGRTMIN+14", false,   false, false, "real time signal 14");
  AddSignal(48,   "SIGRTMIN+15", false,   false, false, "real time signal 15");
  AddSignal(49,   "SIGRTMIN-14", false,   false, false, "real time signal 16");
  AddSignal(50,   "SIGRTMAX-13", false,   false, false, "real time signal 17");
  AddSignal(51,   "SIGRTMAX-12", false,   false, false, "real time signal 18");
  AddSignal(52,   "SIGRTMAX-11", false,   false, false, "real time signal 19");
  AddSignal(53,   "SIGRTMAX-10", false,   false, false, "real time signal 20");
  AddSignal(54,   "SIGRTMAX-9",  false,   false, false, "real time signal 21");
  AddSignal(55,   "SIGRTMAX-8",  false,   false, false, "real time signal 22");
  AddSignal(56,   "SIGRTMAX-7",  false,   false, false, "real time signal 23");
  AddSignal(57,   "SIGRTMAX-6",  false,   false, false, "real time signal 24");
  AddSignal(58,   "SIGRTMAX-5",  false,   false, false, "real time signal 25");
  AddSignal(59,   "SIGRTMAX-4",  false,   false, false, "real time signal 26");
  AddSignal(60,   "SIGRTMAX-3",  false,   false, false, "real time signal 27");
  AddSignal(61,   "SIGRTMAX-2",  false,   false, false, "real time signal 28");
  AddSignal(62,   "SIGRTMAX-1",  false,   false, false, "real time signal 29");
  AddSignal(63,   "SIGRTMAX",    false,   false, false, "real time signal 30");
  // clang-format on
}
