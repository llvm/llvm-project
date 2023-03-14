//===-- LinuxSignals.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LinuxSignals.h"

using namespace lldb_private;

LinuxSignals::LinuxSignals() : UnixSignals() { Reset(); }

void LinuxSignals::Reset() {
  m_signals.clear();
  // clang-format off
  //        SIGNO   NAME            SUPPRESS  STOP    NOTIFY  DESCRIPTION
  //        ======  ==============  ========  ======  ======  ===================================================
  AddSignal(1,      "SIGHUP",       false,    true,   true,   "hangup");
  AddSignal(2,      "SIGINT",       true,     true,   true,   "interrupt");
  AddSignal(3,      "SIGQUIT",      false,    true,   true,   "quit");

  AddSignal(4,      "SIGILL",       false,    true,   true,   "illegal instruction");
  AddSignalCode(4, 1 /*ILL_ILLOPC*/, "illegal opcode");
  AddSignalCode(4, 2 /*ILL_ILLOPN*/, "illegal operand");
  AddSignalCode(4, 3 /*ILL_ILLADR*/, "illegal addressing mode");
  AddSignalCode(4, 4 /*ILL_ILLTRP*/, "illegal trap");
  AddSignalCode(4, 5 /*ILL_PRVOPC*/, "privileged opcode");
  AddSignalCode(4, 6 /*ILL_PRVREG*/, "privileged register");
  AddSignalCode(4, 7 /*ILL_COPROC*/, "coprocessor error");
  AddSignalCode(4, 8 /*ILL_BADSTK*/, "internal stack error");

  AddSignal(5,      "SIGTRAP",      true,     true,   true,   "trace trap (not reset when caught)");
  AddSignal(6,      "SIGABRT",      false,    true,   true,   "abort()/IOT trap", "SIGIOT");

  AddSignal(7,      "SIGBUS",       false,    true,   true,   "bus error");
  AddSignalCode(7, 1 /*BUS_ADRALN*/, "illegal alignment");
  AddSignalCode(7, 2 /*BUS_ADRERR*/, "illegal address");
  AddSignalCode(7, 3 /*BUS_OBJERR*/, "hardware error");

  AddSignal(8,      "SIGFPE",       false,    true,   true,   "floating point exception");
  AddSignalCode(8, 1 /*FPE_INTDIV*/, "integer divide by zero");
  AddSignalCode(8, 2 /*FPE_INTOVF*/, "integer overflow");
  AddSignalCode(8, 3 /*FPE_FLTDIV*/, "floating point divide by zero");
  AddSignalCode(8, 4 /*FPE_FLTOVF*/, "floating point overflow");
  AddSignalCode(8, 5 /*FPE_FLTUND*/, "floating point underflow");
  AddSignalCode(8, 6 /*FPE_FLTRES*/, "floating point inexact result");
  AddSignalCode(8, 7 /*FPE_FLTINV*/, "floating point invalid operation");
  AddSignalCode(8, 8 /*FPE_FLTSUB*/, "subscript out of range");

  AddSignal(9,      "SIGKILL",      false,    true,   true,   "kill");
  AddSignal(10,     "SIGUSR1",      false,    true,   true,   "user defined signal 1");

  AddSignal(11,     "SIGSEGV",      false,    true,   true,   "segmentation violation");
  AddSignalCode(11, 1 /*SEGV_MAPERR*/, "address not mapped to object", SignalCodePrintOption::Address);
  AddSignalCode(11, 2 /*SEGV_ACCERR*/, "invalid permissions for mapped object", SignalCodePrintOption::Address);
  AddSignalCode(11, 3 /*SEGV_BNDERR*/, "failed address bounds checks", SignalCodePrintOption::Bounds);
  AddSignalCode(11, 8 /*SEGV_MTEAERR*/, "async tag check fault");
  AddSignalCode(11, 9 /*SEGV_MTESERR*/, "sync tag check fault", SignalCodePrintOption::Address);
  // Some platforms will occasionally send nonstandard spurious SI_KERNEL
  // codes. One way to get this is via unaligned SIMD loads. Treat it as invalid address.
  AddSignalCode(11, 0x80 /*SI_KERNEL*/, "invalid address", SignalCodePrintOption::Address);

  AddSignal(12,     "SIGUSR2",      false,    true,   true,   "user defined signal 2");
  AddSignal(13,     "SIGPIPE",      false,    true,   true,   "write to pipe with reading end closed");
  AddSignal(14,     "SIGALRM",      false,    false,  false,  "alarm");
  AddSignal(15,     "SIGTERM",      false,    true,   true,   "termination requested");
  AddSignal(16,     "SIGSTKFLT",    false,    true,   true,   "stack fault");
  AddSignal(17,     "SIGCHLD",      false,    false,  true,   "child status has changed", "SIGCLD");
  AddSignal(18,     "SIGCONT",      false,    false,  true,   "process continue");
  AddSignal(19,     "SIGSTOP",      true,     true,   true,   "process stop");
  AddSignal(20,     "SIGTSTP",      false,    true,   true,   "tty stop");
  AddSignal(21,     "SIGTTIN",      false,    true,   true,   "background tty read");
  AddSignal(22,     "SIGTTOU",      false,    true,   true,   "background tty write");
  AddSignal(23,     "SIGURG",       false,    true,   true,   "urgent data on socket");
  AddSignal(24,     "SIGXCPU",      false,    true,   true,   "CPU resource exceeded");
  AddSignal(25,     "SIGXFSZ",      false,    true,   true,   "file size limit exceeded");
  AddSignal(26,     "SIGVTALRM",    false,    true,   true,   "virtual time alarm");
  AddSignal(27,     "SIGPROF",      false,    false,  false,  "profiling time alarm");
  AddSignal(28,     "SIGWINCH",     false,    true,   true,   "window size changes");
  AddSignal(29,     "SIGIO",        false,    true,   true,   "input/output ready/Pollable event", "SIGPOLL");
  AddSignal(30,     "SIGPWR",       false,    true,   true,   "power failure");
  AddSignal(31,     "SIGSYS",       false,    true,   true,   "invalid system call");
  AddSignal(32,     "SIG32",        false,    false,  false,  "threading library internal signal 1");
  AddSignal(33,     "SIG33",        false,    false,  false,  "threading library internal signal 2");
  AddSignal(34,     "SIGRTMIN",     false,    false,  false,  "real time signal 0");
  AddSignal(35,     "SIGRTMIN+1",   false,    false,  false,  "real time signal 1");
  AddSignal(36,     "SIGRTMIN+2",   false,    false,  false,  "real time signal 2");
  AddSignal(37,     "SIGRTMIN+3",   false,    false,  false,  "real time signal 3");
  AddSignal(38,     "SIGRTMIN+4",   false,    false,  false,  "real time signal 4");
  AddSignal(39,     "SIGRTMIN+5",   false,    false,  false,  "real time signal 5");
  AddSignal(40,     "SIGRTMIN+6",   false,    false,  false,  "real time signal 6");
  AddSignal(41,     "SIGRTMIN+7",   false,    false,  false,  "real time signal 7");
  AddSignal(42,     "SIGRTMIN+8",   false,    false,  false,  "real time signal 8");
  AddSignal(43,     "SIGRTMIN+9",   false,    false,  false,  "real time signal 9");
  AddSignal(44,     "SIGRTMIN+10",  false,    false,  false,  "real time signal 10");
  AddSignal(45,     "SIGRTMIN+11",  false,    false,  false,  "real time signal 11");
  AddSignal(46,     "SIGRTMIN+12",  false,    false,  false,  "real time signal 12");
  AddSignal(47,     "SIGRTMIN+13",  false,    false,  false,  "real time signal 13");
  AddSignal(48,     "SIGRTMIN+14",  false,    false,  false,  "real time signal 14");
  AddSignal(49,     "SIGRTMIN+15",  false,    false,  false,  "real time signal 15");
  AddSignal(50,     "SIGRTMAX-14",  false,    false,  false,  "real time signal 16"); // switching to SIGRTMAX-xxx to match "kill -l" output
  AddSignal(51,     "SIGRTMAX-13",  false,    false,  false,  "real time signal 17");
  AddSignal(52,     "SIGRTMAX-12",  false,    false,  false,  "real time signal 18");
  AddSignal(53,     "SIGRTMAX-11",  false,    false,  false,  "real time signal 19");
  AddSignal(54,     "SIGRTMAX-10",  false,    false,  false,  "real time signal 20");
  AddSignal(55,     "SIGRTMAX-9",   false,    false,  false,  "real time signal 21");
  AddSignal(56,     "SIGRTMAX-8",   false,    false,  false,  "real time signal 22");
  AddSignal(57,     "SIGRTMAX-7",   false,    false,  false,  "real time signal 23");
  AddSignal(58,     "SIGRTMAX-6",   false,    false,  false,  "real time signal 24");
  AddSignal(59,     "SIGRTMAX-5",   false,    false,  false,  "real time signal 25");
  AddSignal(60,     "SIGRTMAX-4",   false,    false,  false,  "real time signal 26");
  AddSignal(61,     "SIGRTMAX-3",   false,    false,  false,  "real time signal 27");
  AddSignal(62,     "SIGRTMAX-2",   false,    false,  false,  "real time signal 28");
  AddSignal(63,     "SIGRTMAX-1",   false,    false,  false,  "real time signal 29");
  AddSignal(64,     "SIGRTMAX",     false,    false,  false,  "real time signal 30");
  // clang-format on
}
