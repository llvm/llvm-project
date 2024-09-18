//===-- QNXSignals.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QNXSignals.h"

#if defined(__QNX__)
#include <signal.h>
#include <sys/siginfo.h>

#define ADD_SIGCODE(signal_name, signal_value, code_name, code_value, ...)     \
  static_assert(signal_name == signal_value,                                   \
                "Value mismatch for signal number " #signal_name);             \
  static_assert(code_name == code_value,                                       \
                "Value mismatch for signal code " #code_name #code_value);     \
  AddSignalCode(signal_value, code_value, __VA_ARGS__)
#else
#define ADD_SIGCODE(signal_name, signal_value, code_name, code_value, ...)     \
  AddSignalCode(signal_value, code_value, __VA_ARGS__)
#endif // defined(__QNX__)

using namespace lldb_private;

QNXSignals::QNXSignals() : UnixSignals() { Reset(); }

void QNXSignals::Reset() {
  m_signals.clear();
  // clang-format off
  //        SIGNO   NAME            SUPPRESS  STOP    NOTIFY  DESCRIPTION
  //        ======  ==============  ========  ======  ======  ===================================================
  AddSignal(1,      "SIGHUP",       false,    true,   true,   "hangup");
  AddSignal(2,      "SIGINT",       true,     true,   true,   "interrupt");
  AddSignal(3,      "SIGQUIT",      false,    true,   true,   "quit");

  AddSignal(4,      "SIGILL",       false,    true,   true,   "illegal instruction");
  ADD_SIGCODE(SIGILL, 4, ILL_ILLOPC, 1, "illegal opcode");
  ADD_SIGCODE(SIGILL, 4, ILL_ILLOPN, 2, "illegal operand");
  ADD_SIGCODE(SIGILL, 4, ILL_ILLADR, 3, "illegal addressing mode");
  ADD_SIGCODE(SIGILL, 4, ILL_ILLTRP, 4, "illegal trap");
  ADD_SIGCODE(SIGILL, 4, ILL_PRVOPC, 5, "privileged opcode");
  ADD_SIGCODE(SIGILL, 4, ILL_PRVREG, 6, "privileged register");
  ADD_SIGCODE(SIGILL, 4, ILL_COPROC, 7, "coprocessor error");
  ADD_SIGCODE(SIGILL, 4, ILL_BADSTK, 8, "internal stack error");

  AddSignal(5,      "SIGTRAP",      true,     true,   true,   "trace trap (not reset when caught)");
  ADD_SIGCODE(SIGTRAP, 5, TRAP_BRKPT, 1, "Break Point");
  ADD_SIGCODE(SIGTRAP, 5, TRAP_TRACE, 2, "Trace");
  ADD_SIGCODE(SIGTRAP, 5, TRAP_KDBRK, 3, "Kdebug Break Point");
  ADD_SIGCODE(SIGTRAP, 5, TRAP_CRASH, 4, "Crash");

  AddSignal(6,      "SIGIOT",       false,    true,   true,   "IOT trap/abort()", "SIGABRT");
  AddSignal(7,      "SIGEMT",       false,    true,   true,   "EMT instruction/Mutex deadlock", "SIGDEADLK");

  AddSignal(8,      "SIGFPE",       false,    true,   true,   "floating point exception");
  ADD_SIGCODE(SIGFPE, 8, FPE_INTDIV, 1,  "Integer divide by zero");
  ADD_SIGCODE(SIGFPE, 8, FPE_INTOVF, 2,  "Integer overflow");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTDIV, 3,  "Floating point divide by zero");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTOVF, 4,  "Floating point overflow");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTUND, 5,  "Floating point underflow");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTRES, 6,  "Floating point inexact result");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTINV, 7,  "Invalid floating point operation");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTSUB, 8,  "Subscript out of range");
  ADD_SIGCODE(SIGFPE, 8, FPE_NOFPU,  9,  "No FPU or emulator");
  ADD_SIGCODE(SIGFPE, 8, FPE_NOMEM,  10, "No kernel space for FPU save area");

  AddSignal(9,      "SIGKILL",      false,    true,   true,   "kill");

  AddSignal(10,     "SIGBUS",       false,    true,   true,   "bus error");
  ADD_SIGCODE(SIGBUS, 10, BUS_ADRALN,  1,  "Invalid address alignment");
  ADD_SIGCODE(SIGBUS, 10, BUS_ADRERR,  2,  "Non-existant physical address");
  ADD_SIGCODE(SIGBUS, 10, BUS_OBJERR,  3,  "Object specific hardware error (e.g. NMI parity error)");
  ADD_SIGCODE(SIGBUS, 10, BUS_BADPAGE, 4,  "N/A");
  ADD_SIGCODE(SIGBUS, 10, BUS_ENDOBJ,  5,  "N/A");
  ADD_SIGCODE(SIGBUS, 10, BUS_EOTHER,  6,  "Other not translated error faults");
  ADD_SIGCODE(SIGBUS, 10, BUS_ENOENT,  7,  "N/A");
  ADD_SIGCODE(SIGBUS, 10, BUS_EAGAIN,  8,  "Transient fault - many times a no memory condition");
  ADD_SIGCODE(SIGBUS, 10, BUS_ENOMEM,  9,  "No memory available to execute request");
  ADD_SIGCODE(SIGBUS, 10, BUS_EFAULT,  10, "N/A");
  ADD_SIGCODE(SIGBUS, 10, BUS_EINVAL,  11, "Invalid argument(s) passed");
  ADD_SIGCODE(SIGBUS, 10, BUS_EACCES,  12, "N/A");
  ADD_SIGCODE(SIGBUS, 10, BUS_EBADF,   13, "Invalid file descriptor");
  ADD_SIGCODE(SIGBUS, 10, BUS_SRVERR,  50, "N/A");

  AddSignal(11,     "SIGSEGV",      false,    true,   true,   "segmentation violation");
  ADD_SIGCODE(SIGSEGV, 11, SEGV_MAPERR, 1, "Address not mapped");
  ADD_SIGCODE(SIGSEGV, 11, SEGV_ACCERR, 2, "No permissions");
  ADD_SIGCODE(SIGSEGV, 11, SEGV_STKERR, 3, "Stack exception");
  ADD_SIGCODE(SIGSEGV, 11, SEGV_GPERR,  4, "General protection");
  ADD_SIGCODE(SIGSEGV, 11, SEGV_IRQERR, 5, "Interrupt handler fault");

  AddSignal(12,     "SIGSYS",       false,    true,   true,   "invalid system call");
  AddSignal(13,     "SIGPIPE",      false,    true,   true,   "write to pipe with reading end closed");
  AddSignal(14,     "SIGALRM",      false,    false,  false,  "alarm");
  AddSignal(15,     "SIGTERM",      false,    true,   true,   "termination requested");
  AddSignal(16,     "SIGUSR1",      false,    true,   true,   "user defined signal 1");
  AddSignal(17,     "SIGUSR2",      false,    true,   true,   "user defined signal 2");

  AddSignal(18,     "SIGCHLD",      false,    false,  true,   "child status has changed", "SIGCLD");
  ADD_SIGCODE(SIGCHLD, 18, CLD_EXITED,    1, "Child has exited");
  ADD_SIGCODE(SIGCHLD, 18, CLD_KILLED,    2, "Child was killed");
  ADD_SIGCODE(SIGCHLD, 18, CLD_DUMPED,    3, "Child terminated abnormally");
  ADD_SIGCODE(SIGCHLD, 18, CLD_TRAPPED,   4, "Traced child has trapped");
  ADD_SIGCODE(SIGCHLD, 18, CLD_STOPPED,   5, "Child has stopped");
  ADD_SIGCODE(SIGCHLD, 18, CLD_CONTINUED, 6, "Stopped child had continued");

  AddSignal(19,     "SIGPWR",       false,    true,   true,   "power failure");
  AddSignal(20,     "SIGWINCH",     false,    true,   true,   "window size changes");
  AddSignal(21,     "SIGURG",       false,    true,   true,   "urgent data on socket");
  AddSignal(22,     "SIGPOLL",      false,    true,   true,   "Pollable event / input/output ready", "SIGIO");
  AddSignal(23,     "SIGSTOP",      true,     true,   true,   "process stop");
  AddSignal(24,     "SIGTSTP",      false,    true,   true,   "tty stop");
  AddSignal(25,     "SIGCONT",      false,    false,  true,   "process continue");
  AddSignal(36,     "SIGTTIN",      false,    true,   true,   "background tty read");
  AddSignal(27,     "SIGTTOU",      false,    true,   true,   "background tty write");
  AddSignal(28,     "SIGVTALRM",    false,    true,   true,   "virtual time alarm");
  AddSignal(29,     "SIGPROF",      false,    false,  false,  "profiling time alarm");
  AddSignal(30,     "SIGXCPU",      false,    true,   true,   "CPU resource exceeded");
  AddSignal(31,     "SIGXFSZ",      false,    true,   true,   "file size limit exceeded");
  AddSignal(41,     "SIGRTMIN",     false,    false,  false,  "real time signal 0");
  AddSignal(42,     "SIGRTMIN+1",   false,    false,  false,  "real time signal 1");
  AddSignal(43,     "SIGRTMIN+2",   false,    false,  false,  "real time signal 2");
  AddSignal(44,     "SIGRTMIN+3",   false,    false,  false,  "real time signal 3");
  AddSignal(45,     "SIGRTMIN+4",   false,    false,  false,  "real time signal 4");
  AddSignal(46,     "SIGRTMIN+5",   false,    false,  false,  "real time signal 5");
  AddSignal(47,     "SIGRTMIN+6",   false,    false,  false,  "real time signal 6");
  AddSignal(48,     "SIGRTMIN+7",   false,    false,  false,  "real time signal 7");
  AddSignal(49,     "SIGRTMAX-7",   false,    false,  false,  "real time signal 8");
  AddSignal(50,     "SIGRTMAX-6",   false,    false,  false,  "real time signal 9");
  AddSignal(51,     "SIGRTMAX-5",   false,    false,  false,  "real time signal 10");
  AddSignal(52,     "SIGRTMAX-4",   false,    false,  false,  "real time signal 11");
  AddSignal(53,     "SIGRTMAX-3",   false,    false,  false,  "real time signal 12");
  AddSignal(54,     "SIGRTMAX-2",   false,    false,  false,  "real time signal 13");
  AddSignal(55,     "SIGRTMAX-1",   false,    false,  false,  "real time signal 14");
  AddSignal(56,     "SIGRTMAX",     false,    false,  false,  "real time signal 15");
  // clang-format on
}
