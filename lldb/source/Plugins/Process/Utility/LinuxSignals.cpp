//===-- LinuxSignals.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LinuxSignals.h"

// mips-linux debugging is not supported and mips uses different numbers for
// some signals (e.g. SIGBUS) on linux, so we skip the static checks below. The
// definitions here can be used for debugging non-mips targets on a mips-hosted
// lldb.
#if defined(__linux__) && !defined(__mips__)
#include <csignal>

#ifndef SEGV_BNDERR
#define SEGV_BNDERR 3
#endif
#ifndef SEGV_MTEAERR
#define SEGV_MTEAERR 8
#endif
#ifndef SEGV_MTESERR
#define SEGV_MTESERR 9
#endif
#ifndef SEGV_CPERR
#define SEGV_CPERR 10
#endif
#ifndef SI_QUEUE
#define SI_QUEUE -1
#endif
#ifndef SI_TIMER
#define SI_TIMER -2
#endif
#ifndef SI_MESGQ
#define SI_MESGQ -3
#endif
#ifndef SI_ASYNCIO
#define SI_ASYNCIO -4
#endif
#ifndef SI_SIGIO
#define SI_SIGIO -5
#endif
#ifndef SI_TKILL
#define SI_TKILL -6
#endif
#ifndef SI_DETHREAD
#define SI_DETHREAD -7
#endif
#ifndef SI_ASYNCNL
#define SI_ASYNCNL -60
#endif

#define ADD_SIGCODE(signal_name, signal_value, code_name, code_value, ...)     \
  static_assert(signal_name == signal_value,                                   \
                "Value mismatch for signal number " #signal_name);             \
  static_assert(code_name == code_value,                                       \
                "Value mismatch for signal code " #code_name);                 \
  AddSignalCode(signal_value, code_value, __VA_ARGS__)
#else
#define ADD_SIGCODE(signal_name, signal_value, code_name, code_value, ...)     \
  AddSignalCode(signal_value, code_value, __VA_ARGS__)
#endif /* if defined(__linux__) && !defined(__mips__) */
// See siginfo.h in the Linux Kernel, these codes can be sent for any signal.
#define ADD_LINUX_SIGNAL(signo, name, ...)                                     \
  AddSignal(signo, name, __VA_ARGS__);                                         \
  ADD_SIGCODE(signo, signo, SI_USER, 0, "sent by kill, sigsend or raise",      \
              SignalCodePrintOption::Sender);                                  \
  ADD_SIGCODE(signo, signo, SI_KERNEL, 0x80, "sent by kernel (SI_KERNEL)",     \
              SignalCodePrintOption::Sender);                                  \
  ADD_SIGCODE(signo, signo, SI_QUEUE, -1, "sent by sigqueue",                  \
              SignalCodePrintOption::Sender);                                  \
  ADD_SIGCODE(signo, signo, SI_TIMER, -2, "sent by timer expiration",          \
              SignalCodePrintOption::Sender);                                  \
  ADD_SIGCODE(signo, signo, SI_MESGQ, -3,                                      \
              "sent by real time mesq state change",                           \
              SignalCodePrintOption::Sender);                                  \
  ADD_SIGCODE(signo, signo, SI_ASYNCIO, -4, "sent by AIO completion",          \
              SignalCodePrintOption::Sender);                                  \
  ADD_SIGCODE(signo, signo, SI_SIGIO, -5, "sent by queued SIGIO",              \
              SignalCodePrintOption::Sender);                                  \
  ADD_SIGCODE(signo, signo, SI_TKILL, -6, "sent by tkill system call",         \
              SignalCodePrintOption::Sender);                                  \
  ADD_SIGCODE(signo, signo, SI_DETHREAD, -7,                                   \
              "sent by execve() killing subsidiary threads",                   \
              SignalCodePrintOption::Sender);                                  \
  ADD_SIGCODE(signo, signo, SI_ASYNCNL, -60,                                   \
              "sent by glibc async name lookup completion",                    \
              SignalCodePrintOption::Sender);

using namespace lldb_private;

LinuxSignals::LinuxSignals() : UnixSignals() { Reset(); }

void LinuxSignals::Reset() {
  m_signals.clear();
  // clang-format off
  //               SIGNO   NAME            SUPPRESS  STOP    NOTIFY  DESCRIPTION
  //               ======  ==============  ========  ======  ======  ===================================================
  ADD_LINUX_SIGNAL(1,      "SIGHUP",       false,    true,   true,   "hangup");
  ADD_LINUX_SIGNAL(2,      "SIGINT",       true,     true,   true,   "interrupt");
  ADD_LINUX_SIGNAL(3,      "SIGQUIT",      false,    true,   true,   "quit");

  ADD_LINUX_SIGNAL(4,      "SIGILL",       false,    true,   true,   "illegal instruction");
  ADD_SIGCODE(SIGILL, 4, ILL_ILLOPC, 1, "illegal opcode");
  ADD_SIGCODE(SIGILL, 4, ILL_ILLOPN, 2, "illegal operand");
  ADD_SIGCODE(SIGILL, 4, ILL_ILLADR, 3, "illegal addressing mode");
  ADD_SIGCODE(SIGILL, 4, ILL_ILLTRP, 4, "illegal trap");
  ADD_SIGCODE(SIGILL, 4, ILL_PRVOPC, 5, "privileged opcode");
  ADD_SIGCODE(SIGILL, 4, ILL_PRVREG, 6, "privileged register");
  ADD_SIGCODE(SIGILL, 4, ILL_COPROC, 7, "coprocessor error");
  ADD_SIGCODE(SIGILL, 4, ILL_BADSTK, 8, "internal stack error");

  ADD_LINUX_SIGNAL(5,      "SIGTRAP",      true,     true,   true,   "trace trap (not reset when caught)");
  ADD_LINUX_SIGNAL(6,      "SIGABRT",      false,    true,   true,   "abort()/IOT trap", "SIGIOT");

  ADD_LINUX_SIGNAL(7,      "SIGBUS",       false,    true,   true,   "bus error");
  ADD_SIGCODE(SIGBUS, 7, BUS_ADRALN, 1, "illegal alignment");
  ADD_SIGCODE(SIGBUS, 7, BUS_ADRERR, 2, "illegal address");
  ADD_SIGCODE(SIGBUS, 7, BUS_OBJERR, 3, "hardware error");

  ADD_LINUX_SIGNAL(8,      "SIGFPE",       false,    true,   true,   "floating point exception");
  ADD_SIGCODE(SIGFPE, 8, FPE_INTDIV, 1, "integer divide by zero");
  ADD_SIGCODE(SIGFPE, 8, FPE_INTOVF, 2, "integer overflow");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTDIV, 3, "floating point divide by zero");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTOVF, 4, "floating point overflow");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTUND, 5, "floating point underflow");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTRES, 6, "floating point inexact result");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTINV, 7, "floating point invalid operation");
  ADD_SIGCODE(SIGFPE, 8, FPE_FLTSUB, 8, "subscript out of range");

  ADD_LINUX_SIGNAL(9,      "SIGKILL",      false,    true,   true,   "kill");
  ADD_LINUX_SIGNAL(10,     "SIGUSR1",      false,    true,   true,   "user defined signal 1");

  ADD_LINUX_SIGNAL(11,     "SIGSEGV",      false,    true,   true,   "segmentation violation");
  ADD_SIGCODE(SIGSEGV, 11, SEGV_MAPERR,  1, "address not mapped to object", SignalCodePrintOption::Address);
  ADD_SIGCODE(SIGSEGV, 11, SEGV_ACCERR,  2, "invalid permissions for mapped object", SignalCodePrintOption::Address);
  ADD_SIGCODE(SIGSEGV, 11, SEGV_BNDERR,  3, "failed address bounds checks", SignalCodePrintOption::Bounds);
  ADD_SIGCODE(SIGSEGV, 11, SEGV_MTEAERR, 8, "async tag check fault");
  ADD_SIGCODE(SIGSEGV, 11, SEGV_MTESERR, 9, "sync tag check fault", SignalCodePrintOption::Address);
  ADD_SIGCODE(SIGSEGV, 11, SEGV_CPERR,  10, "control protection fault");
  // Some platforms will occasionally send nonstandard spurious SI_KERNEL
  // codes. One way to get this is via unaligned SIMD loads. Treat it as invalid address.
  ADD_SIGCODE(SIGSEGV, 11, SI_KERNEL, 0x80, "invalid address", SignalCodePrintOption::Address);

  ADD_LINUX_SIGNAL(12,     "SIGUSR2",      false,    true,   true,   "user defined signal 2");
  ADD_LINUX_SIGNAL(13,     "SIGPIPE",      false,    true,   true,   "write to pipe with reading end closed");
  ADD_LINUX_SIGNAL(14,     "SIGALRM",      false,    false,  false,  "alarm");
  ADD_LINUX_SIGNAL(15,     "SIGTERM",      false,    true,   true,   "termination requested");
  ADD_LINUX_SIGNAL(16,     "SIGSTKFLT",    false,    true,   true,   "stack fault");
  ADD_LINUX_SIGNAL(17,     "SIGCHLD",      false,    false,  true,   "child status has changed", "SIGCLD");
  ADD_LINUX_SIGNAL(18,     "SIGCONT",      false,    false,  true,   "process continue");
  ADD_LINUX_SIGNAL(19,     "SIGSTOP",      true,     true,   true,   "process stop");
  ADD_LINUX_SIGNAL(20,     "SIGTSTP",      false,    true,   true,   "tty stop");
  ADD_LINUX_SIGNAL(21,     "SIGTTIN",      false,    true,   true,   "background tty read");
  ADD_LINUX_SIGNAL(22,     "SIGTTOU",      false,    true,   true,   "background tty write");
  ADD_LINUX_SIGNAL(23,     "SIGURG",       false,    true,   true,   "urgent data on socket");
  ADD_LINUX_SIGNAL(24,     "SIGXCPU",      false,    true,   true,   "CPU resource exceeded");
  ADD_LINUX_SIGNAL(25,     "SIGXFSZ",      false,    true,   true,   "file size limit exceeded");
  ADD_LINUX_SIGNAL(26,     "SIGVTALRM",    false,    true,   true,   "virtual time alarm");
  ADD_LINUX_SIGNAL(27,     "SIGPROF",      false,    false,  false,  "profiling time alarm");
  ADD_LINUX_SIGNAL(28,     "SIGWINCH",     false,    false,   false,   "window size changes");
  ADD_LINUX_SIGNAL(29,     "SIGIO",        false,    true,   true,   "input/output ready/Pollable event", "SIGPOLL");
  ADD_LINUX_SIGNAL(30,     "SIGPWR",       false,    true,   true,   "power failure");
  ADD_LINUX_SIGNAL(31,     "SIGSYS",       false,    true,   true,   "invalid system call");
  ADD_LINUX_SIGNAL(32,     "SIG32",        false,    false,  false,  "threading library internal signal 1");
  ADD_LINUX_SIGNAL(33,     "SIG33",        false,    false,  false,  "threading library internal signal 2");
  ADD_LINUX_SIGNAL(34,     "SIGRTMIN",     false,    false,  false,  "real time signal 0");
  ADD_LINUX_SIGNAL(35,     "SIGRTMIN+1",   false,    false,  false,  "real time signal 1");
  ADD_LINUX_SIGNAL(36,     "SIGRTMIN+2",   false,    false,  false,  "real time signal 2");
  ADD_LINUX_SIGNAL(37,     "SIGRTMIN+3",   false,    false,  false,  "real time signal 3");
  ADD_LINUX_SIGNAL(38,     "SIGRTMIN+4",   false,    false,  false,  "real time signal 4");
  ADD_LINUX_SIGNAL(39,     "SIGRTMIN+5",   false,    false,  false,  "real time signal 5");
  ADD_LINUX_SIGNAL(40,     "SIGRTMIN+6",   false,    false,  false,  "real time signal 6");
  ADD_LINUX_SIGNAL(41,     "SIGRTMIN+7",   false,    false,  false,  "real time signal 7");
  ADD_LINUX_SIGNAL(42,     "SIGRTMIN+8",   false,    false,  false,  "real time signal 8");
  ADD_LINUX_SIGNAL(43,     "SIGRTMIN+9",   false,    false,  false,  "real time signal 9");
  ADD_LINUX_SIGNAL(44,     "SIGRTMIN+10",  false,    false,  false,  "real time signal 10");
  ADD_LINUX_SIGNAL(45,     "SIGRTMIN+11",  false,    false,  false,  "real time signal 11");
  ADD_LINUX_SIGNAL(46,     "SIGRTMIN+12",  false,    false,  false,  "real time signal 12");
  ADD_LINUX_SIGNAL(47,     "SIGRTMIN+13",  false,    false,  false,  "real time signal 13");
  ADD_LINUX_SIGNAL(48,     "SIGRTMIN+14",  false,    false,  false,  "real time signal 14");
  ADD_LINUX_SIGNAL(49,     "SIGRTMIN+15",  false,    false,  false,  "real time signal 15");
  ADD_LINUX_SIGNAL(50,     "SIGRTMAX-14",  false,    false,  false,  "real time signal 16"); // switching to SIGRTMAX-xxx to match "kill -l" output
  ADD_LINUX_SIGNAL(51,     "SIGRTMAX-13",  false,    false,  false,  "real time signal 17");
  ADD_LINUX_SIGNAL(52,     "SIGRTMAX-12",  false,    false,  false,  "real time signal 18");
  ADD_LINUX_SIGNAL(53,     "SIGRTMAX-11",  false,    false,  false,  "real time signal 19");
  ADD_LINUX_SIGNAL(54,     "SIGRTMAX-10",  false,    false,  false,  "real time signal 20");
  ADD_LINUX_SIGNAL(55,     "SIGRTMAX-9",   false,    false,  false,  "real time signal 21");
  ADD_LINUX_SIGNAL(56,     "SIGRTMAX-8",   false,    false,  false,  "real time signal 22");
  ADD_LINUX_SIGNAL(57,     "SIGRTMAX-7",   false,    false,  false,  "real time signal 23");
  ADD_LINUX_SIGNAL(58,     "SIGRTMAX-6",   false,    false,  false,  "real time signal 24");
  ADD_LINUX_SIGNAL(59,     "SIGRTMAX-5",   false,    false,  false,  "real time signal 25");
  ADD_LINUX_SIGNAL(60,     "SIGRTMAX-4",   false,    false,  false,  "real time signal 26");
  ADD_LINUX_SIGNAL(61,     "SIGRTMAX-3",   false,    false,  false,  "real time signal 27");
  ADD_LINUX_SIGNAL(62,     "SIGRTMAX-2",   false,    false,  false,  "real time signal 28");
  ADD_LINUX_SIGNAL(63,     "SIGRTMAX-1",   false,    false,  false,  "real time signal 29");
  ADD_LINUX_SIGNAL(64,     "SIGRTMAX",     false,    false,  false,  "real time signal 30");
  // clang-format on
}
