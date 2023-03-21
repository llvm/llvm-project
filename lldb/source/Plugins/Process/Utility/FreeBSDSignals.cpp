//===-- FreeBSDSignals.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FreeBSDSignals.h"

using namespace lldb_private;

FreeBSDSignals::FreeBSDSignals() : UnixSignals() { Reset(); }

void FreeBSDSignals::Reset() {
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
  AddSignalCode(8, 1 /*FPE_INTOVF*/, "integer overflow");
  AddSignalCode(8, 2 /*FPE_INTDIV*/, "integer divide by zero");
  AddSignalCode(8, 3 /*FPE_FLTDIV*/, "floating point divide by zero");
  AddSignalCode(8, 4 /*FPE_FLTOVF*/, "floating point overflow");
  AddSignalCode(8, 5 /*FPE_FLTUND*/, "floating point underflow");
  AddSignalCode(8, 6 /*FPE_FLTRES*/, "floating point inexact result");
  AddSignalCode(8, 7 /*FPE_FLTINV*/, "invalid floating point operation");
  AddSignalCode(8, 8 /*FPE_FLTSUB*/, "subscript out of range");
  AddSignalCode(8, 9 /*FPE_FLTIDO*/, "input denormal operation");

  // SIGBUS
  AddSignalCode(10, 1   /*BUS_ADRALN*/, "invalid address alignment");
  AddSignalCode(10, 2   /*BUS_ADRERR*/, "nonexistent physical address");
  AddSignalCode(10, 3   /*BUS_OBJERR*/, "object-specific hardware error");
  AddSignalCode(10, 100 /*BUS_OOMERR*/, "no memory");

  // SIGSEGV
  AddSignalCode(11, 1   /*SEGV_MAPERR*/, "address not mapped to object",
                SignalCodePrintOption::Address);
  AddSignalCode(11, 2   /*SEGV_ACCERR*/, "invalid permissions for mapped object",
                SignalCodePrintOption::Address);
  AddSignalCode(11, 100 /*SEGV_PKUERR*/, "PKU violation",
                SignalCodePrintOption::Address);

  //        SIGNO NAME           SUPPRESS STOP   NOTIFY DESCRIPTION
  //        ===== ============== ======== ====== ====== ========================
  AddSignal(32,   "SIGTHR",      false,   false, false, "thread interrupt");
  AddSignal(33,   "SIGLIBRT",    false,   false, false, "reserved by real-time library");
  AddSignal(65,   "SIGRTMIN",    false,   false, false, "real time signal 0");
  AddSignal(66,   "SIGRTMIN+1",  false,   false, false, "real time signal 1");
  AddSignal(67,   "SIGRTMIN+2",  false,   false, false, "real time signal 2");
  AddSignal(68,   "SIGRTMIN+3",  false,   false, false, "real time signal 3");
  AddSignal(69,   "SIGRTMIN+4",  false,   false, false, "real time signal 4");
  AddSignal(70,   "SIGRTMIN+5",  false,   false, false, "real time signal 5");
  AddSignal(71,   "SIGRTMIN+6",  false,   false, false, "real time signal 6");
  AddSignal(72,   "SIGRTMIN+7",  false,   false, false, "real time signal 7");
  AddSignal(73,   "SIGRTMIN+8",  false,   false, false, "real time signal 8");
  AddSignal(74,   "SIGRTMIN+9",  false,   false, false, "real time signal 9");
  AddSignal(75,   "SIGRTMIN+10", false,   false, false, "real time signal 10");
  AddSignal(76,   "SIGRTMIN+11", false,   false, false, "real time signal 11");
  AddSignal(77,   "SIGRTMIN+12", false,   false, false, "real time signal 12");
  AddSignal(78,   "SIGRTMIN+13", false,   false, false, "real time signal 13");
  AddSignal(79,   "SIGRTMIN+14", false,   false, false, "real time signal 14");
  AddSignal(80,   "SIGRTMIN+15", false,   false, false, "real time signal 15");
  AddSignal(81,   "SIGRTMIN+16", false,   false, false, "real time signal 16");
  AddSignal(82,   "SIGRTMIN+17", false,   false, false, "real time signal 17");
  AddSignal(83,   "SIGRTMIN+18", false,   false, false, "real time signal 18");
  AddSignal(84,   "SIGRTMIN+19", false,   false, false, "real time signal 19");
  AddSignal(85,   "SIGRTMIN+20", false,   false, false, "real time signal 20");
  AddSignal(86,   "SIGRTMIN+21", false,   false, false, "real time signal 21");
  AddSignal(87,   "SIGRTMIN+22", false,   false, false, "real time signal 22");
  AddSignal(88,   "SIGRTMIN+23", false,   false, false, "real time signal 23");
  AddSignal(89,   "SIGRTMIN+24", false,   false, false, "real time signal 24");
  AddSignal(90,   "SIGRTMIN+25", false,   false, false, "real time signal 25");
  AddSignal(91,   "SIGRTMIN+26", false,   false, false, "real time signal 26");
  AddSignal(92,   "SIGRTMIN+27", false,   false, false, "real time signal 27");
  AddSignal(93,   "SIGRTMIN+28", false,   false, false, "real time signal 28");
  AddSignal(94,   "SIGRTMIN+29", false,   false, false, "real time signal 29");
  AddSignal(95,   "SIGRTMIN+30", false,   false, false, "real time signal 30");
  AddSignal(96,   "SIGRTMAX-30", false,   false, false, "real time signal 31");
  AddSignal(97,   "SIGRTMAX-29", false,   false, false, "real time signal 32");
  AddSignal(98,   "SIGRTMAX-28", false,   false, false, "real time signal 33");
  AddSignal(99,   "SIGRTMAX-27", false,   false, false, "real time signal 34");
  AddSignal(100,  "SIGRTMAX-26", false,   false, false, "real time signal 35");
  AddSignal(101,  "SIGRTMAX-25", false,   false, false, "real time signal 36");
  AddSignal(102,  "SIGRTMAX-24", false,   false, false, "real time signal 37");
  AddSignal(103,  "SIGRTMAX-23", false,   false, false, "real time signal 38");
  AddSignal(104,  "SIGRTMAX-22", false,   false, false, "real time signal 39");
  AddSignal(105,  "SIGRTMAX-21", false,   false, false, "real time signal 40");
  AddSignal(106,  "SIGRTMAX-20", false,   false, false, "real time signal 41");
  AddSignal(107,  "SIGRTMAX-19", false,   false, false, "real time signal 42");
  AddSignal(108,  "SIGRTMAX-18", false,   false, false, "real time signal 43");
  AddSignal(109,  "SIGRTMAX-17", false,   false, false, "real time signal 44");
  AddSignal(110,  "SIGRTMAX-16", false,   false, false, "real time signal 45");
  AddSignal(111,  "SIGRTMAX-15", false,   false, false, "real time signal 46");
  AddSignal(112,  "SIGRTMAX-14", false,   false, false, "real time signal 47");
  AddSignal(113,  "SIGRTMAX-13", false,   false, false, "real time signal 48");
  AddSignal(114,  "SIGRTMAX-12", false,   false, false, "real time signal 49");
  AddSignal(115,  "SIGRTMAX-11", false,   false, false, "real time signal 50");
  AddSignal(116,  "SIGRTMAX-10", false,   false, false, "real time signal 51");
  AddSignal(117,  "SIGRTMAX-9",  false,   false, false, "real time signal 52");
  AddSignal(118,  "SIGRTMAX-8",  false,   false, false, "real time signal 53");
  AddSignal(119,  "SIGRTMAX-7",  false,   false, false, "real time signal 54");
  AddSignal(120,  "SIGRTMAX-6",  false,   false, false, "real time signal 55");
  AddSignal(121,  "SIGRTMAX-5",  false,   false, false, "real time signal 56");
  AddSignal(122,  "SIGRTMAX-4",  false,   false, false, "real time signal 57");
  AddSignal(123,  "SIGRTMAX-3",  false,   false, false, "real time signal 58");
  AddSignal(124,  "SIGRTMAX-2",  false,   false, false, "real time signal 59");
  AddSignal(125,  "SIGRTMAX-1",  false,   false, false, "real time signal 60");
  AddSignal(126,  "SIGRTMAX",    false,   false, false, "real time signal 61");
  // clang-format on
}
