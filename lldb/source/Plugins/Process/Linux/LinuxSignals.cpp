//===-- LinuxSignals.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <signal.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "LinuxSignals.h"

LinuxSignals::LinuxSignals()
    : UnixSignals()
{
    Reset();
}

void
LinuxSignals::Reset()
{
    m_signals.clear();

#define ADDSIGNAL(S, SUPPRESS, STOP, NOTIFY, DESCRIPTION) \
    AddSignal(SIG ## S, "SIG" #S, #S, SUPPRESS, STOP, NOTIFY, DESCRIPTION)

    ADDSIGNAL(HUP,    false,  true,  true, "hangup");
    ADDSIGNAL(INT,    true,   true,  true, "interrupt");
    ADDSIGNAL(QUIT,   false,  true,  true, "quit");
    ADDSIGNAL(ILL,    false,  true,  true, "illegal instruction");
    ADDSIGNAL(TRAP,   true,   true,  true, "trace trap (not reset when caught)");
    ADDSIGNAL(ABRT,   false,  true,  true, "abort");
    ADDSIGNAL(IOT,    false,  true,  true, "abort");
    ADDSIGNAL(BUS,    false,  true,  true, "bus error");
    ADDSIGNAL(FPE,    false,  true,  true, "floating point exception");
    ADDSIGNAL(KILL,   false,  true,  true, "kill");
    ADDSIGNAL(USR1,   false,  true,  true, "user defined signal 1");
    ADDSIGNAL(SEGV,   false,  true,  true, "segmentation violation");
    ADDSIGNAL(USR2,   false,  true,  true, "user defined signal 2");
    ADDSIGNAL(PIPE,   false,  true,  true, "write to pipe with reading end closed");
    ADDSIGNAL(ALRM,   false,  false, true, "alarm");
    ADDSIGNAL(TERM,   false,  true,  true, "termination requested");
    ADDSIGNAL(STKFLT, false,  true,  true, "stack fault");
    ADDSIGNAL(CHLD,   false,  false, true, "child process exit");
    ADDSIGNAL(CONT,   false,  true,  true, "process continue");
    ADDSIGNAL(STOP,   false,  true,  true, "process stop");
    ADDSIGNAL(TSTP,   false,  true,  true, "tty stop");
    ADDSIGNAL(TTIN,   false,  true,  true, "background tty read");
    ADDSIGNAL(TTOU,   false,  true,  true, "background tty write");
    ADDSIGNAL(URG,    false,  true,  true, "urgent data on socket");
    ADDSIGNAL(XCPU,   false,  true,  true, "CPU resource exceeded");
    ADDSIGNAL(XFSZ,   false,  true,  true, "file size limit exceeded");
    ADDSIGNAL(VTALRM, false,  true,  true, "virtual alarm");
    ADDSIGNAL(PROF,   false,  true,  true, "profiling alarm");
    ADDSIGNAL(WINCH,  false,  true,  true, "window size change");
    ADDSIGNAL(POLL,   false,  true,  true, "pollable event");
    ADDSIGNAL(IO,     false,  true,  true, "input/output ready");
    ADDSIGNAL(PWR,    false,  true,  true, "power failure");
    ADDSIGNAL(SYS,    false,  true,  true, "invalid system call");

#undef ADDSIGNAL
}
