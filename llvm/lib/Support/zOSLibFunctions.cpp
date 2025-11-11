//===-- zOSLibFunctions.cpp -----------------------------------------------===//
////
//// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//// See https://llvm.org/LICENSE.txt for license information.
//// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
////
////===--------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// This file defines z/OS implementations for common functions.
//
//===----------------------------------------------------------------------===//

#ifdef __MVS__
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>


char *signalName[] = {
  /*  0 */ nullptr,
  /*  1 */ "Hangup", // SIGHUP - Posix
  /*  2 */ "Interrupt", // SIGINT
  /*  3 */ "Aborted", // SIGABRT
  /*  4 */ "Illegal instruction", // SIGILL
  /*  5 */ "Polling event", // SIGPOLL
  /*  6 */ "Socket data available", // SIGURG
  /*  7 */ "Stopped (signal)", // SIGSTOP - Posix
  /*  8 */ "Floating point exception", // SIGFPE
  /*  9 */ "Killed", // SIGKILL - Posix
  /* 10 */ "Bus error", // SIGBUS
  /* 11 */ "Segmentation fault", // SIGSEGV
  /* 12 */ "Bad system call", // SIGSYS
  /* 13 */ "Broken pipe", // SIGPIPE - Posix
  /* 14 */ "Alarm clock", // SIGALRM - Posix
  /* 15 */ "Terminated", // SIGTERM
  /* 16 */ "User defined signal 1", // SIGUSR1
  /* 17 */ "User defined signal 2", // SIGUSR2
  /* 18 */ "Abend", // SIGABND
  /* 19 */ "Continued", // SIGCONT - Posix
  /* 20 */ "Child exited", // SIGCHLD - Posix
  /* 21 */ "Stopped (tty input)", // SIGTTIN - Posix
  /* 22 */ "Stopped (tty output)", // SIGTTOU - Posix
  /* 23 */ "I/O complete", // SIGIO
  /* 24 */ "Quit", // SIGQUIT - Posix
  /* 25 */ "Stopped", // SIGTSTP - Posix
  /* 26 */ "Trace/breakpoint trap", // SIGTRAP
  /* 27 */ "I/O error", // SIGIOERR
  /* 28 */ "Window changed", // SIGWINCH
  /* 29 */ "CPU time limit exceeded", // SIGXCPU
  /* 30 */ "File size limit exceeded", // SIGXFSZ
  /* 31 */ "Virtual timer expired", // SIGVTALRM
  /* 32 */ "Profiling timer expired",
  /* 33 */ "OMVS subsystem shutdown",  // SIGDANGER
  /* 34 */ "Thread stop", // SIGTHSTOP
  /* 35 */ "Thread resume", // SIGTHCONT
  /* 36 */ nullptr,
  /* 37 */ "Toggle syscall trace", // SIGTRACE
  /* 38 */ nullptr, // SIGDCE
  /* 39 */ "System dump", // SIGDUMP - Posix
};

// z/OS Unix System Services does not have strsignal() support, so the
// strsignal() function is implemented here.
char *strsignal(int sig) {
  if (sig < sizeof(signalName)/sizeof(signalName[0]) && signalName[sig])
    return signalName[sig];
  static char msg[256];
  sprintf(msg, "Unknown signal %d", sig);
  return msg;
}

// z/OS Unix System Services does not have strnlen() support, so the strnlen()
// function is implemented here.
size_t strnlen(const char *S, size_t MaxLen) {
  const char *PtrToNullChar =
      static_cast<const char *>(memchr(S, '\0', MaxLen));
  return PtrToNullChar ? PtrToNullChar - S : MaxLen;
}
#endif
