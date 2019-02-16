/* wait.c -- functions for getting wait status values.

   Copyright 2011 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.

   We use C code to extract the wait status so that we can easily be
   OS-independent.  */

#include <stdint.h>
#include <sys/wait.h>

#include "runtime.h"

extern _Bool Exited (uint32_t *w)
  __asm__ (GOSYM_PREFIX "syscall.Exited.N18_syscall.WaitStatus");

_Bool
Exited (uint32_t *w)
{
  return WIFEXITED (*w) != 0;
}

extern _Bool Signaled (uint32_t *w)
  __asm__ (GOSYM_PREFIX "syscall.Signaled.N18_syscall.WaitStatus");

_Bool
Signaled (uint32_t *w)
{
  return WIFSIGNALED (*w) != 0;
}

extern _Bool Stopped (uint32_t *w)
  __asm__ (GOSYM_PREFIX "syscall.Stopped.N18_syscall.WaitStatus");

_Bool
Stopped (uint32_t *w)
{
  return WIFSTOPPED (*w) != 0;
}

extern _Bool Continued (uint32_t *w)
  __asm__ (GOSYM_PREFIX "syscall.Continued.N18_syscall.WaitStatus");

_Bool
Continued (uint32_t *w)
{
  return WIFCONTINUED (*w) != 0;
}

extern _Bool CoreDump (uint32_t *w)
  __asm__ (GOSYM_PREFIX "syscall.CoreDump.N18_syscall.WaitStatus");

_Bool
CoreDump (uint32_t *w)
{
  return WCOREDUMP (*w) != 0;
}

extern int ExitStatus (uint32_t *w)
  __asm__ (GOSYM_PREFIX "syscall.ExitStatus.N18_syscall.WaitStatus");

int
ExitStatus (uint32_t *w)
{
  if (!WIFEXITED (*w))
    return -1;
  return WEXITSTATUS (*w);
}

extern int Signal (uint32_t *w)
  __asm__ (GOSYM_PREFIX "syscall.Signal.N18_syscall.WaitStatus");

int
Signal (uint32_t *w)
{
  if (!WIFSIGNALED (*w))
    return -1;
  return WTERMSIG (*w);
}

extern int StopSignal (uint32_t *w)
  __asm__ (GOSYM_PREFIX "syscall.StopSignal.N18_syscall.WaitStatus");

int
StopSignal (uint32_t *w)
{
  if (!WIFSTOPPED (*w))
    return -1;
  return WSTOPSIG (*w);
}

extern int TrapCause (uint32_t *w)
  __asm__ (GOSYM_PREFIX "syscall.TrapCause.N18_syscall.WaitStatus");

int
TrapCause (uint32_t *w __attribute__ ((unused)))
{
#ifndef __linux__
  return -1;
#else
  if (!WIFSTOPPED (*w) || WSTOPSIG (*w) != SIGTRAP)
    return -1;
  return *w >> 16;
#endif
}
