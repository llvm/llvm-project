/* go-breakpoint.c -- the runtime.Breakpoint function.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <sched.h>

#include "runtime.h"

void Breakpoint (void) __asm__ (GOSYM_PREFIX "runtime.Breakpoint");

void
Breakpoint (void)
{
  __builtin_trap ();
}
