/* syslog_c.c -- call syslog for Go.

   Copyright 2011 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <syslog.h>

#include "runtime.h"

/* We need to use a C function to call the syslog function, because we
   can't represent a C varargs function in Go.  */

void syslog_c(intgo, const char*)
  __asm__ (GOSYM_PREFIX "log_syslog.syslog_c");

void
syslog_c (intgo priority, const char *msg)
{
  syslog (priority, "%s", msg);
}
