/* signame.c -- get the name of a signal

   Copyright 2012 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <string.h>

#include "runtime.h"
#include "arch.h"
#include "malloc.h"

String Signame (intgo sig) __asm__ (GOSYM_PREFIX "syscall.Signame");

String
Signame (intgo sig)
{
  const char* s = NULL;
  char buf[100];
  size_t len;
  byte *data;
  String ret;

#if defined(HAVE_STRSIGNAL)
  s = strsignal (sig);
#endif

  if (s == NULL)
    {
      snprintf(buf, sizeof buf, "signal %ld", (long) sig);
      s = buf;
    }
  len = __builtin_strlen (s);
  data = runtime_mallocgc (len, 0, FlagNoScan);
  __builtin_memcpy (data, s, len);
  ret.str = data;
  ret.len = len;
  return ret;
}
