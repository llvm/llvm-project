/* go-traceback.c -- stack backtrace for Go.

   Copyright 2012 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "config.h"

#include "runtime.h"

/* Print a stack trace for the current goroutine.  */

void
runtime_traceback ()
{
  Location locbuf[100];
  int32 c;

  c = runtime_callers (1, locbuf, nelem (locbuf), false);
  runtime_printtrace (locbuf, c, true);
}

void
runtime_printtrace (Location *locbuf, int32 c, bool current)
{
  int32 i;

  for (i = 0; i < c; ++i)
    {
      if (runtime_showframe (locbuf[i].function, current))
	{
	  runtime_printf ("%S\n", locbuf[i].function);
	  runtime_printf ("\t%S:%D\n", locbuf[i].filename,
			  (int64) locbuf[i].lineno);
	}
    }
}
