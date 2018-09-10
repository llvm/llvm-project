/* go-matherr.c -- a Go version of the matherr function.

   Copyright 2012 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

/* The gccgo version of the math library calls libc functions.  On
   some systems, such as Solaris, those functions will call matherr on
   exceptional conditions.  This is a version of matherr appropriate
   for Go, one which returns the values that the Go math library
   expects.  This is fine for pure Go programs.  For mixed Go and C
   programs this will be problematic if the C programs themselves use
   matherr.  Normally the C version of matherr will override this, and
   the Go code will just have to cope.  If this turns out to be too
   problematic we can change to run pure Go code in the math library
   on systems that use matherr.  */

#include <math.h>
#include <stdint.h>

#include "config.h"

#if defined(HAVE_MATHERR) && defined(HAVE_STRUCT_EXCEPTION)

#define PI 3.14159265358979323846264338327950288419716939937510582097494459

int
matherr (struct exception* e)
{
  const char *n;

  if (e->type != DOMAIN)
    return 0;

  n = e->name;
  if (__builtin_strcmp (n, "acos") == 0
      || __builtin_strcmp (n, "asin") == 0)
    e->retval = __builtin_nan ("");
  else if (__builtin_strcmp (n, "atan2") == 0)
    {
      if (e->arg1 == 0 && e->arg2 == 0)
	{
	  double nz;

	  nz = -0.0;
	  if (__builtin_memcmp (&e->arg2, &nz, sizeof (double)) != 0)
	    e->retval = e->arg1;
	  else
	    e->retval = copysign (PI, e->arg1);
	}
      else
	return 0;
    }
  else if (__builtin_strcmp (n, "log") == 0
	   || __builtin_strcmp (n, "log10") == 0)
    e->retval = __builtin_nan ("");
  else if (__builtin_strcmp (n, "pow") == 0)
    {
      if (e->arg1 < 0)
	e->retval = __builtin_nan ("");
      else if (e->arg1 == 0 && e->arg2 == 0)
	e->retval = 1.0;
      else if (e->arg1 == 0 && e->arg2 < 0)
	{
	  double i;

	  if (modf (e->arg2, &i) == 0 && ((int64_t) i & 1) == 1)
	    e->retval = copysign (__builtin_inf (), e->arg1);
	  else
	    e->retval = __builtin_inf ();
	}
      else
	return 0;
    }
  else if (__builtin_strcmp (n, "sqrt") == 0)
    {
      if (e->arg1 < 0)
	e->retval = __builtin_nan ("");
      else
	return 0;
    }
  else
    return 0;

  return 1;
}

#endif
