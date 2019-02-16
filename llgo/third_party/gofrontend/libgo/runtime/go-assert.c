/* go-assert.c -- libgo specific assertions

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stdio.h>
#include <stdlib.h>

#include "runtime.h"
#include "go-assert.h"

void
__go_assert_fail (const char *file, unsigned int lineno)
{
  /* FIXME: Eventually we should dump a stack trace here.  */
  runtime_printf ("%s:%U: libgo assertion failure\n", file, (uint64) lineno);
  abort ();
}
