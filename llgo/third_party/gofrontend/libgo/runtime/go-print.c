/* go-print.c -- support for the go print statement.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "runtime.h"
#include "array.h"
#include "go-panic.h"
#include "interface.h"

/* This implements the various little functions which are called by
   the predeclared functions print/println/panic/panicln.  */

void
__go_print_empty_interface (struct __go_empty_interface e)
{
  runtime_printf ("(%p,%p)", e.__type_descriptor, e.__object);
}

void
__go_print_interface (struct __go_interface i)
{
  runtime_printf ("(%p,%p)", i.__methods, i.__object);
}

void
__go_print_slice (struct __go_open_array val)
{
  runtime_printf ("[%d/%d]", val.__count, val.__capacity);
  runtime_printpointer (val.__values);
}
