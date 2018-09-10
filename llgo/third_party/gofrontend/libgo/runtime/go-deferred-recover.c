/* go-deferred-recover.c -- support for a deferred recover function.

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stddef.h>

#include "runtime.h"
#include "go-panic.h"
#include "go-defer.h"

/* This is called when a call to recover is deferred.  That is,
   something like
     defer recover()

   We need to handle this specially.  In 6g/8g, the recover function
   looks up the stack frame.  In particular, that means that a
   deferred recover will not recover a panic thrown in the same
   function that defers the recover.  It will only recover a panic
   thrown in a function that defers the deferred call to recover.

   In other words:

   func f1() {
	defer recover()	// does not stop panic
	panic(0)
   }

   func f2() {
	defer func() {
		defer recover()	// stops panic(0)
	}()
	panic(0)
   }

   func f3() {
	defer func() {
		defer recover()	// does not stop panic
		panic(0)
	}()
	panic(1)
   }

   func f4() {
	defer func() {
		defer func() {
			defer recover()	// stops panic(0)
		}()
		panic(0)
	}()
	panic(1)
   }

   The interesting case here is f3.  As can be seen from f2, the
   deferred recover could pick up panic(1).  However, this does not
   happen because it is blocked by the panic(0).

   When a function calls recover, then when we invoke it we pass a
   hidden parameter indicating whether it should recover something.
   This parameter is set based on whether the function is being
   invoked directly from defer.  The parameter winds up determining
   whether __go_recover or __go_deferred_recover is called at all.

   In the case of a deferred recover, the hidden parameter which
   controls the call is actually the one set up for the function which
   runs the defer recover() statement.  That is the right thing in all
   the cases above except for f3.  In f3 the function is permitted to
   call recover, but the deferred recover call is not.  We address
   that here by checking for that specific case before calling
   recover.  If this function was deferred when there is already a
   panic on the panic stack, then we can only recover that panic, not
   any other.

   Note that we can get away with using a special function here
   because you are not permitted to take the address of a predeclared
   function like recover.  */

struct __go_empty_interface
__go_deferred_recover ()
{
  G *g;

  g = runtime_g ();
  if (g->defer == NULL || g->defer->__panic != g->panic)
    {
      struct __go_empty_interface ret;

      ret.__type_descriptor = NULL;
      ret.__object = NULL;
      return ret;
    }
  return __go_recover ();
}
