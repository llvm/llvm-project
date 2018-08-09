/* go-defer.c -- manage the defer stack.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stddef.h>

#include "runtime.h"
#include "go-alloc.h"
#include "go-panic.h"
#include "go-defer.h"

/* This function is called each time we need to defer a call.  */

void
__go_defer (_Bool *frame, void (*pfn) (void *), void *arg)
{
  G *g;
  struct __go_defer_stack *n;

  g = runtime_g ();
  n = runtime_newdefer ();
  n->__next = g->defer;
  n->__frame = frame;
  n->__panic = g->panic;
  n->__pfn = pfn;
  n->__arg = arg;
  n->__retaddr = NULL;
  n->__makefunc_can_recover = 0;
  n->__special = 0;
  g->defer = n;
}

/* This function is called when we want to undefer the stack.  */

void
__go_undefer (_Bool *frame)
{
  G *g;

  g = runtime_g ();
  while (g->defer != NULL && g->defer->__frame == frame)
    {
      struct __go_defer_stack *d;
      void (*pfn) (void *);

      d = g->defer;
      pfn = d->__pfn;
      d->__pfn = NULL;

      if (pfn != NULL)
	(*pfn) (d->__arg);

      g->defer = d->__next;

      /* This may be called by a cgo callback routine to defer the
	 call to syscall.CgocallBackDone, in which case we will not
	 have a memory context.  Don't try to free anything in that
	 case--the GC will release it later.  */
      if (runtime_m () != NULL)
	runtime_freedefer (d);

      /* Since we are executing a defer function here, we know we are
	 returning from the calling function.  If the calling
	 function, or one of its callees, paniced, then the defer
	 functions would be executed by __go_panic.  */
      *frame = 1;
    }
}

/* This function is called to record the address to which the deferred
   function returns.  This may in turn be checked by __go_can_recover.
   The frontend relies on this function returning false.  */

_Bool
__go_set_defer_retaddr (void *retaddr)
{
  G *g;

  g = runtime_g ();
  if (g->defer != NULL)
    g->defer->__retaddr = __builtin_extract_return_addr (retaddr);
  return 0;
}
