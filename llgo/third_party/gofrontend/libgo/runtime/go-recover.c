/* go-recover.c -- support for the go recover function.

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "interface.h"
#include "go-panic.h"
#include "go-defer.h"

/* If the top of the defer stack can be recovered, then return it.
   Otherwise return NULL.  */

static struct __go_defer_stack *
current_defer ()
{
  G *g;
  struct __go_defer_stack *d;

  g = runtime_g ();

  d = g->defer;
  if (d == NULL)
    return NULL;

  /* The panic which would be recovered is the one on the top of the
     panic stack.  We do not want to recover it if that panic was on
     the top of the panic stack when this function was deferred.  */
  if (d->__panic == g->panic)
    return NULL;

  /* The deferred thunk will call _go_set_defer_retaddr.  If this has
     not happened, then we have not been called via defer, and we can
     not recover.  */
  if (d->__retaddr == NULL)
    return NULL;

  return d;
}

/* This is called by a thunk to see if the real function should be
   permitted to recover a panic value.  Recovering a value is
   permitted if the thunk was called directly by defer.  RETADDR is
   the return address of the function which is calling
   __go_can_recover--this is, the thunk.  */

_Bool
__go_can_recover (void *retaddr)
{
  struct __go_defer_stack *d;
  const char* ret;
  const char* dret;
  Location locs[16];
  const byte *name;
  intgo len;
  int n;
  int i;
  _Bool found_ffi_callback;

  d = current_defer ();
  if (d == NULL)
    return 0;

  ret = (const char *) __builtin_extract_return_addr (retaddr);

  dret = (const char *) d->__retaddr;
  if (ret <= dret && ret + 16 >= dret)
    return 1;

  /* On some systems, in some cases, the return address does not work
     reliably.  See http://gcc.gnu.org/PR60406.  If we are permitted
     to call recover, the call stack will look like this:
       __go_panic, __go_undefer, etc.
       thunk to call deferred function (calls __go_set_defer_retaddr)
       function that calls __go_can_recover (passing return address)
       __go_can_recover
     Calling runtime_callers will skip the thunks.  So if our caller's
     caller starts with __go, then we are permitted to call
     recover.  */

  if (runtime_callers (1, &locs[0], 2, false) < 2)
    return 0;

  name = locs[1].function.str;
  len = locs[1].function.len;

  /* Although locs[1].function is a Go string, we know it is
     NUL-terminated.  */
  if (len > 4
      && __builtin_strchr ((const char *) name, '.') == NULL
      && __builtin_strncmp ((const char *) name, "__go_", 4) == 0)
    return 1;

  /* If we are called from __go_makefunc_can_recover, then we need to
     look one level higher.  */
  if (locs[0].function.len > 0
      && __builtin_strcmp ((const char *) locs[0].function.str,
			   "__go_makefunc_can_recover") == 0)
    {
      if (runtime_callers (3, &locs[0], 1, false) < 1)
	return 0;
      name = locs[0].function.str;
      len = locs[0].function.len;
      if (len > 4
	  && __builtin_strchr ((const char *) name, '.') == NULL
	  && __builtin_strncmp ((const char *) name, "__go_", 4) == 0)
	return 1;
    }

  /* If the function calling recover was created by reflect.MakeFunc,
     then __go_makefunc_can_recover or __go_makefunc_ffi_can_recover
     will have set the __makefunc_can_recover field.  */
  if (!d->__makefunc_can_recover)
    return 0;

  /* We look up the stack, ignoring libffi functions and functions in
     the reflect package, until we find reflect.makeFuncStub or
     reflect.ffi_callback called by FFI functions.  Then we check the
     caller of that function.  */

  n = runtime_callers (2, &locs[0], sizeof locs / sizeof locs[0], false);
  found_ffi_callback = 0;
  for (i = 0; i < n; i++)
    {
      const byte *name;

      if (locs[i].function.len == 0)
	{
	  /* No function name means this caller isn't Go code.  Assume
	     that this is libffi.  */
	  continue;
	}

      /* Ignore functions in libffi.  */
      name = locs[i].function.str;
      if (__builtin_strncmp ((const char *) name, "ffi_", 4) == 0)
	continue;

      if (found_ffi_callback)
	break;

      if (__builtin_strcmp ((const char *) name, "reflect.ffi_callback") == 0)
	{
	  found_ffi_callback = 1;
	  continue;
	}

      if (__builtin_strcmp ((const char *) name, "reflect.makeFuncStub") == 0)
	{
	  i++;
	  break;
	}

      /* Ignore other functions in the reflect package.  */
      if (__builtin_strncmp ((const char *) name, "reflect.", 8) == 0)
	continue;

      /* We should now be looking at the real caller.  */
      break;
    }

  if (i < n && locs[i].function.len > 0)
    {
      name = locs[i].function.str;
      if (__builtin_strncmp ((const char *) name, "__go_", 4) == 0)
	return 1;
    }

  return 0;
}

/* This function is called when code is about to enter a function
   created by reflect.MakeFunc.  It is called by the function stub
   used by MakeFunc.  If the stub is permitted to call recover, then a
   real MakeFunc function is permitted to call recover.  */

void
__go_makefunc_can_recover (void *retaddr)
{
  struct __go_defer_stack *d;

  d = current_defer ();
  if (d == NULL)
    return;

  /* If we are already in a call stack of MakeFunc functions, there is
     nothing we can usefully check here.  */
  if (d->__makefunc_can_recover)
    return;

  if (__go_can_recover (retaddr))
    d->__makefunc_can_recover = 1;
}

/* This function is called when code is about to enter a function
   created by the libffi version of reflect.MakeFunc.  This function
   is passed the names of the callers of the libffi code that called
   the stub.  It uses to decide whether it is permitted to call
   recover, and sets d->__makefunc_can_recover so that __go_recover
   can make the same decision.  */

void
__go_makefunc_ffi_can_recover (struct Location *loc, int n)
{
  struct __go_defer_stack *d;
  const byte *name;
  intgo len;

  d = current_defer ();
  if (d == NULL)
    return;

  /* If we are already in a call stack of MakeFunc functions, there is
     nothing we can usefully check here.  */
  if (d->__makefunc_can_recover)
    return;

  /* LOC points to the caller of our caller.  That will be a thunk.
     If its caller was a runtime function, then it was called directly
     by defer.  */

  if (n < 2)
    return;

  name = (loc + 1)->function.str;
  len = (loc + 1)->function.len;
  if (len > 4
      && __builtin_strchr ((const char *) name, '.') == NULL
      && __builtin_strncmp ((const char *) name, "__go_", 4) == 0)
    d->__makefunc_can_recover = 1;
}

/* This function is called when code is about to exit a function
   created by reflect.MakeFunc.  It is called by the function stub
   used by MakeFunc.  It clears the __makefunc_can_recover field.
   It's OK to always clear this field, because __go_can_recover will
   only be called by a stub created for a function that calls recover.
   That stub will not call a function created by reflect.MakeFunc, so
   by the time we get here any caller higher up on the call stack no
   longer needs the information.  */

void
__go_makefunc_returning (void)
{
  struct __go_defer_stack *d;

  d = runtime_g ()->defer;
  if (d != NULL)
    d->__makefunc_can_recover = 0;
}

/* This is only called when it is valid for the caller to recover the
   value on top of the panic stack, if there is one.  */

struct __go_empty_interface
__go_recover ()
{
  G *g;
  struct __go_panic_stack *p;

  g = runtime_g ();

  if (g->panic == NULL || g->panic->__was_recovered)
    {
      struct __go_empty_interface ret;

      ret.__type_descriptor = NULL;
      ret.__object = NULL;
      return ret;
    }
  p = g->panic;
  p->__was_recovered = 1;
  return p->__arg;
}
