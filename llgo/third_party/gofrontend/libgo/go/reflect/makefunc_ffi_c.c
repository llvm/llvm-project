// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "go-type.h"
#include "go-panic.h"

#ifdef USE_LIBFFI

#include "go-ffi.h"

#if FFI_GO_CLOSURES
#define USE_LIBFFI_CLOSURES
#endif

#endif /* defined(USE_LIBFFI) */

/* Declare C functions with the names used to call from Go.  */

void makeFuncFFI(const struct __go_func_type *ftyp, void *impl)
  __asm__ (GOSYM_PREFIX "reflect.makeFuncFFI");

#ifdef USE_LIBFFI_CLOSURES

/* The function that we pass to ffi_prep_closure_loc.  This calls the Go
   function ffiCall with the pointer to the arguments, the results area,
   and the closure structure.  */

void FFICallbackGo(void *result, void **args, ffi_go_closure *closure)
  __asm__ (GOSYM_PREFIX "reflect.FFICallbackGo");

static void ffi_callback (ffi_cif *, void *, void **, void *)
  __asm__ ("reflect.ffi_callback");

static void
ffi_callback (ffi_cif* cif __attribute__ ((unused)), void *results,
	      void **args, void *closure)
{
  Location locs[8];
  int n;
  int i;

  /* This function is called from some series of FFI closure functions
     called by a Go function.  We want to see whether the caller of
     the closure functions can recover.  Look up the stack and skip
     the FFI functions.  */
  n = runtime_callers (1, &locs[0], sizeof locs / sizeof locs[0], true);
  for (i = 0; i < n; i++)
    {
      const byte *name;

      if (locs[i].function.len == 0)
	continue;
      if (locs[i].function.len < 4)
	break;
      name = locs[i].function.str;
      if (name[0] != 'f' || name[1] != 'f' || name[2] != 'i' || name[3] != '_')
	break;
    }
  if (i < n)
    __go_makefunc_ffi_can_recover (locs + i, n - i);

  FFICallbackGo(results, args, closure);

  if (i < n)
    __go_makefunc_returning ();
}

/* Allocate an FFI closure and arrange to call ffi_callback.  */

void
makeFuncFFI(const struct __go_func_type *ftyp, void *impl)
{
  ffi_cif *cif;

  cif = (ffi_cif *) __go_alloc (sizeof (ffi_cif));
  __go_func_to_cif (ftyp, 0, 0, cif);

  ffi_prep_go_closure(impl, cif, ffi_callback);
}

#else /* !defined(USE_LIBFFI_CLOSURES) */

void
makeFuncFFI(const struct __go_func_type *ftyp __attribute__ ((unused)),
	    void *impl __attribute__ ((unused)))
{
  runtime_panicstring ("libgo built without FFI does not support "
		       "reflect.MakeFunc");
}

#endif
