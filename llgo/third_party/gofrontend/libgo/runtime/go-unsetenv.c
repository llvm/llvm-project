/* go-unsetenv.c -- unset an environment variable from Go.

   Copyright 2015 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "config.h"

#include <stddef.h>
#include <stdlib.h>

#include "go-alloc.h"
#include "runtime.h"
#include "arch.h"
#include "malloc.h"

/* Unset an environment variable from Go.  This is called by
   syscall.Unsetenv.  */

void unsetenv_c (String) __asm__ (GOSYM_PREFIX "syscall.unsetenv_c");

void
unsetenv_c (String k)
{
  const byte *ks;
  unsigned char *kn;
  intgo len;

  ks = k.str;
  if (ks == NULL)
    ks = (const byte *) "";
  kn = NULL;

#ifdef HAVE_UNSETENV

  if (ks != NULL && ks[k.len] != 0)
    {
      // Objects that are explicitly freed must be at least 16 bytes in size,
      // so that they are not allocated using tiny alloc.
      len = k.len + 1;
      if (len < TinySize)
	len = TinySize;
      kn = __go_alloc (len);
      __builtin_memcpy (kn, ks, k.len);
      ks = kn;
    }

  unsetenv ((const char *) ks);

#endif /* !defined(HAVE_UNSETENV) */

  if (kn != NULL)
    __go_free (kn);
}
