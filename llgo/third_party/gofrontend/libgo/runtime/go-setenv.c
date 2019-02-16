/* go-setenv.c -- set the C environment from Go.

   Copyright 2011 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "config.h"

#include <stddef.h>
#include <stdlib.h>

#include "go-alloc.h"
#include "runtime.h"
#include "arch.h"
#include "malloc.h"

/* Set the C environment from Go.  This is called by syscall.Setenv.  */

void setenv_c (String, String) __asm__ (GOSYM_PREFIX "syscall.setenv_c");

void
setenv_c (String k, String v)
{
  const byte *ks;
  unsigned char *kn;
  const byte *vs;
  unsigned char *vn;
  intgo len;

  ks = k.str;
  if (ks == NULL)
    ks = (const byte *) "";
  kn = NULL;

  vs = v.str;
  if (vs == NULL)
    vs = (const byte *) "";
  vn = NULL;

#ifdef HAVE_SETENV

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

  if (vs != NULL && vs[v.len] != 0)
    {
      len = v.len + 1;
      if (len < TinySize)
	len = TinySize;
      vn = __go_alloc (len);
      __builtin_memcpy (vn, vs, v.len);
      vs = vn;
    }

  setenv ((const char *) ks, (const char *) vs, 1);

#else /* !defined(HAVE_SETENV) */

  len = k.len + v.len + 2;
  if (len < TinySize)
    len = TinySize;
  kn = __go_alloc (len);
  __builtin_memcpy (kn, ks, k.len);
  kn[k.len] = '=';
  __builtin_memcpy (kn + k.len + 1, vs, v.len);
  kn[k.len + v.len + 1] = '\0';
  putenv ((char *) kn);

#endif /* !defined(HAVE_SETENV) */

  if (kn != NULL)
    __go_free (kn);
  if (vn != NULL)
    __go_free (vn);
}
