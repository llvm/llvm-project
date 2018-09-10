/* go-string-to-int-array.c -- convert a string to an array of ints in Go.

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-alloc.h"
#include "go-string.h"
#include "array.h"
#include "arch.h"
#include "malloc.h"

struct __go_open_array
__go_string_to_int_array (String str)
{
  size_t c;
  const unsigned char *p;
  const unsigned char *pend;
  uintptr mem;
  uint32_t *data;
  uint32_t *pd;
  struct __go_open_array ret;

  c = 0;
  p = str.str;
  pend = p + str.len;
  while (p < pend)
    {
      int rune;

      ++c;
      p += __go_get_rune (p, pend - p, &rune);
    }

  if (c > MaxMem / sizeof (uint32_t))
    runtime_throw ("out of memory");

  mem = runtime_roundupsize (c * sizeof (uint32_t));
  data = (uint32_t *) runtime_mallocgc (mem, 0, FlagNoScan | FlagNoZero);
  p = str.str;
  pd = data;
  while (p < pend)
    {
      int rune;

      p += __go_get_rune (p, pend - p, &rune);
      *pd++ = rune;
    }
  if (mem > (uintptr) c * sizeof (uint32_t))
    __builtin_memset (data + c, 0, mem - (uintptr) c * sizeof (uint32_t));
  ret.__values = (void *) data;
  ret.__count = c;
  ret.__capacity = (intgo) (mem / sizeof (uint32_t));
  return ret;
}
