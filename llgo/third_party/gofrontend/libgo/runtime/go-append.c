/* go-append.c -- the go builtin append function.

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-panic.h"
#include "go-type.h"
#include "array.h"
#include "arch.h"
#include "malloc.h"

/* We should be OK if we don't split the stack here, since the only
   libc functions we call are memcpy and memmove.  If we don't do
   this, we will always split the stack, because of memcpy and
   memmove.  */
extern struct __go_open_array
__go_append (struct __go_open_array, void *, uintptr_t, uintptr_t)
  __attribute__ ((no_split_stack));

struct __go_open_array
__go_append (struct __go_open_array a, void *bvalues, uintptr_t bcount,
	     uintptr_t element_size)
{
  uintptr_t ucount;
  intgo count;

  if (bvalues == NULL || bcount == 0)
    return a;

  ucount = (uintptr_t) a.__count + bcount;
  count = (intgo) ucount;
  if ((uintptr_t) count != ucount || count <= a.__count)
    runtime_panicstring ("append: slice overflow");

  if (count > a.__capacity)
    {
      intgo m;
      uintptr capmem;
      void *n;

      m = a.__capacity;
      if (m + m < count)
	m = count;
      else
	{
	  do
	    {
	      if (a.__count < 1024)
		m += m;
	      else
		m += m / 4;
	    }
	  while (m < count);
	}

      if (element_size > 0 && (uintptr) m > MaxMem / element_size)
	runtime_panicstring ("growslice: cap out of range");

      capmem = runtime_roundupsize (m * element_size);

      n = __go_alloc (capmem);
      __builtin_memcpy (n, a.__values, a.__count * element_size);

      a.__values = n;
      a.__capacity = m;
    }

  __builtin_memmove ((char *) a.__values + a.__count * element_size,
		     bvalues, bcount * element_size);
  a.__count = count;
  return a;
}
