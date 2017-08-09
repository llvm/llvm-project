/* go-append.c -- the go builtin copy function.

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stddef.h>
#include <stdint.h>

/* We should be OK if we don't split the stack here, since we are just
   calling memmove which shouldn't need much stack.  If we don't do
   this we will always split the stack, because of memmove.  */

extern void
__go_copy (void *, void *, uintptr_t)
  __attribute__ ((no_split_stack));

void
__go_copy (void *a, void *b, uintptr_t len)
{
  __builtin_memmove (a, b, len);
}
