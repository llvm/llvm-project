/* indexbyte.c -- implement strings.IndexByte for Go.

   Copyright 2013 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stddef.h>

#include "runtime.h"
#include "go-string.h"

/* This is in C so that the compiler can optimize it appropriately.
   We deliberately don't split the stack in case it does call the
   library function, which shouldn't need much stack space.  */

intgo IndexByte (String, char)
  __asm__ (GOSYM_PREFIX "strings.IndexByte")
  __attribute__ ((no_split_stack));

intgo
IndexByte (String s, char b)
{
  const char *p;

  p = __builtin_memchr ((const char *) s.str, b, s.len);
  if (p == NULL)
    return -1;
  return p - (const char *) s.str;
}
