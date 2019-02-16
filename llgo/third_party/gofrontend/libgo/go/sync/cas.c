/* cas.c -- implement sync.cas for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stdint.h>

#include "runtime.h"

_Bool cas (int32_t *, int32_t, int32_t) __asm__ (GOSYM_PREFIX "libgo_sync.sync.cas");

_Bool
cas (int32_t *ptr, int32_t old, int32_t new)
{
  return __sync_bool_compare_and_swap (ptr, old, new);
}
