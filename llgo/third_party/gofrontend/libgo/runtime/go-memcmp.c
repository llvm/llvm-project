/* go-memcmp.c -- the go memory comparison function.

   Copyright 2012 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"

intgo
__go_memcmp (const void *p1, const void *p2, uintptr len)
{
  return __builtin_memcmp (p1, p2, len);
}
