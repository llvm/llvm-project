/* go-strcmp.c -- the go string comparison function.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"

intgo
__go_strcmp(String s1, String s2)
{
  int i;

  i = __builtin_memcmp(s1.str, s2.str,
		       (s1.len < s2.len ? s1.len : s2.len));
  if (i != 0)
    return i;

  if (s1.len < s2.len)
    return -1;
  else if (s1.len > s2.len)
    return 1;
  else
    return 0;
}
