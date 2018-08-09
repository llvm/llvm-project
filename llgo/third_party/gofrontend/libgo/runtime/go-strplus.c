/* go-strplus.c -- the go string append function.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "arch.h"
#include "malloc.h"

String
__go_string_plus (String s1, String s2)
{
  int len;
  byte *retdata;
  String ret;

  if (s1.len == 0)
    return s2;
  else if (s2.len == 0)
    return s1;

  len = s1.len + s2.len;
  retdata = runtime_mallocgc (len, 0, FlagNoScan | FlagNoZero);
  __builtin_memcpy (retdata, s1.str, s1.len);
  __builtin_memcpy (retdata + s1.len, s2.str, s2.len);
  ret.str = retdata;
  ret.len = len;
  return ret;
}
