/* go-strslice.c -- the go string slice function.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "go-panic.h"
#include "runtime.h"
#include "arch.h"
#include "malloc.h"

String
__go_string_slice (String s, intgo start, intgo end)
{
  intgo len;
  String ret;

  len = s.len;
  if (end == -1)
    end = len;
  if (start > len || end < start || end > len)
    runtime_panicstring ("string index out of bounds");
  ret.str = s.str + start;
  ret.len = end - start;
  return ret;
}
