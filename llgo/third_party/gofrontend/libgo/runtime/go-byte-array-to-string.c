/* go-byte-array-to-string.c -- convert an array of bytes to a string in Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "arch.h"
#include "malloc.h"

String
__go_byte_array_to_string (const void* p, intgo len)
{
  const unsigned char *bytes;
  unsigned char *retdata;
  String ret;

  bytes = (const unsigned char *) p;
  retdata = runtime_mallocgc ((uintptr) len, 0, FlagNoScan);
  __builtin_memcpy (retdata, bytes, len);
  ret.str = retdata;
  ret.len = len;
  return ret;
}
