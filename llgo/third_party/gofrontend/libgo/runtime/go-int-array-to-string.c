/* go-int-array-to-string.c -- convert an array of ints to a string in Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "go-assert.h"
#include "runtime.h"
#include "arch.h"
#include "malloc.h"

String
__go_int_array_to_string (const void* p, intgo len)
{
  const int32 *ints;
  intgo slen;
  intgo i;
  unsigned char *retdata;
  String ret;
  unsigned char *s;

  ints = (const int32 *) p;

  slen = 0;
  for (i = 0; i < len; ++i)
    {
      int32 v;

      v = ints[i];

      if (v < 0 || v > 0x10ffff)
	v = 0xfffd;
      else if (0xd800 <= v && v <= 0xdfff)
	v = 0xfffd;

      if (v <= 0x7f)
	slen += 1;
      else if (v <= 0x7ff)
	slen += 2;
      else if (v <= 0xffff)
	slen += 3;
      else
	slen += 4;
    }

  retdata = runtime_mallocgc ((uintptr) slen, 0, FlagNoScan);
  ret.str = retdata;
  ret.len = slen;

  s = retdata;
  for (i = 0; i < len; ++i)
    {
      int32 v;

      v = ints[i];

      /* If V is out of range for UTF-8, substitute the replacement
	 character.  */
      if (v < 0 || v > 0x10ffff)
	v = 0xfffd;
      else if (0xd800 <= v && v <= 0xdfff)
	v = 0xfffd;

      if (v <= 0x7f)
	*s++ = v;
      else if (v <= 0x7ff)
	{
	  *s++ = 0xc0 | ((v >> 6) & 0x1f);
	  *s++ = 0x80 | (v & 0x3f);
	}
      else if (v <= 0xffff)
	{
	  *s++ = 0xe0 | ((v >> 12) & 0xf);
	  *s++ = 0x80 | ((v >> 6) & 0x3f);
	  *s++ = 0x80 | (v & 0x3f);
	}
      else
	{
	  *s++ = 0xf0 | ((v >> 18) & 0x7);
	  *s++ = 0x80 | ((v >> 12) & 0x3f);
	  *s++ = 0x80 | ((v >> 6) & 0x3f);
	  *s++ = 0x80 | (v & 0x3f);
	}
    }

  __go_assert (s - retdata == slen);

  return ret;
}
