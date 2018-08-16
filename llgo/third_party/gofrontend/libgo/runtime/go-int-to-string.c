/* go-int-to-string.c -- convert an integer to a string in Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "arch.h"
#include "malloc.h"

String
__go_int_to_string (intgo v)
{
  char buf[4];
  int len;
  unsigned char *retdata;
  String ret;

  /* A negative value is not valid UTF-8; turn it into the replacement
     character.  */
  if (v < 0)
    v = 0xfffd;

  if (v <= 0x7f)
    {
      buf[0] = v;
      len = 1;
    }
  else if (v <= 0x7ff)
    {
      buf[0] = 0xc0 + (v >> 6);
      buf[1] = 0x80 + (v & 0x3f);
      len = 2;
    }
  else
    {
      /* If the value is out of range for UTF-8, turn it into the
	 "replacement character".  */
      if (v > 0x10ffff)
	v = 0xfffd;
      /* If the value is a surrogate pair, which is invalid in UTF-8,
	 turn it into the replacement character.  */
      if (v >= 0xd800 && v < 0xe000)
	v = 0xfffd;

      if (v <= 0xffff)
	{
	  buf[0] = 0xe0 + (v >> 12);
	  buf[1] = 0x80 + ((v >> 6) & 0x3f);
	  buf[2] = 0x80 + (v & 0x3f);
	  len = 3;
	}
      else
	{
	  buf[0] = 0xf0 + (v >> 18);
	  buf[1] = 0x80 + ((v >> 12) & 0x3f);
	  buf[2] = 0x80 + ((v >> 6) & 0x3f);
	  buf[3] = 0x80 + (v & 0x3f);
	  len = 4;
	}
    }

  retdata = runtime_mallocgc (len, 0, FlagNoScan);
  __builtin_memcpy (retdata, buf, len);
  ret.str = retdata;
  ret.len = len;

  return ret;
}
