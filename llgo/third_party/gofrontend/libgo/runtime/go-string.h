/* go-string.h -- the string type for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#ifndef LIBGO_GO_STRING_H
#define LIBGO_GO_STRING_H

#include <stddef.h>

static inline _Bool
__go_strings_equal (String s1, String s2)
{
  return (s1.len == s2.len
	  && __builtin_memcmp (s1.str, s2.str, s1.len) == 0);
}

static inline _Bool
__go_ptr_strings_equal (const String *ps1, const String *ps2)
{
  if (ps1 == NULL)
    return ps2 == NULL;
  if (ps2 == NULL)
    return 0;
  return __go_strings_equal (*ps1, *ps2);
}

extern int __go_get_rune (const unsigned char *, size_t, int32 *);

#endif /* !defined(LIBGO_GO_STRING_H) */
