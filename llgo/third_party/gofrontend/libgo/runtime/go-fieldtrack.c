/* go-fieldtrack.c -- structure field data analysis.

   Copyright 2012 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-type.h"
#include "map.h"

/* The compiler will track fields that have the tag go:"track".  Any
   function that refers to such a field will call this function with a
   string
       fieldtrack "package.type.field"

   This function does not actually do anything.  Instead, we gather
   the field tracking information by looking for strings of that form
   in the read-only data section.  This is, of course, a horrible
   hack, but it's good enough for now.  We can improve it, e.g., by a
   linker plugin, if this turns out to be useful.  */

void
__go_fieldtrack (byte *p __attribute__ ((unused)))
{
}

/* A runtime function to add all the tracked fields to a
   map[string]bool.  */

extern const char _etext[] __attribute__ ((weak));
extern const char __etext[] __attribute__ ((weak));
extern const char __data_start[] __attribute__ ((weak));
extern const char _edata[] __attribute__ ((weak));
extern const char __edata[] __attribute__ ((weak));
extern const char __bss_start[] __attribute__ ((weak));

void runtime_Fieldtrack (struct __go_map *) __asm__ (GOSYM_PREFIX "runtime.Fieldtrack");

void
runtime_Fieldtrack (struct __go_map *m)
{
  const char *p;
  const char *pend;
  const char *prefix;
  size_t prefix_len;

  p = __data_start;
  if (p == NULL)
    p = __etext;
  if (p == NULL)
    p = _etext;
  if (p == NULL)
    return;

  pend = __edata;
  if (pend == NULL)
    pend = _edata;
  if (pend == NULL)
    pend = __bss_start;
  if (pend == NULL)
    return;

  prefix = "fieldtrack ";
  prefix_len = __builtin_strlen (prefix);

  while (p < pend)
    {
      const char *q1;
      const char *q2;

      q1 = __builtin_memchr (p + prefix_len, '"', pend - (p + prefix_len));
      if (q1 == NULL)
	break;

      if (__builtin_memcmp (q1 - prefix_len, prefix, prefix_len) != 0)
	{
	  p = q1 + 1;
	  continue;
	}

      q1++;
      q2 = __builtin_memchr (q1, '"', pend - q1);
      if (q2 == NULL)
	break;

      if (__builtin_memchr (q1, '\0', q2 - q1) == NULL)
	{
	  String s;
	  void *v;
	  _Bool *pb;

	  s.str = (const byte *) q1;
	  s.len = q2 - q1;
	  v = __go_map_index (m, &s, 1);
	  pb = (_Bool *) v;
	  *pb = 1;
	}

      p = q2;
    }
}
