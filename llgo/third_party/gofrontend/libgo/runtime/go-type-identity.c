/* go-type-identity.c -- hash and equality identity functions.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stddef.h>

#include "runtime.h"
#include "go-type.h"

/* An identity hash function for a type.  This is used for types where
   we can simply use the type value itself as a hash code.  This is
   true of, e.g., integers and pointers.  */

uintptr_t
__go_type_hash_identity (const void *key, uintptr_t key_size)
{
  uintptr_t ret;
  uintptr_t i;
  const unsigned char *p;

  if (key_size <= 8)
    {
      union
      {
	uint64 v;
	unsigned char a[8];
      } u;
      u.v = 0;
#ifdef WORDS_BIGENDIAN
      __builtin_memcpy (&u.a[8 - key_size], key, key_size);
#else
      __builtin_memcpy (&u.a[0], key, key_size);
#endif
      if (sizeof (uintptr_t) >= 8)
	return (uintptr_t) u.v;
      else
	return (uintptr_t) ((u.v >> 32) ^ (u.v & 0xffffffff));
    }

  ret = 5381;
  for (i = 0, p = (const unsigned char *) key; i < key_size; i++, p++)
    ret = ret * 33 + *p;
  return ret;
}

const FuncVal __go_type_hash_identity_descriptor =
  { (void *) __go_type_hash_identity };

/* An identity equality function for a type.  This is used for types
   where we can check for equality by checking that the values have
   the same bits.  */

_Bool
__go_type_equal_identity (const void *k1, const void *k2, uintptr_t key_size)
{
  return __builtin_memcmp (k1, k2, key_size) == 0;
}

const FuncVal __go_type_equal_identity_descriptor =
  { (void *) __go_type_equal_identity };
