/* go-type-string.c -- hash and equality string functions.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-type.h"
#include "go-string.h"

/* A string hash function for a map.  */

uintptr_t
__go_type_hash_string (const void *vkey,
		       uintptr_t key_size __attribute__ ((unused)))
{
  uintptr_t ret;
  const String *key;
  intgo len;
  intgo i;
  const byte *p;

  ret = 5381;
  key = (const String *) vkey;
  len = key->len;
  for (i = 0, p = key->str; i < len; i++, p++)
    ret = ret * 33 + *p;
  return ret;
}

const FuncVal __go_type_hash_string_descriptor =
  { (void *) __go_type_hash_string };

/* A string equality function for a map.  */

_Bool
__go_type_equal_string (const void *vk1, const void *vk2,
			uintptr_t key_size __attribute__ ((unused)))
{
  const String *k1;
  const String *k2;

  k1 = (const String *) vk1;
  k2 = (const String *) vk2;
  return __go_ptr_strings_equal (k1, k2);
}

const FuncVal __go_type_equal_string_descriptor =
  { (void *) __go_type_equal_string };
