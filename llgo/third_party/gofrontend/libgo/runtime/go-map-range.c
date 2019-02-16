/* go-map-range.c -- implement a range clause over a map.

   Copyright 2009, 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-assert.h"
#include "map.h"

/* Initialize a range over a map.  */

void
__go_mapiterinit (const struct __go_map *h, struct __go_hash_iter *it)
{
  it->entry = NULL;
  if (h != NULL)
    {
      it->map = h;
      it->next_entry = NULL;
      it->bucket = 0;
      --it->bucket;
      __go_mapiternext(it);
    }
}

/* Move to the next iteration, updating *HITER.  */

void
__go_mapiternext (struct __go_hash_iter *it)
{
  const void *entry;

  entry = it->next_entry;
  if (entry == NULL)
    {
      const struct __go_map *map;
      uintptr_t bucket;

      map = it->map;
      bucket = it->bucket;
      while (1)
	{
	  ++bucket;
	  if (bucket >= map->__bucket_count)
	    {
	      /* Map iteration is complete.  */
	      it->entry = NULL;
	      return;
	    }
	  entry = map->__buckets[bucket];
	  if (entry != NULL)
	    break;
	}
      it->bucket = bucket;
    }
  it->entry = entry;
  it->next_entry = *(const void * const *) entry;
}

/* Get the key of the current iteration.  */

void
__go_mapiter1 (struct __go_hash_iter *it, unsigned char *key)
{
  const struct __go_map *map;
  const struct __go_map_descriptor *descriptor;
  const struct __go_type_descriptor *key_descriptor;
  const char *p;

  map = it->map;
  descriptor = map->__descriptor;
  key_descriptor = descriptor->__map_descriptor->__key_type;
  p = it->entry;
  __go_assert (p != NULL);
  __builtin_memcpy (key, p + descriptor->__key_offset, key_descriptor->__size);
}

/* Get the key and value of the current iteration.  */

void
__go_mapiter2 (struct __go_hash_iter *it, unsigned char *key,
	       unsigned char *val)
{
  const struct __go_map *map;
  const struct __go_map_descriptor *descriptor;
  const struct __go_map_type *map_descriptor;
  const struct __go_type_descriptor *key_descriptor;
  const struct __go_type_descriptor *val_descriptor;
  const char *p;

  map = it->map;
  descriptor = map->__descriptor;
  map_descriptor = descriptor->__map_descriptor;
  key_descriptor = map_descriptor->__key_type;
  val_descriptor = map_descriptor->__val_type;
  p = it->entry;
  __go_assert (p != NULL);
  __builtin_memcpy (key, p + descriptor->__key_offset,
		    key_descriptor->__size);
  __builtin_memcpy (val, p + descriptor->__val_offset,
		    val_descriptor->__size);
}
