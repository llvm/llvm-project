/* go-map-index.c -- find or insert an entry in a map.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stddef.h>
#include <stdlib.h>

#include "runtime.h"
#include "malloc.h"
#include "go-alloc.h"
#include "go-assert.h"
#include "map.h"

/* Rehash MAP to a larger size.  */

static void
__go_map_rehash (struct __go_map *map)
{
  const struct __go_map_descriptor *descriptor;
  const struct __go_type_descriptor *key_descriptor;
  uintptr_t key_offset;
  size_t key_size;
  const FuncVal *hashfn;
  uintptr_t old_bucket_count;
  void **old_buckets;
  uintptr_t new_bucket_count;
  void **new_buckets;
  uintptr_t i;

  descriptor = map->__descriptor;

  key_descriptor = descriptor->__map_descriptor->__key_type;
  key_offset = descriptor->__key_offset;
  key_size = key_descriptor->__size;
  hashfn = key_descriptor->__hashfn;

  old_bucket_count = map->__bucket_count;
  old_buckets = map->__buckets;

  new_bucket_count = __go_map_next_prime (old_bucket_count * 2);
  new_buckets = (void **) __go_alloc (new_bucket_count * sizeof (void *));
  __builtin_memset (new_buckets, 0, new_bucket_count * sizeof (void *));

  for (i = 0; i < old_bucket_count; ++i)
    {
      char* entry;
      char* next;

      for (entry = old_buckets[i]; entry != NULL; entry = next)
	{
	  size_t key_hash;
	  size_t new_bucket_index;

	  /* We could speed up rehashing at the cost of memory space
	     by caching the hash code.  */
	  key_hash = __go_call_hashfn (hashfn, entry + key_offset, key_size);
	  new_bucket_index = key_hash % new_bucket_count;

	  next = *(char **) entry;
	  *(char **) entry = new_buckets[new_bucket_index];
	  new_buckets[new_bucket_index] = entry;
	}
    }

  if (old_bucket_count * sizeof (void *) >= TinySize)
    __go_free (old_buckets);

  map->__bucket_count = new_bucket_count;
  map->__buckets = new_buckets;
}

/* Find KEY in MAP, return a pointer to the value.  If KEY is not
   present, then if INSERT is false, return NULL, and if INSERT is
   true, insert a new value and zero-initialize it before returning a
   pointer to it.  */

void *
__go_map_index (struct __go_map *map, const void *key, _Bool insert)
{
  const struct __go_map_descriptor *descriptor;
  const struct __go_type_descriptor *key_descriptor;
  uintptr_t key_offset;
  const FuncVal *equalfn;
  size_t key_hash;
  size_t key_size;
  size_t bucket_index;
  char *entry;

  if (map == NULL)
    {
      if (insert)
	runtime_panicstring ("assignment to entry in nil map");
      return NULL;
    }

  descriptor = map->__descriptor;

  key_descriptor = descriptor->__map_descriptor->__key_type;
  key_offset = descriptor->__key_offset;
  key_size = key_descriptor->__size;
  __go_assert (key_size != -1UL);
  equalfn = key_descriptor->__equalfn;

  key_hash = __go_call_hashfn (key_descriptor->__hashfn, key, key_size);
  bucket_index = key_hash % map->__bucket_count;

  entry = (char *) map->__buckets[bucket_index];
  while (entry != NULL)
    {
      if (__go_call_equalfn (equalfn, key, entry + key_offset, key_size))
	return entry + descriptor->__val_offset;
      entry = *(char **) entry;
    }

  if (!insert)
    return NULL;

  if (map->__element_count >= map->__bucket_count)
    {
      __go_map_rehash (map);
      bucket_index = key_hash % map->__bucket_count;
    }

  entry = (char *) __go_alloc (descriptor->__entry_size);
  __builtin_memset (entry, 0, descriptor->__entry_size);

  __builtin_memcpy (entry + key_offset, key, key_size);

  *(char **) entry = map->__buckets[bucket_index];
  map->__buckets[bucket_index] = entry;

  map->__element_count += 1;

  return entry + descriptor->__val_offset;
}
