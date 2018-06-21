/* map.h -- the map type for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stddef.h>
#include <stdint.h>

#include "go-type.h"

/* A map descriptor is what we need to manipulate the map.  This is
   constant for a given map type.  */

struct __go_map_descriptor
{
  /* A pointer to the type descriptor for the type of the map itself.  */
  const struct __go_map_type *__map_descriptor;

  /* A map entry is a struct with three fields:
       map_entry_type *next_entry;
       key_type key;
       value_type value;
     This is the size of that struct.  */
  uintptr_t __entry_size;

  /* The offset of the key field in a map entry struct.  */
  uintptr_t __key_offset;

  /* The offset of the value field in a map entry struct (the value
     field immediately follows the key field, but there may be some
     bytes inserted for alignment).  */
  uintptr_t __val_offset;
};

struct __go_map
{
  /* The constant descriptor for this map.  */
  const struct __go_map_descriptor *__descriptor;

  /* The number of elements in the hash table.  */
  uintptr_t __element_count;

  /* The number of entries in the __buckets array.  */
  uintptr_t __bucket_count;

  /* Each bucket is a pointer to a linked list of map entries.  */
  void **__buckets;
};

/* For a map iteration the compiled code will use a pointer to an
   iteration structure.  The iteration structure will be allocated on
   the stack.  The Go code must allocate at least enough space.  */

struct __go_hash_iter
{
  /* A pointer to the current entry.  This will be set to NULL when
     the range has completed.  The Go will test this field, so it must
     be the first one in the structure.  */
  const void *entry;
  /* The map we are iterating over.  */
  const struct __go_map *map;
  /* A pointer to the next entry in the current bucket.  This permits
     deleting the current entry.  This will be NULL when we have seen
     all the entries in the current bucket.  */
  const void *next_entry;
  /* The bucket index of the current and next entry.  */
  uintptr_t bucket;
};

extern struct __go_map *__go_new_map (const struct __go_map_descriptor *,
				      uintptr_t);

extern uintptr_t __go_map_next_prime (uintptr_t);

extern void *__go_map_index (struct __go_map *, const void *, _Bool);

extern void __go_map_delete (struct __go_map *, const void *);

extern void __go_mapiterinit (const struct __go_map *, struct __go_hash_iter *);

extern void __go_mapiternext (struct __go_hash_iter *);

extern void __go_mapiter1 (struct __go_hash_iter *it, unsigned char *key);

extern void __go_mapiter2 (struct __go_hash_iter *it, unsigned char *key,
			   unsigned char *val);
