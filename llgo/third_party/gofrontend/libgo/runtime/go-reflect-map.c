/* go-reflect-map.c -- map reflection support for Go.

   Copyright 2009, 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stdlib.h>
#include <stdint.h>

#include "runtime.h"
#include "go-alloc.h"
#include "go-assert.h"
#include "go-type.h"
#include "map.h"

/* This file implements support for reflection on maps.  These
   functions are called from reflect/value.go.  */

extern void *mapaccess (struct __go_map_type *, void *, void *)
  __asm__ (GOSYM_PREFIX "reflect.mapaccess");

void *
mapaccess (struct __go_map_type *mt, void *m, void *key)
{
  struct __go_map *map = (struct __go_map *) m;

  __go_assert ((mt->__common.__code & GO_CODE_MASK) == GO_MAP);
  if (map == NULL)
    return NULL;
  else
    return __go_map_index (map, key, 0);
}

extern void mapassign (struct __go_map_type *, void *, void *, void *)
  __asm__ (GOSYM_PREFIX "reflect.mapassign");

void
mapassign (struct __go_map_type *mt, void *m, void *key, void *val)
{
  struct __go_map *map = (struct __go_map *) m;
  void *p;

  __go_assert ((mt->__common.__code & GO_CODE_MASK) == GO_MAP);
  if (map == NULL)
    runtime_panicstring ("assignment to entry in nil map");
  p = __go_map_index (map, key, 1);
  __builtin_memcpy (p, val, mt->__val_type->__size);
}

extern void mapdelete (struct __go_map_type *, void *, void *)
  __asm__ (GOSYM_PREFIX "reflect.mapdelete");

void
mapdelete (struct __go_map_type *mt, void *m, void *key)
{
  struct __go_map *map = (struct __go_map *) m;

  __go_assert ((mt->__common.__code & GO_CODE_MASK) == GO_MAP);
  if (map == NULL)
    return;
  __go_map_delete (map, key);
}

extern int32_t maplen (void *) __asm__ (GOSYM_PREFIX "reflect.maplen");

int32_t
maplen (void *m)
{
  struct __go_map *map = (struct __go_map *) m;

  if (map == NULL)
    return 0;
  return (int32_t) map->__element_count;
}

extern unsigned char *mapiterinit (struct __go_map_type *, void *)
  __asm__ (GOSYM_PREFIX "reflect.mapiterinit");

unsigned char *
mapiterinit (struct __go_map_type *mt, void *m)
{
  struct __go_hash_iter *it;

  __go_assert ((mt->__common.__code & GO_CODE_MASK) == GO_MAP);
  it = __go_alloc (sizeof (struct __go_hash_iter));
  __go_mapiterinit ((struct __go_map *) m, it);
  return (unsigned char *) it;
}

extern void mapiternext (void *) __asm__ (GOSYM_PREFIX "reflect.mapiternext");

void
mapiternext (void *it)
{
  __go_mapiternext ((struct __go_hash_iter *) it);
}

extern void *mapiterkey (void *) __asm__ (GOSYM_PREFIX "reflect.mapiterkey");

void *
mapiterkey (void *ita)
{
  struct __go_hash_iter *it = (struct __go_hash_iter *) ita;
  const struct __go_type_descriptor *key_descriptor;
  void *key;

  if (it->entry == NULL)
    return NULL;

  key_descriptor = it->map->__descriptor->__map_descriptor->__key_type;
  key = __go_alloc (key_descriptor->__size);
  __go_mapiter1 (it, key);
  return key;
}

/* Make a new map.  We have to build our own map descriptor.  */

extern struct __go_map *makemap (const struct __go_map_type *)
  __asm__ (GOSYM_PREFIX "reflect.makemap");

struct __go_map *
makemap (const struct __go_map_type *t)
{
  struct __go_map_descriptor *md;
  unsigned int o;
  const struct __go_type_descriptor *kt;
  const struct __go_type_descriptor *vt;

  md = (struct __go_map_descriptor *) __go_alloc (sizeof (*md));
  md->__map_descriptor = t;
  o = sizeof (void *);
  kt = t->__key_type;
  o = (o + kt->__field_align - 1) & ~ (kt->__field_align - 1);
  md->__key_offset = o;
  o += kt->__size;
  vt = t->__val_type;
  o = (o + vt->__field_align - 1) & ~ (vt->__field_align - 1);
  md->__val_offset = o;
  o += vt->__size;
  o = (o + sizeof (void *) - 1) & ~ (sizeof (void *) - 1);
  o = (o + kt->__field_align - 1) & ~ (kt->__field_align - 1);
  o = (o + vt->__field_align - 1) & ~ (vt->__field_align - 1);
  md->__entry_size = o;

  return __go_new_map (md, 0);
}

extern _Bool ismapkey (const struct __go_type_descriptor *)
  __asm__ (GOSYM_PREFIX "reflect.ismapkey");

_Bool
ismapkey (const struct __go_type_descriptor *typ)
{
  return (typ != NULL
	  && (void *) typ->__hashfn->fn != (void *) __go_type_hash_error);
}
