/* go-construct-map.c -- construct a map from an initializer.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "runtime.h"
#include "map.h"

struct __go_map *
__go_construct_map (const struct __go_map_descriptor *descriptor,
		    uintptr_t count, uintptr_t entry_size,
		    uintptr_t val_offset, uintptr_t val_size,
		    const void *ventries)
{
  struct __go_map *ret;
  const unsigned char *entries;
  uintptr_t i;

  ret = __go_new_map (descriptor, count);

  entries = (const unsigned char *) ventries;
  for (i = 0; i < count; ++i)
    {
      void *val = __go_map_index (ret, entries, 1);
      __builtin_memcpy (val, entries + val_offset, val_size);
      entries += entry_size;
    }

  return ret;
}
