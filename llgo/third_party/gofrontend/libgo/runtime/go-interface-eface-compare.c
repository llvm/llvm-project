/* go-interface-eface-compare.c -- compare non-empty and empty interface.

   Copyright 2011 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-type.h"
#include "interface.h"

/* Compare a non-empty interface value with an empty interface value.
   Return 0 for equal, not zero for not equal (return value is like
   strcmp).  */

intgo
__go_interface_empty_compare (struct __go_interface left,
			      struct __go_empty_interface right)
{
  const struct __go_type_descriptor *left_descriptor;

  if (left.__methods == NULL && right.__type_descriptor == NULL)
    return 0;
  if (left.__methods == NULL || right.__type_descriptor == NULL)
    return 1;
  left_descriptor = left.__methods[0];
  if (!__go_type_descriptors_equal (left_descriptor, right.__type_descriptor))
    return 1;
  if (__go_is_pointer_type (left_descriptor))
    return left.__object == right.__object ? 0 : 1;
  if (!__go_call_equalfn (left_descriptor->__equalfn, left.__object,
			  right.__object, left_descriptor->__size))
    return 1;
  return 0;
}
