/* go-eface-compare.c -- compare two empty values.

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-type.h"
#include "interface.h"

/* Compare two interface values.  Return 0 for equal, not zero for not
   equal (return value is like strcmp).  */

intgo
__go_empty_interface_compare (struct __go_empty_interface left,
			      struct __go_empty_interface right)
{
  const struct __go_type_descriptor *left_descriptor;

  left_descriptor = left.__type_descriptor;

  if (left_descriptor == NULL && right.__type_descriptor == NULL)
    return 0;
  if (left_descriptor == NULL || right.__type_descriptor == NULL)
    return 1;
  if (!__go_type_descriptors_equal (left_descriptor,
				    right.__type_descriptor))
    return 1;
  if (__go_is_pointer_type (left_descriptor))
    return left.__object == right.__object ? 0 : 1;
  if (!__go_call_equalfn (left_descriptor->__equalfn, left.__object,
			  right.__object, left_descriptor->__size))
    return 1;
  return 0;
}
