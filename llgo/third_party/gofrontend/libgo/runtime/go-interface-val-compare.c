/* go-interface-val-compare.c -- compare an interface to a value.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-type.h"
#include "interface.h"

/* Compare two interface values.  Return 0 for equal, not zero for not
   equal (return value is like strcmp).  */

intgo
__go_interface_value_compare (
    struct __go_interface left,
    const struct __go_type_descriptor *right_descriptor,
    const void *val)
{
  const struct __go_type_descriptor *left_descriptor;

  if (left.__methods == NULL)
    return 1;
  left_descriptor = left.__methods[0];
  if (!__go_type_descriptors_equal (left_descriptor, right_descriptor))
    return 1;
  if (__go_is_pointer_type (left_descriptor))
    return left.__object == val ? 0 : 1;
  if (!__go_call_equalfn (left_descriptor->__equalfn, left.__object, val,
			  left_descriptor->__size))
    return 1;
  return 0;
}
