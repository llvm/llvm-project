/* go-assert-interface.c -- interface type assertion for Go.

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-alloc.h"
#include "go-assert.h"
#include "go-panic.h"
#include "go-type.h"
#include "interface.h"

/* This is called by the compiler to implement a type assertion from
   one interface type to another.  This returns the value that should
   go in the first field of the result tuple.  The result may be an
   empty or a non-empty interface.  */

const void *
__go_assert_interface (const struct __go_type_descriptor *lhs_descriptor,
		       const struct __go_type_descriptor *rhs_descriptor)
{
  const struct __go_interface_type *lhs_interface;

  if (rhs_descriptor == NULL)
    {
      struct __go_empty_interface panic_arg;

      /* A type assertion is not permitted with a nil interface.  */

      runtime_newTypeAssertionError (NULL, NULL, lhs_descriptor->__reflection,
				     NULL, &panic_arg);
      __go_panic (panic_arg);
    }

  /* A type assertion to an empty interface just returns the object
     descriptor.  */

  __go_assert ((lhs_descriptor->__code & GO_CODE_MASK) == GO_INTERFACE);
  lhs_interface = (const struct __go_interface_type *) lhs_descriptor;
  if (lhs_interface->__methods.__count == 0)
    return rhs_descriptor;

  return __go_convert_interface_2 (lhs_descriptor, rhs_descriptor, 0);
}
