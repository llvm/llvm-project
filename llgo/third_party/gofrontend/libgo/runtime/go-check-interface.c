/* go-check-interface.c -- check an interface type for a conversion

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-panic.h"
#include "go-type.h"
#include "interface.h"

/* Check that an interface type matches for a conversion to a
   non-interface type.  This panics if the types are bad.  The actual
   extraction of the object is inlined.  */

void
__go_check_interface_type (
    const struct __go_type_descriptor *lhs_descriptor,
    const struct __go_type_descriptor *rhs_descriptor,
    const struct __go_type_descriptor *rhs_inter_descriptor)
{
  if (rhs_descriptor == NULL)
    {
      struct __go_empty_interface panic_arg;

      runtime_newTypeAssertionError(NULL, NULL, lhs_descriptor->__reflection,
				    NULL, &panic_arg);
      __go_panic(panic_arg);
    }

  if (lhs_descriptor != rhs_descriptor
      && !__go_type_descriptors_equal (lhs_descriptor, rhs_descriptor)
      && ((lhs_descriptor->__code & GO_CODE_MASK) != GO_UNSAFE_POINTER
	  || !__go_is_pointer_type (rhs_descriptor))
      && ((rhs_descriptor->__code & GO_CODE_MASK) != GO_UNSAFE_POINTER
	  || !__go_is_pointer_type (lhs_descriptor)))
    {
      struct __go_empty_interface panic_arg;

      runtime_newTypeAssertionError(rhs_inter_descriptor->__reflection,
				    rhs_descriptor->__reflection,
				    lhs_descriptor->__reflection,
				    NULL, &panic_arg);
      __go_panic(panic_arg);
    }
}
