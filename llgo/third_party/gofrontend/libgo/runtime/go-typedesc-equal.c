/* go-typedesc-equal.c -- return whether two type descriptors are equal.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-string.h"
#include "go-type.h"

/* Compare type descriptors for equality.  This is necessary because
   types may have different descriptors in different shared libraries.
   Also, unnamed types may have multiple type descriptors even in a
   single shared library.  */

_Bool
__go_type_descriptors_equal (const struct __go_type_descriptor *td1,
			     const struct __go_type_descriptor *td2)
{
  if (td1 == td2)
    return 1;
  /* In a type switch we can get a NULL descriptor.  */
  if (td1 == NULL || td2 == NULL)
    return 0;
  if (td1->__code != td2->__code || td1->__hash != td2->__hash)
    return 0;
  return __go_ptr_strings_equal (td1->__reflection, td2->__reflection);
}
