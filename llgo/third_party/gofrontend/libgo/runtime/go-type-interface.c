/* go-type-interface.c -- hash and equality interface functions.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "interface.h"
#include "go-type.h"

/* A hash function for an interface.  */

uintptr_t
__go_type_hash_interface (const void *vval,
			  uintptr_t key_size __attribute__ ((unused)))
{
  const struct __go_interface *val;
  const struct __go_type_descriptor *descriptor;
  uintptr_t size;

  val = (const struct __go_interface *) vval;
  if (val->__methods == NULL)
    return 0;
  descriptor = (const struct __go_type_descriptor *) val->__methods[0];
  size = descriptor->__size;
  if (__go_is_pointer_type (descriptor))
    return __go_call_hashfn (descriptor->__hashfn, &val->__object, size);
  else
    return __go_call_hashfn (descriptor->__hashfn, val->__object, size);
}

const FuncVal __go_type_hash_interface_descriptor =
  { (void *) __go_type_hash_interface };

/* An equality function for an interface.  */

_Bool
__go_type_equal_interface (const void *vv1, const void *vv2,
			   uintptr_t key_size __attribute__ ((unused)))
{
  const struct __go_interface *v1;
  const struct __go_interface *v2;
  const struct __go_type_descriptor* v1_descriptor;
  const struct __go_type_descriptor* v2_descriptor;

  v1 = (const struct __go_interface *) vv1;
  v2 = (const struct __go_interface *) vv2;
  if (v1->__methods == NULL || v2->__methods == NULL)
    return v1->__methods == v2->__methods;
  v1_descriptor = (const struct __go_type_descriptor *) v1->__methods[0];
  v2_descriptor = (const struct __go_type_descriptor *) v2->__methods[0];
  if (!__go_type_descriptors_equal (v1_descriptor, v2_descriptor))
    return 0;
  if (__go_is_pointer_type (v1_descriptor))
    return v1->__object == v2->__object;
  else
    return __go_call_equalfn (v1_descriptor->__equalfn, v1->__object,
			      v2->__object, v1_descriptor->__size);
}

const FuncVal __go_type_equal_interface_descriptor =
  { (void *) __go_type_equal_interface };
