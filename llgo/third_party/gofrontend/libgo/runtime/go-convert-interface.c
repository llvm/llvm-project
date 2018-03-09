/* go-convert-interface.c -- convert interfaces for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-alloc.h"
#include "go-assert.h"
#include "go-panic.h"
#include "go-string.h"
#include "go-type.h"
#include "interface.h"

/* This is called when converting one interface type into another
   interface type.  LHS_DESCRIPTOR is the type descriptor of the
   resulting interface.  RHS_DESCRIPTOR is the type descriptor of the
   object being converted.  This builds and returns a new interface
   method table.  If any method in the LHS_DESCRIPTOR interface is not
   implemented by the object, the conversion fails.  If the conversion
   fails, then if MAY_FAIL is true this returns NULL; otherwise, it
   panics.  */

void *
__go_convert_interface_2 (const struct __go_type_descriptor *lhs_descriptor,
			  const struct __go_type_descriptor *rhs_descriptor,
			  _Bool may_fail)
{
  const struct __go_interface_type *lhs_interface;
  int lhs_method_count;
  const struct __go_interface_method* lhs_methods;
  const void **methods;
  const struct __go_uncommon_type *rhs_uncommon;
  int rhs_method_count;
  const struct __go_method *p_rhs_method;
  int i;

  if (rhs_descriptor == NULL)
    {
      /* A nil value always converts to nil.  */
      return NULL;
    }

  __go_assert ((lhs_descriptor->__code & GO_CODE_MASK) == GO_INTERFACE);
  lhs_interface = (const struct __go_interface_type *) lhs_descriptor;
  lhs_method_count = lhs_interface->__methods.__count;
  lhs_methods = ((const struct __go_interface_method *)
		 lhs_interface->__methods.__values);

  /* This should not be called for an empty interface.  */
  __go_assert (lhs_method_count > 0);

  rhs_uncommon = rhs_descriptor->__uncommon;
  if (rhs_uncommon == NULL || rhs_uncommon->__methods.__count == 0)
    {
      struct __go_empty_interface panic_arg;

      if (may_fail)
	return NULL;

      runtime_newTypeAssertionError (NULL, rhs_descriptor->__reflection,
				     lhs_descriptor->__reflection,
				     lhs_methods[0].__name,
				     &panic_arg);
      __go_panic (panic_arg);
    }

  rhs_method_count = rhs_uncommon->__methods.__count;
  p_rhs_method = ((const struct __go_method *)
		  rhs_uncommon->__methods.__values);

  methods = NULL;

  for (i = 0; i < lhs_method_count; ++i)
    {
      const struct __go_interface_method *p_lhs_method;

      p_lhs_method = &lhs_methods[i];

      while (rhs_method_count > 0
	     && (!__go_ptr_strings_equal (p_lhs_method->__name,
					  p_rhs_method->__name)
		 || !__go_ptr_strings_equal (p_lhs_method->__pkg_path,
					     p_rhs_method->__pkg_path)))
	{
	  ++p_rhs_method;
	  --rhs_method_count;
	}

      if (rhs_method_count == 0
	  || !__go_type_descriptors_equal (p_lhs_method->__type,
					   p_rhs_method->__mtype))
	{
	  struct __go_empty_interface panic_arg;

	  if (methods != NULL)
	    __go_free (methods);

	  if (may_fail)
	    return NULL;

	  runtime_newTypeAssertionError (NULL, rhs_descriptor->__reflection,
					 lhs_descriptor->__reflection,
					 p_lhs_method->__name, &panic_arg);
	  __go_panic (panic_arg);
	}

      if (methods == NULL)
	{
	  methods = (const void **) __go_alloc ((lhs_method_count + 1)
						* sizeof (void *));

	  /* The first field in the method table is always the type of
	     the object.  */
	  methods[0] = rhs_descriptor;
	}

      methods[i + 1] = p_rhs_method->__function;
    }

  return methods;
}

/* This is called by the compiler to convert a value from one
   interface type to another.  */

void *
__go_convert_interface (const struct __go_type_descriptor *lhs_descriptor,
			const struct __go_type_descriptor *rhs_descriptor)
{
  return __go_convert_interface_2 (lhs_descriptor, rhs_descriptor, 0);
}
