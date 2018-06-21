/* go-can-convert-interface.c -- can we convert to an interface?

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-assert.h"
#include "go-string.h"
#include "go-type.h"
#include "interface.h"

/* Return whether we can convert from the type in FROM_DESCRIPTOR to
   the interface in TO_DESCRIPTOR.  This is used for type
   switches.  */

_Bool
__go_can_convert_to_interface (
    const struct __go_type_descriptor *to_descriptor,
    const struct __go_type_descriptor *from_descriptor)
{
  const struct __go_interface_type *to_interface;
  int to_method_count;
  const struct __go_interface_method *to_method;
  const struct __go_uncommon_type *from_uncommon;
  int from_method_count;
  const struct __go_method *from_method;
  int i;

  /* In a type switch FROM_DESCRIPTOR can be NULL.  */
  if (from_descriptor == NULL)
    return 0;

  __go_assert ((to_descriptor->__code & GO_CODE_MASK) == GO_INTERFACE);
  to_interface = (const struct __go_interface_type *) to_descriptor;
  to_method_count = to_interface->__methods.__count;
  to_method = ((const struct __go_interface_method *)
	       to_interface->__methods.__values);

  from_uncommon = from_descriptor->__uncommon;
  if (from_uncommon == NULL)
    {
      from_method_count = 0;
      from_method = NULL;
    }
  else
    {
      from_method_count = from_uncommon->__methods.__count;
      from_method = ((const struct __go_method *)
		     from_uncommon->__methods.__values);
    }

  for (i = 0; i < to_method_count; ++i)
    {
      while (from_method_count > 0
	     && (!__go_ptr_strings_equal (from_method->__name,
					  to_method->__name)
		 || !__go_ptr_strings_equal (from_method->__pkg_path,
					     to_method->__pkg_path)))
	{
	  ++from_method;
	  --from_method_count;
	}

      if (from_method_count == 0)
	return 0;

      if (!__go_type_descriptors_equal (from_method->__mtype,
					to_method->__type))
	return 0;

      ++to_method;
      ++from_method;
      --from_method_count;
    }

  return 1;
}
