/* interface.h -- the interface type for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#ifndef LIBGO_INTERFACE_H
#define LIBGO_INTERFACE_H

struct __go_type_descriptor;

/* A variable of interface type is an instance of this struct, if the
   interface has any methods.  */

struct __go_interface
{
  /* A pointer to the interface method table.  The first pointer is
     the type descriptor of the object.  Subsequent pointers are
     pointers to functions.  This is effectively the vtable for this
     interface.  The function pointers are in the same order as the
     list in the internal representation of the interface, which sorts
     them by name.  */
  const void **__methods;

  /* The object.  If the object is a pointer--if the type descriptor
     code is GO_PTR or GO_UNSAFE_POINTER--then this field is the value
     of the object itself.  Otherwise this is a pointer to memory
     which holds the value.  */
  void *__object;
};

/* A variable of an empty interface type is an instance of this
   struct.  */

struct __go_empty_interface
{
  /* The type descriptor of the object.  */
  const struct __go_type_descriptor *__type_descriptor;

  /* The object.  This is the same as __go_interface above.  */
  void *__object;
};

extern void *
__go_convert_interface (const struct __go_type_descriptor *,
			const struct __go_type_descriptor *);

extern void *
__go_convert_interface_2 (const struct __go_type_descriptor *,
			  const struct __go_type_descriptor *,
			  _Bool may_fail);

extern _Bool
__go_can_convert_to_interface(const struct __go_type_descriptor *,
			      const struct __go_type_descriptor *);

#endif /* !defined(LIBGO_INTERFACE_H) */
