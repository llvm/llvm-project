/* go-runtime-error.c -- Go runtime error.

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"

/* The compiler generates calls to this function.  This enum values
   are known to the compiler and used by compiled code.  Any change
   here must be reflected in the compiler.  */

enum
{
  /* Slice index out of bounds: negative or larger than the length of
     the slice.  */
  SLICE_INDEX_OUT_OF_BOUNDS = 0,

  /* Array index out of bounds.  */
  ARRAY_INDEX_OUT_OF_BOUNDS = 1,

  /* String index out of bounds.  */
  STRING_INDEX_OUT_OF_BOUNDS = 2,

  /* Slice slice out of bounds: negative or larger than the length of
     the slice or high bound less than low bound.  */
  SLICE_SLICE_OUT_OF_BOUNDS = 3,

  /* Array slice out of bounds.  */
  ARRAY_SLICE_OUT_OF_BOUNDS = 4,

  /* String slice out of bounds.  */
  STRING_SLICE_OUT_OF_BOUNDS = 5,

  /* Dereference of nil pointer.  This is used when there is a
     dereference of a pointer to a very large struct or array, to
     ensure that a gigantic array is not used a proxy to access random
     memory locations.  */
  NIL_DEREFERENCE = 6,

  /* Slice length or capacity out of bounds in make: negative or
     overflow or length greater than capacity.  */
  MAKE_SLICE_OUT_OF_BOUNDS = 7,

  /* Map capacity out of bounds in make: negative or overflow.  */
  MAKE_MAP_OUT_OF_BOUNDS = 8,

  /* Channel capacity out of bounds in make: negative or overflow.  */
  MAKE_CHAN_OUT_OF_BOUNDS = 9,

  /* Integer division by zero.  */
  DIVISION_BY_ZERO = 10
};

extern void __go_runtime_error () __attribute__ ((noreturn));

void
__go_runtime_error (int32 i)
{
  switch (i)
    {
    case SLICE_INDEX_OUT_OF_BOUNDS:
    case ARRAY_INDEX_OUT_OF_BOUNDS:
    case STRING_INDEX_OUT_OF_BOUNDS:
      runtime_panicstring ("index out of range");

    case SLICE_SLICE_OUT_OF_BOUNDS:
    case ARRAY_SLICE_OUT_OF_BOUNDS:
    case STRING_SLICE_OUT_OF_BOUNDS:
      runtime_panicstring ("slice bounds out of range");

    case NIL_DEREFERENCE:
      runtime_panicstring ("nil pointer dereference");

    case MAKE_SLICE_OUT_OF_BOUNDS:
      runtime_panicstring ("make slice len or cap out of range");

    case MAKE_MAP_OUT_OF_BOUNDS:
      runtime_panicstring ("make map len out of range");

    case MAKE_CHAN_OUT_OF_BOUNDS:
      runtime_panicstring ("make chan len out of range");

    case DIVISION_BY_ZERO:
      runtime_panicstring ("integer divide by zero");

    default:
      runtime_panicstring ("unknown runtime error");
    }
}
