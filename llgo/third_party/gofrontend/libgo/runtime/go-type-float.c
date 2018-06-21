/* go-type-float.c -- hash and equality float functions.

   Copyright 2012 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <math.h>
#include <stdint.h>
#include "runtime.h"
#include "go-type.h"

/* Hash function for float types.  */

uintptr_t
__go_type_hash_float (const void *vkey, uintptr_t key_size)
{
  if (key_size == 4)
    {
      const float *fp;
      float f;
      uint32_t si;

      fp = (const float *) vkey;
      f = *fp;

      if (isinf (f) || f == 0)
	return 0;

      /* NaN != NaN, so the hash code of a NaN is irrelevant.  Make it
	 random so that not all NaNs wind up in the same place.  */
      if (isnan (f))
	return runtime_fastrand1 ();

      memcpy (&si, vkey, 4);
      return (uintptr_t) si;
    }
  else if (key_size == 8)
    {
      const double *dp;
      double d;
      uint64_t di;

      dp = (const double *) vkey;
      d = *dp;

      if (isinf (d) || d == 0)
	return 0;

      if (isnan (d))
	return runtime_fastrand1 ();

      memcpy (&di, vkey, 8);
      return (uintptr_t) di;
    }
  else
    runtime_throw ("__go_type_hash_float: invalid float size");
}

const FuncVal __go_type_hash_float_descriptor =
  { (void *) __go_type_hash_float };

/* Equality function for float types.  */

_Bool
__go_type_equal_float (const void *vk1, const void *vk2, uintptr_t key_size)
{
  if (key_size == 4)
    {
      const float *fp1;
      const float *fp2;

      fp1 = (const float *) vk1;
      fp2 = (const float *) vk2;

      return *fp1 == *fp2;
    }
  else if (key_size == 8)
    {
      const double *dp1;
      const double *dp2;

      dp1 = (const double *) vk1;
      dp2 = (const double *) vk2;

      return *dp1 == *dp2;
    }
  else
    runtime_throw ("__go_type_equal_float: invalid float size");
}

const FuncVal __go_type_equal_float_descriptor =
  { (void *) __go_type_equal_float };
