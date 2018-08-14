/* go-type-complex.c -- hash and equality complex functions.

   Copyright 2012 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "runtime.h"
#include "go-type.h"

/* Hash function for float types.  */

uintptr_t
__go_type_hash_complex (const void *vkey, uintptr_t key_size)
{
  if (key_size == 8)
    {
      const complex float *cfp;
      complex float cf;
      float cfr;
      float cfi;
      uint64_t fi;

      cfp = (const complex float *) vkey;
      cf = *cfp;

      cfr = crealf (cf);
      cfi = cimagf (cf);

      if (isinf (cfr) || isinf (cfi))
	return 0;

      /* NaN != NaN, so the hash code of a NaN is irrelevant.  Make it
	 random so that not all NaNs wind up in the same place.  */
      if (isnan (cfr) || isnan (cfi))
	return runtime_fastrand1 ();

      /* Avoid negative zero.  */
      if (cfr == 0 && cfi == 0)
	return 0;
      else if (cfr == 0)
	cf = cfi * I;
      else if (cfi == 0)
	cf = cfr;

      memcpy (&fi, &cf, 8);
      return (uintptr_t) cfi;
    }
  else if (key_size == 16)
    {
      const complex double *cdp;
      complex double cd;
      double cdr;
      double cdi;
      uint64_t di[2];

      cdp = (const complex double *) vkey;
      cd = *cdp;

      cdr = creal (cd);
      cdi = cimag (cd);

      if (isinf (cdr) || isinf (cdi))
	return 0;

      if (isnan (cdr) || isnan (cdi))
	return runtime_fastrand1 ();

      /* Avoid negative zero.  */
      if (cdr == 0 && cdi == 0)
	return 0;
      else if (cdr == 0)
	cd = cdi * I;
      else if (cdi == 0)
	cd = cdr;

      memcpy (&di, &cd, 16);
      return di[0] ^ di[1];
    }
  else
    runtime_throw ("__go_type_hash_complex: invalid complex size");
}

const FuncVal __go_type_hash_complex_descriptor =
  { (void *) __go_type_hash_complex };

/* Equality function for complex types.  */

_Bool
__go_type_equal_complex (const void *vk1, const void *vk2, uintptr_t key_size)
{
  if (key_size == 8)
    {
      const complex float *cfp1;
      const complex float *cfp2;
      
      cfp1 = (const complex float *) vk1;
      cfp2 = (const complex float *) vk2;

      return *cfp1 == *cfp2;
    }
  else if (key_size == 16)
    {
      const complex double *cdp1;
      const complex double *cdp2;
      
      cdp1 = (const complex double *) vk1;
      cdp2 = (const complex double *) vk2;

      return *cdp1 == *cdp2;
    }
  else
    runtime_throw ("__go_type_equal_complex: invalid complex size");
}

const FuncVal __go_type_equal_complex_descriptor =
  { (void *) __go_type_equal_complex };
