/*
  Name:     iprime.c
  Purpose:  Pseudoprimality testing routines
  Author:   M. J. Fromberger

  Copyright (C) 2002-2008 Michael J. Fromberger, All Rights Reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
 */

#include "iprime.h"
#include <stdlib.h>

static int s_ptab[] = {
    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,
    53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
    199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
    383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571,
    577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
    661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761,
    769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
    877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977,
    983, 991, 997, 0, /* sentinel */
};

mp_result mp_int_is_prime(mp_int z) {
  /* Reject values less than 2 immediately. */
  if (mp_int_compare_value(z, 2) < 0) {
    return MP_FALSE;
  }
  /* First check for divisibility by small primes; this eliminates a large
     number of composite candidates quickly
   */
  for (int i = 0; s_ptab[i] != 0; i++) {
    mp_small rem;
    mp_result res;
    if (mp_int_compare_value(z, s_ptab[i]) == 0) return MP_TRUE;
    if ((res = mp_int_div_value(z, s_ptab[i], NULL, &rem)) != MP_OK) return res;
    if (rem == 0) return MP_FALSE;
  }

  /* Now try Fermat's test for several prime witnesses (since we now know from
     the above that z is not a multiple of any of them)
   */
  mp_result res;
  mpz_t tmp;

  if ((res = mp_int_init(&tmp)) != MP_OK) return res;

  for (int i = 0; i < 10 && s_ptab[i] != 0; i++) {
    if ((res = mp_int_exptmod_bvalue(s_ptab[i], z, z, &tmp)) != MP_OK) {
      return res;
    }
    if (mp_int_compare_value(&tmp, s_ptab[i]) != 0) {
      mp_int_clear(&tmp);
      return MP_FALSE;
    }
  }
  mp_int_clear(&tmp);
  return MP_TRUE;
}

/* Find the first apparent prime in ascending order from z */
mp_result mp_int_find_prime(mp_int z) {
  mp_result res;

  if (mp_int_is_even(z) && ((res = mp_int_add_value(z, 1, z)) != MP_OK))
    return res;

  while ((res = mp_int_is_prime(z)) == MP_FALSE) {
    if ((res = mp_int_add_value(z, 2, z)) != MP_OK) break;
  }

  return res;
}

/* Here there be dragons */
