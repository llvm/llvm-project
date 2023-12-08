/*
  Name:     findprime.c
  Purpose:  Find probable primes.
  Author:   M. J. Fromberger

  Copyright (C) 2002-2008 Michael J. Fromberger, All Rights Reserved.

  Notes:
  Find the first prime number in sequence starting from the given value.
  Demonstrates the use of mp_int_find_prime().

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

#include <stdio.h>

#include "iprime.h"

int main(int argc, char *argv[]) {
  char buf[4096];
  mpz_t seed;
  mp_result res;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s <start-value>\n", argv[0]);
    return 1;
  }

  mp_int_init(&seed);
  if ((res = mp_int_read_string(&seed, 10, argv[1])) != MP_OK) {
    fprintf(stderr, "%s: error reading `%s': %d\n", argv[0], argv[1], res);
    return 2;
  }

  if (mp_int_compare_value(&seed, 131) <= 0) {
    fprintf(stderr, "%s: please enter a start value > 131\n", argv[0]);
    return 1;
  }

  if ((res = mp_int_find_prime(&seed)) != MP_TRUE) {
    fprintf(stderr, "%s: error finding prime: %d\n", argv[0], res);
    return 2;
  }

  mp_int_to_string(&seed, 10, buf, sizeof(buf));
  printf("=> %s\n", buf);

  mp_int_clear(&seed);

  return 0;
}
