/*
  Name:     pi.c
  Purpose:  Computes digits of the physical constant pi.
  Author:   M. J. Fromberger

  Copyright (C) 2002-2008 Michael J. Fromberger, All Rights Reserved.

  Notes:
  Uses Machin's formula, which should be suitable for a few thousand digits.

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
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "imath.h"

int g_radix = 10; /* use this radix for output */

mp_result arctan(mp_small radix, mp_small mul, mp_small x, mp_small prec,
                 mp_int sum);

char g_buf[4096];

int main(int argc, char *argv[]) {
  mp_result res;
  mpz_t sum1, sum2;
  int ndigits, out = 0;
  clock_t start, end;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s <num-digits> [<radix>]\n", argv[0]);
    return 1;
  }

  if ((ndigits = abs(atoi(argv[1]))) == 0) {
    fprintf(stderr, "%s: you must request at least 1 digit\n", argv[0]);
    return 1;
  } else if ((mp_word)ndigits > MP_DIGIT_MAX) {
    fprintf(stderr, "%s: you may request at most %u digits\n", argv[0],
            (unsigned int)MP_DIGIT_MAX);
    return 1;
  }

  if (argc > 2) {
    int radix = atoi(argv[2]);

    if (radix < MP_MIN_RADIX || radix > MP_MAX_RADIX) {
      fprintf(stderr, "%s: you may only specify a radix between %d and %d\n",
              argv[0], MP_MIN_RADIX, MP_MAX_RADIX);
      return 1;
    }
    g_radix = radix;
  }

  mp_int_init(&sum1);
  mp_int_init(&sum2);
  start = clock();

  /* sum1 = 16 * arctan(1/5) */
  if ((res = arctan(g_radix, 16, 5, ndigits, &sum1)) != MP_OK) {
    fprintf(stderr, "%s: error computing arctan: %d\n", argv[0], res);
    out = 1;
    goto CLEANUP;
  }

  /* sum2 = 4 * arctan(1/239) */
  if ((res = arctan(g_radix, 4, 239, ndigits, &sum2)) != MP_OK) {
    fprintf(stderr, "%s: error computing arctan: %d\n", argv[0], res);
    out = 1;
    goto CLEANUP;
  }

  /* pi = sum1 - sum2 */
  if ((res = mp_int_sub(&sum1, &sum2, &sum1)) != MP_OK) {
    fprintf(stderr, "%s: error computing pi: %d\n", argv[0], res);
    out = 1;
    goto CLEANUP;
  }
  end = clock();

  mp_int_to_string(&sum1, g_radix, g_buf, sizeof(g_buf));
  printf("%c.%s\n", g_buf[0], g_buf + 1);

  fprintf(stderr, "Computation took %.2f sec.\n",
          (double)(end - start) / CLOCKS_PER_SEC);

CLEANUP:
  mp_int_clear(&sum1);
  mp_int_clear(&sum2);

  return out;
}

/*
  Compute mul * atan(1/x) to prec digits of precision, and store the
  result in sum.

  Computes atan(1/x) using the formula:

               1     1      1      1
  atan(1/x) = --- - ---- + ---- - ---- + ...
               x    3x^3   5x^5   7x^7

 */
mp_result arctan(mp_small radix, mp_small mul, mp_small x, mp_small prec,
                 mp_int sum) {
  mpz_t t, v;
  mp_result res;
  mp_small rem, sign = 1, coeff = 1;

  mp_int_init(&t);
  mp_int_init(&v);
  ++prec;

  /* Compute mul * radix^prec * x
     The initial multiplication by x saves a special case in the loop for
     the first term of the series.
   */
  if ((res = mp_int_expt_value(radix, prec, &t)) != MP_OK ||
      (res = mp_int_mul_value(&t, mul, &t)) != MP_OK ||
      (res = mp_int_mul_value(&t, x, &t)) != MP_OK)
    goto CLEANUP;

  x *= x; /* assumes x <= sqrt(MP_SMALL_MAX) */
  mp_int_zero(sum);

  do {
    if ((res = mp_int_div_value(&t, x, &t, &rem)) != MP_OK) goto CLEANUP;

    if ((res = mp_int_div_value(&t, coeff, &v, &rem)) != MP_OK) goto CLEANUP;

    /* Add or subtract the result depending on the current sign (1 = add) */
    if (sign > 0)
      res = mp_int_add(sum, &v, sum);
    else
      res = mp_int_sub(sum, &v, sum);

    if (res != MP_OK) goto CLEANUP;
    sign = -sign;
    coeff += 2;

  } while (mp_int_compare_zero(&t) != 0);

  res = mp_int_div_value(sum, radix, sum, NULL);

CLEANUP:
  mp_int_clear(&v);
  mp_int_clear(&t);

  return res;
}

/* Here there be dragons */
