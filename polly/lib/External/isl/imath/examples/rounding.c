/*
  Name:     rounding.c
  Purpose:  Demonstrates rounding modes.
  Author:   M. J. Fromberger

  Bugs:  The rounding mode can only be specified by value, not name.

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
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imath.h"
#include "imrat.h"

int main(int argc, char *argv[]) {
  mp_result mode, len, res = 0;
  mp_size prec, radix;
  mpq_t value;
  char *buf;

  if (argc < 5) {
    fprintf(stderr, "Usage: rounding <mode> <precision> <radix> <value>\n");
    return 1;
  }

  if ((res = mp_rat_init(&value)) != MP_OK) {
    fprintf(stderr, "Error initializing: %s\n", mp_error_string(res));
    return 2;
  }

  mode = atoi(argv[1]);
  prec = atoi(argv[2]);
  radix = atoi(argv[3]);

  printf(
      "Rounding mode:   %d\n"
      "Precision:       %u digits\n"
      "Radix:           %u\n"
      "Input string:    \"%s\"\n",
      mode, prec, radix, argv[4]);

  if ((res = mp_rat_read_decimal(&value, radix, argv[4])) != MP_OK) {
    fprintf(stderr, "Error reading input string: %s\n", mp_error_string(res));
    goto CLEANUP;
  }

  len = mp_rat_decimal_len(&value, radix, prec);
  buf = malloc(len);

  if ((res = mp_rat_to_decimal(&value, radix, prec, mode, buf, len)) != MP_OK) {
    fprintf(stderr, "Error converting output: %s\n", mp_error_string(res));
  }

  printf("Result string:   \"%s\"\n", buf);
  free(buf);

CLEANUP:
  mp_rat_clear(&value);
  return res;
}

/* Here there be dragons */
