/*
  Name:     bintest.c
  Purpose:  Test driver for binary input/output formats from IMath.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imath.h"

int main(int argc, char *argv[]) {
  unsigned char buf[512];
  mpz_t v, w;
  mp_result res;
  int len;

  if (argc < 2 || argv[1][0] == '\0') {
    fprintf(stderr, "Usage: bintest <value>\n");
    return 1;
  }

  mp_int_init(&v);
  mp_int_init(&w);
  res = mp_int_read_string(&v, 10, argv[1]);
  printf("Result code from mp_int_read_string() = %d (%s)\n", res,
         mp_error_string(res));

  len = mp_int_binary_len(&v);
  printf("%d bytes needed to write this value in 2's complement.\n", len);

  res = mp_int_to_binary(&v, buf, sizeof(buf));
  printf("Result code from mp_int_to_binary() = %d (%s)\n", res,
         mp_error_string(res));
  if (res != MP_OK) {
    return 1;
  }
  int ix;
  for (ix = 0; ix < (len - 1); ++ix) {
    printf("%d.", buf[ix]);
  }
  printf("%d\n", buf[ix]);

  /* Try converting back... */
  res = mp_int_read_binary(&w, buf, len);
  printf("Result code from mp_int_read_binary() = %d (%s)\n", res,
         mp_error_string(res));
  if (res == MP_OK) {
    mp_int_to_string(&w, 10, (char *)buf, sizeof(buf));

    printf("[%s]\n\n", buf);
  }

  len = mp_int_unsigned_len(&v);
  printf("%d bytes needed to write this value as unsigned.\n", len);

  res = mp_int_to_unsigned(&v, buf, sizeof(buf));
  printf("Result code from mp_int_to_unsigned() = %d\n", res);
  if (res == MP_OK) {
    int ix;

    for (ix = 0; ix < (len - 1); ++ix) {
      printf("%d.", buf[ix]);
    }

    printf("%d\n", buf[ix]);
  } else {
    return 1;
  }

  res = mp_int_read_unsigned(&w, buf, len);
  printf("Result code from mp_int_read_unsigned() = %d (%s)\n", res,
         mp_error_string(res));
  if (res == MP_OK) {
    mp_int_to_string(&w, 10, (char *)buf, sizeof(buf));

    printf("[%s]\n\n", buf);
  }

  mp_int_clear(&v);
  mp_int_clear(&w);
  return 0;
}
