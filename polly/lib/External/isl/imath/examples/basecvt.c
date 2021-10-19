/*
  Name:     basecvt.c
  Purpose:  Convert integers and rationals from one base to another.
  Author:   M. J. Fromberger

  Copyright (C) 2004-2008 Michael J. Fromberger, All Rights Reserved.

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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imath.h"
#include "imrat.h"

int main(int argc, char *argv[]) {
  mp_size in_rdx, out_rdx;
  mpq_t value;
  mp_result res;
  int ix;

  if (argc < 4) {
    fprintf(stderr, "Usage: basecvt <ibase> <obase> <values>+\n");
    return 1;
  }

  in_rdx = atoi(argv[1]);
  out_rdx = atoi(argv[2]);

  if (in_rdx < MP_MIN_RADIX || in_rdx > MP_MAX_RADIX) {
    fprintf(stderr,
            "basecvt: input radix %u not allowed (minimum %u, maximum %u)\n",
            in_rdx, MP_MIN_RADIX, MP_MAX_RADIX);
    return 3;
  }
  if (out_rdx < MP_MIN_RADIX || out_rdx > MP_MAX_RADIX) {
    fprintf(stderr,
            "basecvt: output radix %u not allowed (minimum %u, maximum %u)\n",
            out_rdx, MP_MIN_RADIX, MP_MAX_RADIX);
    return 3;
  }

  if ((res = mp_rat_init(&value)) != MP_OK) {
    fprintf(stderr, "basecvt: out of memory\n");
    return 2;
  }

  for (ix = 3; ix < argc; ++ix) {
    char *buf, *endp = NULL;
    mp_result len;
    int is_int;

    res = mp_rat_read_ustring(&value, in_rdx, argv[ix], &endp);
    if (res != MP_OK && res != MP_TRUNC) {
      fprintf(stderr, "basecvt:  error reading argument %d: %s\n", ix,
              mp_error_string(res));
      break;
    } else if (*endp != '\0') {
      fprintf(stderr, "basecvt:  argument %d contains '%s' not in base %u\n",
              ix, endp, in_rdx);
      continue;
    }

    is_int = mp_rat_is_integer(&value);
    if (is_int) {
      len = mp_int_string_len(MP_NUMER_P(&value), out_rdx);
    } else {
      len = mp_rat_string_len(&value, out_rdx);
    }

    if ((buf = malloc(len)) == NULL) {
      fprintf(stderr, "basecvt:  out of memory\n");
      break;
    }

    if (is_int) {
      res = mp_int_to_string(MP_NUMER_P(&value), out_rdx, buf, len);
    } else {
      res = mp_rat_to_string(&value, out_rdx, buf, len);
    }

    if (res != MP_OK) {
      fprintf(stderr, "basecvt:  error converting argument %d: %s\n", ix,
              mp_error_string(res));
      free(buf);
      break;
    }

    printf("%s\n", buf);
    free(buf);
  }

  mp_rat_clear(&value);

  return (res != MP_OK);
}

/* Here there be dragons */
