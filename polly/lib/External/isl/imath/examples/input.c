/*
  Name:    input.c
  Purpose: Basic I/O demo for IMath.
  Author:  Michael J. Fromberger

  This program demonstrates how to read and write arbitrary precision integers
  using IMath.

  Copyright (C) 2003-2008 Michael J. Fromberger, All Rights Reserved.

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

#include "imrat.h"

int main(int argc, char *argv[]) {
  mp_size radix = 10; /* Default output radix */
  mpq_t value;
  mp_result res;
  char *endp;

  if (argc < 2) {
    fprintf(stderr, "Usage: input <value> [output-base]\n");
    return 1;
  }
  if (argc > 2) {
    if ((radix = atoi(argv[2])) < MP_MIN_RADIX || (radix > MP_MAX_RADIX)) {
      fprintf(stderr, "Error:  Specified radix is out of range (%d)\n", radix);
      return 1;
    }
  }

  /* Initialize a new value, initially zero; illustrates how to check
     for errors (e.g., out of memory) and display a message.  */
  if ((res = mp_rat_init(&value)) != MP_OK) {
    fprintf(stderr, "Error in mp_rat_init(): %s\n", mp_error_string(res));
    return 1;
  }

  /* Read value in base 10 */
  if ((res = mp_rat_read_ustring(&value, 0, argv[1], &endp)) != MP_OK) {
    fprintf(stderr, "Error in mp_rat_read_ustring(): %s\n",
            mp_error_string(res));

    if (res == MP_TRUNC) fprintf(stderr, " -- remaining input is: %s\n", endp);

    mp_rat_clear(&value);
    return 1;
  }

  printf("Here is your value in base %d\n", radix);
  {
    mp_result buf_size, res;
    char *obuf;

    if (mp_rat_is_integer(&value)) {
      /* Allocate a buffer big enough to hold the given value, including
         sign and zero terminator. */
      buf_size = mp_int_string_len(MP_NUMER_P(&value), radix);
      obuf = malloc(buf_size);

      /* Convert the value to a string in the desired radix. */
      res = mp_int_to_string(MP_NUMER_P(&value), radix, obuf, buf_size);
      if (res != MP_OK) {
        fprintf(stderr, "Converstion to base %d failed: %s\n", radix,
                mp_error_string(res));
        mp_rat_clear(&value);
        return 1;
      }
    } else {
      /* Allocate a buffer big enough to hold the given value, including
         sign and zero terminator. */
      buf_size = mp_rat_string_len(&value, radix);
      obuf = malloc(buf_size);

      /* Convert the value to a string in the desired radix. */
      res = mp_rat_to_string(&value, radix, obuf, buf_size);
      if (res != MP_OK) {
        fprintf(stderr, "Conversion to base %d failed: %s\n", radix,
                mp_error_string(res));
        mp_rat_clear(&value);
        return 1;
      }
    }
    fputs(obuf, stdout);
    fputc('\n', stdout);
    free(obuf);
  }

  /* When you are done with a value, it must be "cleared" to release
     the memory it occupies */
  mp_rat_clear(&value);
  return 0;
}

/* Here there be dragons */
