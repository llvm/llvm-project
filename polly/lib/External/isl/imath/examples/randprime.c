/*
  Name:     randprime.c
  Purpose:  Generate a probable prime at random.
  Author:   M. J. Fromberger

  Usage:  randprime [-s] <bits> [<outfile>]

  Generate a randomly-chosen probable prime having <bits> significant bits, and
  write it to the specified output file or to the standard output.  If the "-s"
  option is given, a prime p is chosen such that (p - 1) / 2 is also prime.

  A prime is obtained by reading random bits from /dev/random, setting the
  low-order bit, and testing for primality.  If the first candidate is not
  prime, successive odd candidates are tried until a probable prime is found.

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

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <getopt.h>
#include <unistd.h>

#include "imath.h"
#include "iprime.h"

/* Load the specified buffer with random bytes */
int randomize(unsigned char *buf, size_t len);

/* Overwrite the specified value with n_bits random bits */
mp_result mp_int_randomize(mp_int a, mp_size n_bits);

/* Find a prime starting from the given odd seed */
mp_result find_prime(mp_int seed, FILE *fb);
mp_result find_strong_prime(mp_int seed, FILE *fb);

typedef mp_result (*find_f)(mp_int, FILE *);

int main(int argc, char *argv[]) {
  int opt, modbits;
  FILE *ofp = stdout;
  mp_result res;
  find_f find_func = find_prime;
  char tag = 'p';
  mpz_t value;

  /* Process command-line arguments */
  while ((opt = getopt(argc, argv, "s")) != EOF) {
    switch (opt) {
      case 's':
        find_func = find_strong_prime;
        tag = 'P';
        break;
      default:
        fprintf(stderr, "Usage: randprime [-s] <bits> [<outfile>]\n");
        return 1;
    }
  }

  if (optind >= argc) {
    fprintf(stderr,
            "Error:  You must specify the number of significant bits.\n");
    fprintf(stderr, "Usage: randprime [-s] <bits> [<outfile>]\n");
    return 1;
  }
  modbits = (int)strtol(argv[optind++], NULL, 0);
  if (modbits < CHAR_BIT) {
    fprintf(stderr, "Error:  Invalid value for number of significant bits.\n");
    return 1;
  }
  if (modbits % 2 == 1) ++modbits;

  /* Check if output file is specified */
  if (optind < argc) {
    if ((ofp = fopen(argv[optind], "wt")) == NULL) {
      fprintf(stderr,
              "Error:  Unable to open output file for writing.\n"
              " - Filename: %s\n"
              " - Error:    %s\n",
              argv[optind], strerror(errno));
      return 1;
    }
  }

  mp_int_init(&value);
  if ((res = mp_int_randomize(&value, modbits - 1)) != MP_OK) {
    fprintf(stderr,
            "Error:  Unable to generate random start value.\n"
            " - %s (%d)\n",
            mp_error_string(res), res);
    goto EXIT;
  }
  fprintf(stderr, "%c: ", tag);
  find_func(&value, stderr);
  fputc('\n', stderr);

  /* Write the completed value to the specified output file */
  {
    int len;
    char *obuf;

    len = mp_int_string_len(&value, 10);
    obuf = malloc(len);
    mp_int_to_string(&value, 10, obuf, len);
    fputs(obuf, ofp);
    fputc('\n', ofp);

    free(obuf);
  }

EXIT:
  fclose(ofp);
  mp_int_clear(&value);
  return 0;
}

int randomize(unsigned char *buf, size_t len) {
  FILE *rnd = fopen("/dev/random", "rb");
  size_t nr;

  if (rnd == NULL) return -1;

  nr = fread(buf, sizeof(*buf), len, rnd);
  fclose(rnd);

  return (int)nr;
}

mp_result mp_int_randomize(mp_int a, mp_size n_bits) {
  mp_size n_bytes = (n_bits + CHAR_BIT - 1) / CHAR_BIT;
  unsigned char *buf;
  mp_result res = MP_OK;

  if ((buf = malloc(n_bytes)) == NULL) return MP_MEMORY;

  if ((mp_size)randomize(buf, n_bytes) != n_bytes) {
    res = MP_TRUNC;
    goto CLEANUP;
  }

  /* Clear bits beyond the number requested */
  if (n_bits % CHAR_BIT != 0) {
    unsigned char b_mask = (1 << (n_bits % CHAR_BIT)) - 1;
    unsigned char t_mask = (1 << (n_bits % CHAR_BIT)) >> 1;

    buf[0] &= b_mask;
    buf[0] |= t_mask;
  }

  /* Set low-order bit to insure value is odd */
  buf[n_bytes - 1] |= 1;

  res = mp_int_read_unsigned(a, buf, n_bytes);

CLEANUP:
  memset(buf, 0, n_bytes);
  free(buf);

  return res;
}

mp_result find_prime(mp_int seed, FILE *fb) {
  mp_result res;
  int count = 0;

  if (mp_int_is_even(seed)) {
    if ((res = mp_int_add_value(seed, 1, seed)) != MP_OK) {
      return res;
    }
  }

  while ((res = mp_int_is_prime(seed)) == MP_FALSE) {
    ++count;

    if (fb != NULL && (count % 50) == 0) {
      fputc('.', fb);
    }
    if ((res = mp_int_add_value(seed, 2, seed)) != MP_OK) {
      return res;
    }
  }

  if (res == MP_TRUE && fb != NULL) fputc('+', fb);

  return res;
}

mp_result find_strong_prime(mp_int seed, FILE *fb) {
  mp_result res = MP_OK;
  mpz_t t;

  mp_int_init(&t);
  for (;;) {
    if (find_prime(seed, fb) != MP_TRUE) break;
    if (mp_int_copy(seed, &t) != MP_OK) break;

    if (mp_int_mul_pow2(&t, 1, &t) != MP_OK ||
        mp_int_add_value(&t, 1, &t) != MP_OK) {
      break;
    }

    if ((res = mp_int_is_prime(&t)) == MP_TRUE) {
      if (fb != NULL) fputc('!', fb);

      res = mp_int_copy(&t, seed);
      break;
    } else if (res != MP_FALSE)
      break;

    if (fb != NULL) fputc('x', fb);
    if (mp_int_add_value(seed, 2, seed) != MP_OK) break;
  }

  mp_int_clear(&t);
  return res;
}

/* Here there be dragons */
