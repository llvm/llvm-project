/*
  Name:     imtimer.c
  Purpose:  Timing tests for the imath library.
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

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <getopt.h>
#include <unistd.h>

#include "imath.h"

double clocks_to_seconds(clock_t start, clock_t end);
double get_multiply_time(int nt, int prec);
double get_exptmod_time(int nt, int prec);
mp_int alloc_values(int nt, int prec);
void randomize_values(mp_int values, int nt, int prec);
void release_values(mp_int values, int nt);
void mp_int_random(mp_int z, int prec);

const int g_mul_factor = 1000;

int main(int argc, char *argv[]) {
  int do_mul = 0, do_exp = 0, do_header = 1;
  int num_tests, precision = 0, opt;
  mp_size threshold = 0;
  unsigned int seed = (unsigned int)time(NULL);

  while ((opt = getopt(argc, argv, "ehmnp:s:t:")) != EOF) {
    switch (opt) {
      case 'e':
        do_exp = 1;
        break;
      case 'm':
        do_mul = 1;
        break;
      case 'n':
        do_header = 0;
        break;
      case 'p':
        precision = atoi(optarg);
        break;
      case 's':
        seed = atoi(optarg);
        break;
      case 't':
        threshold = (mp_size)atoi(optarg);
        break;
      default:
        fprintf(stderr,
                "Usage:  imtimer [options] <num-tests>\n\n"
                "Options understood:\n"
                " -e        -- test modular exponentiation speed.\n"
                " -h        -- display this help message.\n"
                " -m        -- test multiplication speed.\n"
                " -n        -- no header line.\n"
                " -p <dig>  -- use values with <dig> digits.\n"
                " -s <rnd>  -- set random seed to <rnd>.\n"
                " -t <dig>  -- set recursion threshold to <dig> digits.\n\n");
        return (opt != 'h');
    }
  }

  if (optind >= argc) {
    fprintf(stderr,
            "Usage:  imtimer [options] <num-tests>\n"
            "[use \"imtimer -h\" for help with options]\n\n");
    return 1;
  } else
    num_tests = atoi(argv[optind]);

  srand(seed);

  if (num_tests <= 0) {
    fprintf(stderr, "You must request at least one test.\n");
    return 1;
  }
  if (precision < 0) {
    fprintf(stderr, "Precision must be non-negative (0 means default).\n");
    return 1;
  }
  mp_int_multiply_threshold(threshold);

  if (do_header) printf("NUM\tPREC\tBITS\tREC\tRESULT\n");
  printf("%d\t%d\t%d\t%u", num_tests, precision,
         (int)(precision * MP_DIGIT_BIT), threshold);

  if (do_mul) {
    double m_time = get_multiply_time(num_tests, precision);

    printf("\tMUL %.3f %.3f", m_time, m_time / num_tests);
  }

  if (do_exp) {
    double e_time = get_exptmod_time(num_tests, precision);

    printf("\tEXP %.3f %.3f", e_time, e_time / num_tests);
  }
  fputc('\n', stdout);
  fflush(stdout);

  return 0;
}

double clocks_to_seconds(clock_t start, clock_t end) {
  return (double)(end - start) / CLOCKS_PER_SEC;
}

mp_int alloc_values(int nt, int prec) {
  mp_int out = malloc(nt * sizeof(mpz_t));
  int i;

  if (out == NULL) return NULL;

  for (i = 0; i < nt; ++i) {
    if (mp_int_init_size(out + i, prec) != MP_OK) {
      while (--i >= 0) mp_int_clear(out + i);
      return NULL;
    }
  }

  return out;
}

void randomize_values(mp_int values, int nt, int prec) {
  int i;

  for (i = 0; i < nt; ++i) mp_int_random(values + i, prec);
}

void release_values(mp_int values, int nt) {
  int i;

  for (i = 0; i < nt; ++i) mp_int_clear(values + i);

  free(values);
}

double get_multiply_time(int nt, int prec) {
  clock_t start, end;
  mp_int values;
  int i;

  if ((values = alloc_values(3, prec)) == NULL) return 0.0;
  randomize_values(values, 2, prec);

  start = clock();
  for (i = 0; i < nt; ++i) mp_int_mul(values, values + 1, values + 2);
  end = clock();

  release_values(values, 3);

  return clocks_to_seconds(start, end);
}

double get_exptmod_time(int nt, int prec) {
  clock_t start, end;
  mp_int values;
  int i;

  if ((values = alloc_values(4, prec)) == NULL) return 0.0;
  randomize_values(values, 3, prec);

  start = clock();
  for (i = 0; i < nt; ++i)
    mp_int_exptmod(values, values + 1, values + 2, values + 3);
  end = clock();

  release_values(values, 4);

  return clocks_to_seconds(start, end);
}

void mp_int_random(mp_int z, int prec) {
  int i;

  if (prec > (int)MP_ALLOC(z)) prec = (int)MP_ALLOC(z);

  for (i = 0; i < prec; ++i) {
    mp_digit d = 0;
    int j;

    for (j = 0; j < (int)sizeof(d); ++j) {
      d = (d << CHAR_BIT) | (rand() & UCHAR_MAX);
    }

    z->digits[i] = d;
  }
  z->used = prec;
}
