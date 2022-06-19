/*
  Name:     rsakey.c
  Purpose:  Generate keys for the RSA cryptosystem.
  Author:   M. J. Fromberger

  Usage:  rsakey [-e <expt>] <modbits> [<outfile>]

  Generates an RSA key pair with a modulus having <modbits> significant bits,
  and writes it to the specified output file, or to the standard output.  The
  -e option allows the user to specify an encryption exponent; otherwise, an
  encryption exponent is chosen at random.

  Primes p and q are obtained by reading random bits from /dev/random, setting
  the low-order bit, and testing for primality.  If the first candidate is not
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

typedef struct {
  mpz_t p;
  mpz_t q;
  mpz_t n;
  mpz_t e;
  mpz_t d;
} rsa_key;

/* Load the specified buffer with random bytes */
int randomize(unsigned char *buf, size_t len);

/* Overwrite the specified value with n_bits random bits */
mp_result mp_int_randomize(mp_int a, mp_size n_bits);

/* Find a prime starting from the given odd seed */
mp_result find_prime(mp_int seed, FILE *fb);

/* Initialize/destroy an rsa_key structure */
mp_result rsa_key_init(rsa_key *kp);
void rsa_key_clear(rsa_key *kp);
void rsa_key_write(rsa_key *kp, FILE *ofp);

int main(int argc, char *argv[]) {
  int opt, modbits;
  FILE *ofp = stdout;
  char *expt = NULL;
  rsa_key the_key;
  mp_result res;

  /* Process command-line arguments */
  while ((opt = getopt(argc, argv, "e:")) != EOF) {
    switch (opt) {
      case 'e':
        expt = optarg;
        break;
      default:
        fprintf(stderr, "Usage: rsakey [-e <expt>] <modbits> [<outfile>]\n");
        return 1;
    }
  }

  if (optind >= argc) {
    fprintf(stderr, "Error:  You must specify the number of modulus bits.\n");
    fprintf(stderr, "Usage: rsakey [-e <expt>] <modbits> [<outfile>]\n");
    return 1;
  }
  modbits = (int)strtol(argv[optind++], NULL, 0);
  if (modbits < CHAR_BIT) {
    fprintf(stderr, "Error:  Invalid value for number of modulus bits.\n");
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

  if ((res = rsa_key_init(&the_key)) != MP_OK) {
    fprintf(stderr,
            "Error initializing RSA key structure:\n"
            " - %s (%d)\n",
            mp_error_string(res), res);
    return 1;
  }

  /* If specified, try to load the key exponent */
  if (expt != NULL) {
    if ((res = mp_int_read_string(&(the_key.e), 10, expt)) != MP_OK) {
      fprintf(stderr,
              "Error:  Invalid value for encryption exponent.\n"
              " - %s (%d)\n",
              mp_error_string(res), res);
      goto EXIT;
    }
  }

  if ((res = mp_int_randomize(&(the_key.p), (modbits / 2))) != MP_OK) {
    fprintf(stderr,
            "Error:  Unable to randomize first prime.\n"
            " - %s (%d)\n",
            mp_error_string(res), res);
    goto EXIT;
  }
  fprintf(stderr, "p: ");
  find_prime(&(the_key.p), stderr);

  if ((res = mp_int_randomize(&(the_key.q), (modbits / 2))) != MP_OK) {
    fprintf(stderr,
            "Error:  Unable to randomize second prime.\n"
            " - %s (%d)\n",
            mp_error_string(res), res);
    goto EXIT;
  }
  fprintf(stderr, "\nq: ");
  find_prime(&(the_key.q), stderr);
  fputc('\n', stderr);

  /* Temporarily, the key's "n" field will be (p - 1) * (q - 1) for
     purposes of computing the decryption exponent.
   */
  mp_int_mul(&(the_key.p), &(the_key.q), &(the_key.n));
  mp_int_sub(&(the_key.n), &(the_key.p), &(the_key.n));
  mp_int_sub(&(the_key.n), &(the_key.q), &(the_key.n));
  mp_int_add_value(&(the_key.n), 1, &(the_key.n));

  if (expt == NULL &&
      (res = mp_int_randomize(&(the_key.e), (modbits / 2))) != MP_OK) {
    fprintf(stderr,
            "Error:  Unable to randomize encryption exponent.\n"
            " - %s (%d)\n",
            mp_error_string(res), res);
    goto EXIT;
  }
  while ((res = mp_int_invmod(&(the_key.e), &(the_key.n), &(the_key.d))) !=
         MP_OK) {
    if (expt != NULL) {
      fprintf(stderr,
              "Error:  Unable to compute decryption exponent.\n"
              " - %s (%d)\n",
              mp_error_string(res), res);
      goto EXIT;
    }
    if ((res = mp_int_randomize(&(the_key.e), (modbits / 2))) != MP_OK) {
      fprintf(stderr,
              "Error:  Unable to re-randomize encryption exponent.\n"
              " - %s (%d)\n",
              mp_error_string(res), res);
      goto EXIT;
    }
  }

  /* Recompute the real modulus, now that exponents are done. */
  mp_int_mul(&(the_key.p), &(the_key.q), &(the_key.n));

  /* Write completed key to the specified output file */
  rsa_key_write(&the_key, ofp);

EXIT:
  fclose(ofp);
  rsa_key_clear(&the_key);
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

  if (mp_int_is_even(seed))
    if ((res = mp_int_add_value(seed, 1, seed)) != MP_OK) return res;

  while ((res = mp_int_is_prime(seed)) == MP_FALSE) {
    ++count;

    if (fb != NULL && (count % 50) == 0) fputc('.', fb);

    if ((res = mp_int_add_value(seed, 2, seed)) != MP_OK) return res;
  }

  if (res == MP_TRUE && fb != NULL) fputc('+', fb);

  return res;
}

mp_result rsa_key_init(rsa_key *kp) {
  mp_int_init(&(kp->p));
  mp_int_init(&(kp->q));
  mp_int_init(&(kp->n));
  mp_int_init(&(kp->e));
  mp_int_init(&(kp->d));

  return MP_OK;
}

void rsa_key_clear(rsa_key *kp) {
  mp_int_clear(&(kp->p));
  mp_int_clear(&(kp->q));
  mp_int_clear(&(kp->n));
  mp_int_clear(&(kp->e));
  mp_int_clear(&(kp->d));
}

void rsa_key_write(rsa_key *kp, FILE *ofp) {
  int len;
  char *obuf;

  len = mp_int_string_len(&(kp->n), 10);
  obuf = malloc(len);
  mp_int_to_string(&(kp->p), 10, obuf, len);
  fprintf(ofp, "p = %s\n", obuf);
  mp_int_to_string(&(kp->q), 10, obuf, len);
  fprintf(ofp, "q = %s\n", obuf);
  mp_int_to_string(&(kp->e), 10, obuf, len);
  fprintf(ofp, "e = %s\n", obuf);
  mp_int_to_string(&(kp->d), 10, obuf, len);
  fprintf(ofp, "d = %s\n", obuf);
  mp_int_to_string(&(kp->n), 10, obuf, len);
  fprintf(ofp, "n = %s\n", obuf);

  free(obuf);
}

/* Here there be dragons */
