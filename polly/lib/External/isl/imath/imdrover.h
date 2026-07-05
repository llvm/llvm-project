/*
  Name:     imdrover.h
  Purpose:  Keeper of the hordes of testing code.
  Author:   M. J. Fromberger

  Copyright (C) 2002-2007 Michael J. Fromberger, All Rights Reserved.

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

#ifndef IMDROVER_H_
#define IMDROVER_H_

#include <stdbool.h>
#include <stdio.h>

typedef struct {
  int    line;
  char  *file;
  char  *code;
  int    num_inputs;
  char **input;
  int    num_outputs;
  char **output;
} testspec_t;

typedef bool (*test_f)(testspec_t *, FILE *);

/* Call this once at the outset to set up test registers */
void init_testing(void);
void reset_registers(void);

/* Integer tests, and general */
bool test_init(testspec_t* t, FILE* ofp);
bool test_set(testspec_t* t, FILE* ofp);
bool test_neg(testspec_t* t, FILE* ofp);
bool test_abs(testspec_t* t, FILE* ofp);
bool test_add(testspec_t* t, FILE* ofp);
bool test_sub(testspec_t* t, FILE* ofp);
bool test_mul(testspec_t* t, FILE* ofp);
bool test_mulp2(testspec_t* t, FILE* ofp);
bool test_mulv(testspec_t* t, FILE* ofp);
bool test_sqr(testspec_t* t, FILE* ofp);
bool test_div(testspec_t* t, FILE* ofp);
bool test_divp2(testspec_t* t, FILE* ofp);
bool test_divv(testspec_t* t, FILE* ofp);
bool test_expt(testspec_t* t, FILE* ofp);
bool test_exptv(testspec_t* t, FILE* ofp);
bool test_exptf(testspec_t* t, FILE* ofp);
bool test_mod(testspec_t* t, FILE* ofp);
bool test_gcd(testspec_t* t, FILE* ofp);
bool test_egcd(testspec_t* t, FILE* ofp);
bool test_lcm(testspec_t* t, FILE* ofp);
bool test_sqrt(testspec_t* t, FILE* ofp);
bool test_root(testspec_t* t, FILE* ofp);
bool test_invmod(testspec_t* t, FILE* ofp);
bool test_exptmod(testspec_t* t, FILE* ofp);
bool test_exptmod_ev(testspec_t* t, FILE* ofp);
bool test_exptmod_bv(testspec_t* t, FILE* ofp);
bool test_comp(testspec_t* t, FILE* ofp);
bool test_ucomp(testspec_t* t, FILE* ofp);
bool test_zcomp(testspec_t* t, FILE* ofp);
bool test_vcomp(testspec_t* t, FILE* ofp);
bool test_uvcomp(testspec_t* t, FILE* ofp);
bool test_tostr(testspec_t* t, FILE* ofp);
bool test_tobin(testspec_t* t, FILE* ofp);
bool test_to_int(testspec_t* t, FILE* ofp);
bool test_to_uint(testspec_t* t, FILE* ofp);
bool test_read_binary(testspec_t* t, FILE* ofp);
bool test_to_uns(testspec_t* t, FILE* ofp);
bool test_read_uns(testspec_t* t, FILE* ofp);
bool test_meta(testspec_t* t, FILE* ofp);

/* Rational tests */
bool test_qneg(testspec_t* t, FILE* ofp);
bool test_qrecip(testspec_t* t, FILE* ofp);
bool test_qabs(testspec_t* t, FILE* ofp);
bool test_qadd(testspec_t* t, FILE* ofp);
bool test_qsub(testspec_t* t, FILE* ofp);
bool test_qmul(testspec_t* t, FILE* ofp);
bool test_qdiv(testspec_t* t, FILE* ofp);
bool test_qdiv(testspec_t* t, FILE* ofp);
bool test_qaddz(testspec_t* t, FILE* ofp);
bool test_qsubz(testspec_t* t, FILE* ofp);
bool test_qmulz(testspec_t* t, FILE* ofp);
bool test_qdivz(testspec_t* t, FILE* ofp);
bool test_qexpt(testspec_t* t, FILE* ofp);
bool test_qtostr(testspec_t* t, FILE* ofp);
bool test_qtodec(testspec_t* t, FILE* ofp);
bool test_qrdec(testspec_t* t, FILE* ofp);

/* Primality testing tests */
bool test_is_prime(testspec_t* t, FILE *ofp);

#endif /* IMDROVER_H_ */
