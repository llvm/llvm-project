/*
  Name:     imtest.c
  Purpose:  Test driver for imath library.
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

  Reads tests from input files or standard input, and runs them.  Tests have
  the form:

  code:inputs:outputs

  The 'code' is a string identifying the test to be performed.  The inputs and
  outputs are comma-separated sequences of values.  The format of each input
  is:

    1005    number in decimal notation (signs ok)
    #x-C0E  number in hexadecimal notation
    #b1011  number in binary notation
    #o37750 number in octal notation
    =k      use register k for this input

  For rational tests, the following syntax is also legal:
    @5.33   use decimal notation (for rationals only)
            may be combined with radix notation, e.g. #x@A0.5C

  Each output is a string representing the value to which the corresponding
  result is compared in order to pass the test.  By default, tests are expected
  to succeed (i.e., return MP_OK).  To specify an alternate return value, use
  the notation $RESULT, where RESULT is the name of an error (e.g., MP_MEMORY,
  MP_UNDEF, etc.) or a numeric result denoted $#number (e.g., $#-5).

  Results are written to standard output in the following formats:

  filename<tab>line<tab>number<tab>result<eoln>
  filename<tab>line<tab>number<tab>result<tab>message<eoln>

  The filename and line give the offset of the test in its input file, the
  number is the numbet of the test among all inputs, starting from 1.
  The result is a textual description of the result code returned by the
  operation being tested.

  The exit status is 0 if all tests passed, 1 if one or more tests failed or
  had errors.

  Note:  There is currently a fixed limit on the length of lines by this test
  ----   driver.  You can increase it if you wish, but the code doesn't check;
         lines over the length are truncated (split).
 */

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "imath.h"
#include "imdrover.h"

#ifdef LINE_MAX
#undef LINE_MAX
#endif

#define LINE_MAX 4096

typedef struct {
  char *code;
  int num_inputs;
  int num_outputs;
  test_f call;
} test_t;

test_t g_tests[] = {
    /* What it does...    */
    {"initu", 2, 1, test_init},        /* r0 = uv(r1)        */
    {"initv", 2, 1, test_init},        /* r0 = v(r1)         */
    {"setu", 2, 1, test_set},          /* r0 = uv(r1)        */
    {"setv", 2, 1, test_set},          /* r0 = v(r1)         */
    {"neg", 2, 1, test_neg},           /* r1 = -r0           */
    {"abs", 2, 1, test_abs},           /* r1 = |r0|          */
    {"add", 3, 1, test_add},           /* r3 = r1 + r2       */
    {"addv", 3, 1, test_add},          /* r3 = r1 + v(r2)    */
    {"sub", 3, 1, test_sub},           /* r3 = r1 - r2       */
    {"subv", 3, 1, test_sub},          /* r3 = r1 - v(r2)    */
    {"mul", 3, 1, test_mul},           /* r3 = r1 * r2       */
    {"mulp2", 3, 1, test_mulp2},       /* r3 = r1 * 2^v(r2)  */
    {"mulv", 3, 1, test_mulv},         /* r3 = r1 * v(r2)    */
    {"sqr", 2, 1, test_sqr},           /* r2 = r1 * r1       */
    {"div", 4, 2, test_div},           /* r2 = r1 / r2, r3 = r1 % r2         */
    {"divp2", 4, 2, test_divp2},       /* r2 = r1 / 2^v(r2),r3 = r1 % 2^v(r2)*/
    {"divv", 3, 2, test_divv},         /* r2 = r1 / v(r2), r3 = r1 % v(r2)   */
    {"expt", 3, 1, test_expt},         /* r3 = r1 ^ v(r2) */
    {"exptv", 3, 1, test_exptv},       /* r3 = v(r1) ^ v(r2) */
    {"exptf", 3, 1, test_exptf},       /* r3 = r1 ^ r2 */
    {"mod", 3, 1, test_mod},           /* r3 = r1 % r2       */
    {"gcd", 3, 1, test_gcd},           /* r3 = gcd(r1, r2)   */
    {"egcd", 5, 3, test_egcd},         /* r3 = gcd(r1, r2) = r1*r4 + r2*r5   */
    {"lcm", 3, 1, test_lcm},           /* r3 = lcm(r1, r2)   */
    {"sqrt", 2, 1, test_sqrt},         /* r2 = sqrt(r1)      */
    {"root", 3, 1, test_root},         /* r3 = r1^(1/v(r2))  */
    {"invmod", 3, 1, test_invmod},     /* r3 = r1^-1 mod r2  */
    {"emod", 4, 1, test_exptmod},      /* r4 = r1^r2 mod r3  */
    {"emodev", 4, 1, test_exptmod_ev}, /* r4 = r1^v(r2) mod r3  */
    {"emodbv", 4, 1, test_exptmod_bv}, /* r4 = v(r1)^r2 mod r3  */
    {"cmp", 2, 1, test_comp},          /* rtn = compare(r1, r2) */
    {"cmpu", 2, 1, test_ucomp},        /* rtn = compare(|r1|, |r2|) */
    {"cmpz", 1, 1, test_zcomp},        /* rtn = compare(r1, 0)      */
    {"cmpv", 2, 1, test_vcomp},        /* rtn = compare(r1, v(r2))  */
    {"cmpuv", 2, 1, test_uvcomp},      /* rtn = compare(r1, v(r2))  */
    {"tostr", 2, 1, test_tostr},       /* r1: value, r2: radix, o1: result */
    {"tobin", 1, 1, test_tobin},       /* r1: value, o1: result binary     */
    {"readbin", 1, 1, test_read_binary}, /* r1: 2's comp, o1: result value   */
    {"to-uns", 1, 1, test_to_uns},       /* r1: value, o1: result binary     */
    {"readuns", 1, 1, test_read_uns},    /* r1: unsigned, o1: result value   */
    {"to-int", 1, 1, test_to_int},       /* r1: value, o1: result */
    {"to-uint", 1, 1, test_to_uint},     /* r1: value, o1: result */
    {"meta", -1, -1, test_meta},
    {"qneg", 2, 1, test_qneg},        /* r2 = -r1        */
    {"qrecip", 2, 1, test_qrecip},    /* r2 = 1 / r1     */
    {"qabs", 2, 1, test_qabs},        /* r2 = |r1|       */
    {"qadd", 3, 1, test_qadd},        /* r3 = r1 + r2    */
    {"qsub", 3, 1, test_qsub},        /* r3 = r1 - r2    */
    {"qmul", 3, 1, test_qmul},        /* r3 = r1 * r2    */
    {"qdiv", 3, 1, test_qdiv},        /* r3 = r1 / r2    */
    {"qaddz", 3, 1, test_qaddz},      /* r3 = r1 + r2    */
    {"qsubz", 3, 1, test_qsubz},      /* r3 = r1 - r2    */
    {"qmulz", 3, 1, test_qmulz},      /* r3 = r1 * r2    */
    {"qdivz", 3, 1, test_qdivz},      /* r3 = r1 / r2    */
    {"qexpt", 3, 1, test_qexpt},      /* r3 = r1 ^ v(r2) */
    {"qtostr", 2, 1, test_qtostr},    /* r1: value, r2: radix; o1: result */
    {"qtodec", 4, 1, test_qtodec},    /* r1: val, r2: rdx, r3: prec,
                                         r4: rounding mode; o1: res */
    {"qrdec", 2, 1, test_qrdec},      /* r1: dec, r2: rdx; o1: result value  */
    {"isprime", 1, 1, test_is_prime}, /* rtn = prime(r1) ? MP_TRUE : MP_FALSE */
    {NULL, 0, 0, NULL}                /* end of list marker */
};

char g_line[LINE_MAX];

extern mp_result imath_errno;
extern char *imath_errmsg;

const char *g_imath_strerr[] = {"MP_OK",    "MP_TRUE",  "MP_MEMORY", "MP_RANGE",
                                "MP_UNDEF", "MP_TRUNC", "MP_BADARG"};

bool process_file(char *file_name, FILE *ifp, FILE *ofp);
int read_line(FILE *ifp, char *line, int limit);
void trim_line(char *line);
int is_blank(char *line);
int parse_line(char *line, testspec_t *t);
int count_fields(char *line, int delim);
void parse_fields(char *line, int delim, char **start);
int run_test(int test_num, testspec_t *t, FILE *ofp);
void free_test(testspec_t *t);
int find_test(char *code, test_t *info);
char *error_string(mp_result res);

int main(int argc, char *argv[]) {
  int exit_status = 0;

  init_testing();

  if (argc == 1) {
    fprintf(stderr, "[reading from stdin]\n");
    if (!process_file("-", stdin, stdout)) exit_status = 1;
  } else {
    FILE *ifp;
    int i;

    for (i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "-") == 0) {
        ifp = stdin;
      } else if ((ifp = fopen(argv[i], "r")) == NULL) {
        fprintf(stderr, "Cannot open '%s': %s\n", argv[i], strerror(errno));
        return 1;
      }
      if (!process_file(argv[i], ifp, stdout)) exit_status = 1;

      fclose(ifp);
    }
  }
  return exit_status;
}

/** Reads and runs test cases from `ifp` and writes test results to `ofp`. The
    given `file_name` is used for cosmetic attribution. The return value is
    true if all tests passed, false if any tests failed or had errors. */
bool process_file(char *file_name, FILE *ifp, FILE *ofp) {
  int res, line_num, test_num = 0, num_failed = 0, num_bogus = 0;
  clock_t start, finish;

  start = clock();
  while ((line_num = read_line(ifp, g_line, LINE_MAX)) != 0) {
    testspec_t t;
    t.line = line_num;
    t.file = file_name;
    if (parse_line(g_line, &t)) {
      if ((res = run_test(++test_num, &t, ofp)) < 0) {
        ++num_bogus;
      } else if (res == 0) {
        ++num_failed;
      }
      free_test(&t);
    } else {
      fprintf(stderr, "Line %d: Incorrect input syntax.\n", line_num);
      ++num_bogus;
    }
  }
  finish = clock();

  fprintf(ofp,
          "# %s %d tests: %d passed, %d failed, %d errors. (%.2f seconds)\n",
          file_name, test_num, (test_num - num_failed - num_bogus), num_failed,
          num_bogus, ((double)(finish - start) / CLOCKS_PER_SEC));

  return num_failed == 0 && num_bogus == 0;
}

int read_line(FILE *ifp, char *line, int limit) {
  static FILE *current_fp = NULL;
  static int current_line = 0;

  if (ifp != current_fp) {
    current_fp = ifp;
    current_line = 0;
  }

  do {
    if (fgets(line, limit, ifp) == NULL) return 0;

    ++current_line;
  } while (is_blank(line));

  trim_line(line);
  return current_line;
}

/** Removes leading and trailing whitespace from a zero-terminated `line`. */
void trim_line(char *line) {
  int len;
  char *fnw = line;

  /* Remove leading whitespace */
  while (isspace((unsigned char)*fnw)) ++fnw;

  len = strlen(fnw);
  memmove(line, fnw, len);

  /* Remove trailing whitespace (including linefeeds) */
  fnw = line + len - 1;
  while (fnw >= line && isspace((unsigned char)*fnw)) *fnw-- = '\0';
}

/** Reports whether a zero-terminated `line` contains only whitespace after a
    line-trailing comment (`# ...`) is removed. */
int is_blank(char *line) {
  while (*line && *line != '#' && isspace((unsigned char)*line)) ++line;

  return *line == '\0' || *line == '#';
}

int parse_line(char *line, testspec_t *t) {
  char *code_brk, *in_brk;
  int num_fields;

  if ((code_brk = strchr(line, ':')) == NULL) return 0;
  if ((in_brk = strchr(code_brk + 1, ':')) == NULL) return 0;

  *code_brk = '\0';
  t->code = line;
  *in_brk = '\0';

  num_fields = count_fields(code_brk + 1, ',');
  t->num_inputs = num_fields;
  t->input = NULL;

  num_fields = count_fields(in_brk + 1, ',');
  t->num_outputs = num_fields;
  t->output = NULL;

  if (t->num_inputs > 0) {
    t->input = calloc(t->num_inputs, sizeof(char *));
    parse_fields(code_brk + 1, ',', t->input);
  }
  if (t->num_outputs > 0) {
    t->output = calloc(t->num_outputs, sizeof(char *));
    parse_fields(in_brk + 1, ',', t->output);
  }
  return 1;
}

/** Returns the number of `delim` separated fields occur in `line`. */
int count_fields(char *line, int delim) {
  int count = 1;

  if (*line == '\0') return 0;

  while (*line) {
    if (*line == (char)delim && *(line + 1) != '\0') ++count;
    ++line;
  }
  return count;
}

void parse_fields(char *line, int delim, char **start) {
  int pos = 0;

  start[pos++] = line;
  while ((line = strchr(line, delim)) != NULL) {
    *line++ = '\0';
    start[pos++] = line;
  }
}

/** Runs the test cases specified by `t`, and writes its results to `ofp`. The
    `test_num` is used in log output and should reflect the global ordering of
    tests, but is not otherwise interpreted by this function.

    This function returns 0 if the test succeeds, 1 if the test fails, and -1
    if the test is broken (e.g., its code is unknown). */
int run_test(int test_num, testspec_t *t, FILE *ofp) {
  test_t info;

  /* Look up and reality check test parameters */
  if (find_test(t->code, &info) < 0) {
    fprintf(stderr, "Line %d: Test code '%s' is unknown.\n", t->line, t->code);
    return -1;
  } else {
    int errs = 0;

    if (info.num_inputs >= 0 && t->num_inputs != info.num_inputs) {
      fprintf(stderr,
              "Line %d: Wrong number of inputs to %s (want %d, have %d)\n",
              t->line, t->code, info.num_inputs, t->num_inputs);
      ++errs;
    }
    if (info.num_outputs >= 0 && t->num_outputs != info.num_outputs) {
      fprintf(stderr,
              "Line %d: Wrong number of outputs to %s (want %d, have %d)\n",
              t->line, t->code, info.num_outputs, t->num_outputs);
      ++errs;
    }
    if (errs) {
      fprintf(stderr, "Line %d: %d error(s), skipping this test.\n", t->line,
              errs);
      return -1;
    }
  }

  /* If return value is true, just print a generic OK message;
     otherwise, it is assumed that imath_errno has been set to
     a value indicating the problem. */
  if ((info.call)(t, ofp)) {
    fprintf(ofp, "%s\t%d\t%d\tOK\n", t->file, t->line, test_num);
    return 1;
  } else if (imath_errno >= MP_BADARG) {
    fprintf(ofp, "%s\t%d\t%d\t%s\n", t->file, t->line, test_num,
            error_string(imath_errno));
  } else {
    fprintf(ofp, "%s\t%d\t%d\tFAILED\t%s\n", t->file, t->line, test_num,
            imath_errmsg);
  }
  return 0;
}

/** Locates the run instructions for the specified test `code`, and if they are
    found populates `*info` with a copy. It returns -1 if `code` is unknown. */
int find_test(char *code, test_t *info) {
  int i = 0;

  while (g_tests[i].code != NULL) {
    if (strcmp(g_tests[i].code, code) == 0) {
      *info = g_tests[i];
      return i;
    }
    ++i;
  }
  return -1;
}

/** Releases the memory occupied by a test case invocation. */
void free_test(testspec_t *t) {
  assert(t != NULL);

  if (t->input != NULL) {
    free(t->input);
    t->input = NULL;
  }
  if (t->output != NULL) {
    free(t->output);
    t->output = NULL;
  }
}

/** Returns a static label string describing `res`. Note that this is not the
    same as the error string returned by `mp_error_string`, but corresponds to
    the spelling of the constant for its value. */
char *error_string(mp_result res) {
  int v = abs(res);

  return (char *)g_imath_strerr[v];
}

/* Here there be dragons */
