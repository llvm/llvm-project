#include <stdio.h>
#include "i64operations.h"

int64_t         tval_a = 1234567890003LL;
int64_t         tval_b = 2345678901235LL;
int64_t         tval_c = 1234567890001LL;
int64_t         tval_d = 10001LL;
int64_t         tval_e = 10000LL;
uint64_t        tval_f = 0xffffff0750135eb9;
int64_t		tval_g = -1;

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

int
i64_eq(int64_t a, int64_t b)
{
  return (a == b);
}

int
i64_neq(int64_t a, int64_t b)
{
  return (a != b);
}

int
i64_gt(int64_t a, int64_t b)
{
  return (a > b);
}

int
i64_le(int64_t a, int64_t b)
{
  return (a <= b);
}

int
i64_ge(int64_t a, int64_t b) {
  return (a >= b);
}

int
i64_lt(int64_t a, int64_t b) {
  return (a < b);
}

int
i64_uge(uint64_t a, uint64_t b)
{
  return (a >= b);
}

int
i64_ult(uint64_t a, uint64_t b)
{
  return (a < b);
}

int
i64_ugt(uint64_t a, uint64_t b)
{
  return (a > b);
}

int
i64_ule(uint64_t a, uint64_t b)
{
  return (a <= b);
}

int64_t
i64_eq_select(int64_t a, int64_t b, int64_t c, int64_t d)
{
  return ((a == b) ? c : d);
}

int64_t
i64_neq_select(int64_t a, int64_t b, int64_t c, int64_t d)
{
  return ((a != b) ? c : d);
}

int64_t
i64_gt_select(int64_t a, int64_t b, int64_t c, int64_t d) {
  return ((a > b) ? c : d);
}

int64_t
i64_le_select(int64_t a, int64_t b, int64_t c, int64_t d) {
  return ((a <= b) ? c : d);
}

int64_t
i64_ge_select(int64_t a, int64_t b, int64_t c, int64_t d) {
  return ((a >= b) ? c : d);
}

int64_t
i64_lt_select(int64_t a, int64_t b, int64_t c, int64_t d) {
  return ((a < b) ? c : d);
}

uint64_t
i64_ugt_select(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
  return ((a > b) ? c : d);
}

uint64_t
i64_ule_select(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
  return ((a <= b) ? c : d);
}

uint64_t
i64_uge_select(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  return ((a >= b) ? c : d);
}

uint64_t
i64_ult_select(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  return ((a < b) ? c : d);
}

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

struct harness_int64_pred int64_tests_eq[] = {
  {"a %s a", &tval_a, &tval_a, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"a %s b", &tval_a, &tval_b, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"a %s c", &tval_a, &tval_c, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"d %s e", &tval_d, &tval_e, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"e %s e", &tval_e, &tval_e, &tval_c, &tval_d, TRUE_VAL, &tval_c}
};

struct harness_int64_pred int64_tests_neq[] = {
  {"a %s a", &tval_a, &tval_a, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"a %s b", &tval_a, &tval_b, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"a %s c", &tval_a, &tval_c, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"d %s e", &tval_d, &tval_e, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"e %s e", &tval_e, &tval_e, &tval_c, &tval_d, FALSE_VAL, &tval_d}
};

struct harness_int64_pred int64_tests_sgt[] = {
  {"a %s a", &tval_a, &tval_a, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"a %s b", &tval_a, &tval_b, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"a %s c", &tval_a, &tval_c, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"d %s e", &tval_d, &tval_e, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"e %s e", &tval_e, &tval_e, &tval_c, &tval_d, FALSE_VAL, &tval_d}
};

struct harness_int64_pred int64_tests_sle[] = {
  {"a %s a", &tval_a, &tval_a, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"a %s b", &tval_a, &tval_b, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"a %s c", &tval_a, &tval_c, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"d %s e", &tval_d, &tval_e, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"e %s e", &tval_e, &tval_e, &tval_c, &tval_d, TRUE_VAL, &tval_c}
};

struct harness_int64_pred int64_tests_sge[] = {
  {"a %s a", &tval_a, &tval_a, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"a %s b", &tval_a, &tval_b, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"a %s c", &tval_a, &tval_c, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"d %s e", &tval_d, &tval_e, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"e %s e", &tval_e, &tval_e, &tval_c, &tval_d, TRUE_VAL, &tval_c}
};

struct harness_int64_pred int64_tests_slt[] = {
  {"a %s a", &tval_a, &tval_a, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"a %s b", &tval_a, &tval_b, &tval_c, &tval_d, TRUE_VAL, &tval_c},
  {"a %s c", &tval_a, &tval_c, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"d %s e", &tval_d, &tval_e, &tval_c, &tval_d, FALSE_VAL, &tval_d},
  {"e %s e", &tval_e, &tval_e, &tval_c, &tval_d, FALSE_VAL, &tval_d}
};

struct int64_pred_s int64_preds[] = {
  {"eq", i64_eq, i64_eq_select,
     int64_tests_eq, ARR_SIZE(int64_tests_eq)},
  {"neq", i64_neq, i64_neq_select,
     int64_tests_neq, ARR_SIZE(int64_tests_neq)},
  {"gt", i64_gt, i64_gt_select,
     int64_tests_sgt, ARR_SIZE(int64_tests_sgt)},
  {"le", i64_le, i64_le_select,
     int64_tests_sle, ARR_SIZE(int64_tests_sle)},
  {"ge", i64_ge, i64_ge_select,
     int64_tests_sge, ARR_SIZE(int64_tests_sge)},
  {"lt", i64_lt, i64_lt_select,
     int64_tests_slt, ARR_SIZE(int64_tests_slt)}
};

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

struct harness_uint64_pred uint64_tests_ugt[] = {
  {"a %s a", (uint64_t *) &tval_a, (uint64_t *) &tval_a, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, FALSE_VAL, (uint64_t *) &tval_d},
  {"a %s b", (uint64_t *) &tval_a, (uint64_t *) &tval_b, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, FALSE_VAL, (uint64_t *) &tval_d },
  {"a %s c", (uint64_t *) &tval_a, (uint64_t *) &tval_c, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, TRUE_VAL, (uint64_t *) &tval_c },
  {"d %s e", (uint64_t *) &tval_d, (uint64_t *) &tval_e, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, TRUE_VAL, (uint64_t *) &tval_c },
  {"e %s e", (uint64_t *) &tval_e, (uint64_t *) &tval_e, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, FALSE_VAL, (uint64_t *) &tval_d }
};

struct harness_uint64_pred uint64_tests_ule[] = {
  {"a %s a", (uint64_t *) &tval_a, (uint64_t *) &tval_a, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, TRUE_VAL, (uint64_t *) &tval_c},
  {"a %s b", (uint64_t *) &tval_a, (uint64_t *) &tval_b, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, TRUE_VAL, (uint64_t *) &tval_c},
  {"a %s c", (uint64_t *) &tval_a, (uint64_t *) &tval_c, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, FALSE_VAL, (uint64_t *) &tval_d},
  {"d %s e", (uint64_t *) &tval_d, (uint64_t *) &tval_e, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, FALSE_VAL, (uint64_t *) &tval_d},
  {"e %s e", (uint64_t *) &tval_e, (uint64_t *) &tval_e, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, TRUE_VAL, (uint64_t *) &tval_c}
};

struct harness_uint64_pred uint64_tests_uge[] = {
  {"a %s a", (uint64_t *) &tval_a, (uint64_t *) &tval_a, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, TRUE_VAL, (uint64_t *) &tval_c},
  {"a %s b", (uint64_t *) &tval_a, (uint64_t *) &tval_b, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, FALSE_VAL, (uint64_t *) &tval_d},
  {"a %s c", (uint64_t *) &tval_a, (uint64_t *) &tval_c, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, TRUE_VAL, (uint64_t *) &tval_c},
  {"d %s e", (uint64_t *) &tval_d, (uint64_t *) &tval_e, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, TRUE_VAL, (uint64_t *) &tval_c},
  {"e %s e", (uint64_t *) &tval_e, (uint64_t *) &tval_e, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, TRUE_VAL, (uint64_t *) &tval_c}
};

struct harness_uint64_pred uint64_tests_ult[] = {
  {"a %s a", (uint64_t *) &tval_a, (uint64_t *) &tval_a, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, FALSE_VAL, (uint64_t *) &tval_d},
  {"a %s b", (uint64_t *) &tval_a, (uint64_t *) &tval_b, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, TRUE_VAL, (uint64_t *) &tval_c},
  {"a %s c", (uint64_t *) &tval_a, (uint64_t *) &tval_c, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, FALSE_VAL, (uint64_t *) &tval_d},
  {"d %s e", (uint64_t *) &tval_d, (uint64_t *) &tval_e, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, FALSE_VAL, (uint64_t *) &tval_d},
  {"e %s e", (uint64_t *) &tval_e, (uint64_t *) &tval_e, (uint64_t *) &tval_c,
     (uint64_t *) &tval_d, FALSE_VAL, (uint64_t *) &tval_d}
};

struct uint64_pred_s uint64_preds[] = {
  {"ugt", i64_ugt, i64_ugt_select,
     uint64_tests_ugt, ARR_SIZE(uint64_tests_ugt)},
  {"ule", i64_ule, i64_ule_select,
     uint64_tests_ule, ARR_SIZE(uint64_tests_ule)},
  {"uge", i64_uge, i64_uge_select,
     uint64_tests_uge, ARR_SIZE(uint64_tests_uge)},
  {"ult", i64_ult, i64_ult_select,
     uint64_tests_ult, ARR_SIZE(uint64_tests_ult)}
};

int
compare_expect_int64(const struct int64_pred_s * pred)
{
  int             j, failed = 0;

  for (j = 0; j < pred->n_tests; ++j) {
    int             pred_result;

    pred_result = (*pred->predfunc) (*pred->tests[j].lhs, *pred->tests[j].rhs);

    if (pred_result != pred->tests[j].expected) {
      char            str[64];

      sprintf(str, pred->tests[j].fmt_string, pred->name);
      printf("%s: returned value is %d, expecting %d\n", str,
	     pred_result, pred->tests[j].expected);
      printf("  lhs = %19lld (0x%016llx)\n", *pred->tests[j].lhs,
             *pred->tests[j].lhs);
      printf("  rhs = %19lld (0x%016llx)\n", *pred->tests[j].rhs,
             *pred->tests[j].rhs);
      ++failed;
    } else {
      int64_t         selresult;

      selresult = (pred->selfunc) (*pred->tests[j].lhs, *pred->tests[j].rhs,
                                   *pred->tests[j].select_a,
                                   *pred->tests[j].select_b);

      if (selresult != *pred->tests[j].select_expected) {
	char            str[64];

	sprintf(str, pred->tests[j].fmt_string, pred->name);
	printf("%s select: returned value is %d, expecting %d\n", str,
	       pred_result, pred->tests[j].expected);
	printf("  lhs   = %19lld (0x%016llx)\n", *pred->tests[j].lhs,
	       *pred->tests[j].lhs);
	printf("  rhs   = %19lld (0x%016llx)\n", *pred->tests[j].rhs,
	       *pred->tests[j].rhs);
	printf("  true  = %19lld (0x%016llx)\n", *pred->tests[j].select_a,
	       *pred->tests[j].select_a);
	printf("  false = %19lld (0x%016llx)\n", *pred->tests[j].select_b,
	       *pred->tests[j].select_b);
	++failed;
      }
    }
  }

  printf("  %d tests performed, should be %d.\n", j, pred->n_tests);

  return failed;
}

int
compare_expect_uint64(const struct uint64_pred_s * pred)
{
  int             j, failed = 0;

  for (j = 0; j < pred->n_tests; ++j) {
    int             pred_result;

    pred_result = (*pred->predfunc) (*pred->tests[j].lhs, *pred->tests[j].rhs);
    if (pred_result != pred->tests[j].expected) {
      char            str[64];

      sprintf(str, pred->tests[j].fmt_string, pred->name);
      printf("%s: returned value is %d, expecting %d\n", str,
	     pred_result, pred->tests[j].expected);
      printf("  lhs = %19llu (0x%016llx)\n", *pred->tests[j].lhs,
             *pred->tests[j].lhs);
      printf("  rhs = %19llu (0x%016llx)\n", *pred->tests[j].rhs,
             *pred->tests[j].rhs);
      ++failed;
    } else {
      uint64_t        selresult;

      selresult = (pred->selfunc) (*pred->tests[j].lhs, *pred->tests[j].rhs,
                                   *pred->tests[j].select_a,
                                   *pred->tests[j].select_b);
      if (selresult != *pred->tests[j].select_expected) {
	char            str[64];

	sprintf(str, pred->tests[j].fmt_string, pred->name);
	printf("%s select: returned value is %d, expecting %d\n", str,
	       pred_result, pred->tests[j].expected);
	printf("  lhs   = %19llu (0x%016llx)\n", *pred->tests[j].lhs,
	       *pred->tests[j].lhs);
	printf("  rhs   = %19llu (0x%016llx)\n", *pred->tests[j].rhs,
	       *pred->tests[j].rhs);
	printf("  true  = %19llu (0x%016llx)\n", *pred->tests[j].select_a,
	       *pred->tests[j].select_a);
	printf("  false = %19llu (0x%016llx)\n", *pred->tests[j].select_b,
	       *pred->tests[j].select_b);
	++failed;
      }
    }
  }

  printf("  %d tests performed, should be %d.\n", j, pred->n_tests);

  return failed;
}

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

int
test_i64_sext_i32(int in, int64_t expected) {
  int64_t result = (int64_t) in;

  if (result != expected) {
    char str[64];
    sprintf(str, "i64_sext_i32(%d) returns %lld\n", in, result);
    return 1;
  }

  return 0;
}

int
test_i64_sext_i16(short in, int64_t expected) {
  int64_t result = (int64_t) in;

  if (result != expected) {
    char str[64];
    sprintf(str, "i64_sext_i16(%hd) returns %lld\n", in, result);
    return 1;
  }

  return 0;
}

int
test_i64_sext_i8(signed char in, int64_t expected) {
  int64_t result = (int64_t) in;

  if (result != expected) {
    char str[64];
    sprintf(str, "i64_sext_i8(%d) returns %lld\n", in, result);
    return 1;
  }

  return 0;
}

int
test_i64_zext_i32(unsigned int in, uint64_t expected) {
  uint64_t result = (uint64_t) in;

  if (result != expected) {
    char str[64];
    sprintf(str, "i64_zext_i32(%u) returns %llu\n", in, result);
    return 1;
  }

  return 0;
}

int
test_i64_zext_i16(unsigned short in, uint64_t expected) {
  uint64_t result = (uint64_t) in;

  if (result != expected) {
    char str[64];
    sprintf(str, "i64_zext_i16(%hu) returns %llu\n", in, result);
    return 1;
  }

  return 0;
}

int
test_i64_zext_i8(unsigned char in, uint64_t expected) {
  uint64_t result = (uint64_t) in;

  if (result != expected) {
    char str[64];
    sprintf(str, "i64_zext_i8(%u) returns %llu\n", in, result);
    return 1;
  }

  return 0;
}

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

int64_t
i64_shl_const(int64_t a) {
  return a << 10;
}

int64_t
i64_shl(int64_t a, int amt) {
  return a << amt;
}

uint64_t
u64_shl_const(uint64_t a) {
  return a << 10;
}

uint64_t
u64_shl(uint64_t a, int amt) {
  return a << amt;
}

int64_t
i64_srl_const(int64_t a) {
  return a >> 10;
}

int64_t
i64_srl(int64_t a, int amt) {
  return a >> amt;
}

uint64_t
u64_srl_const(uint64_t a) {
  return a >> 10;
}

uint64_t
u64_srl(uint64_t a, int amt) {
  return a >> amt;
}

int64_t
i64_sra_const(int64_t a) {
  return a >> 10;
}

int64_t
i64_sra(int64_t a, int amt) {
  return a >> amt;
}

uint64_t
u64_sra_const(uint64_t a) {
  return a >> 10;
}

uint64_t
u64_sra(uint64_t a, int amt) {
  return a >> amt;
}

int
test_u64_constant_shift(const char *func_name, uint64_t (*func)(uint64_t), uint64_t a, uint64_t expected) {
  uint64_t result = (*func)(a);

  if (result != expected) {
    printf("%s(0x%016llx) returns 0x%016llx, expected 0x%016llx\n", func_name, a, result, expected);
    return 1;
  }

  return 0;
}

int
test_i64_constant_shift(const char *func_name, int64_t (*func)(int64_t), int64_t a, int64_t expected) {
  int64_t result = (*func)(a);

  if (result != expected) {
    printf("%s(0x%016llx) returns 0x%016llx, expected 0x%016llx\n", func_name, a, result, expected);
    return 1;
  }

  return 0;
}

int
test_u64_variable_shift(const char *func_name, uint64_t (*func)(uint64_t, int), uint64_t a, unsigned int b, uint64_t expected) {
  uint64_t result = (*func)(a, b);

  if (result != expected) {
    printf("%s(0x%016llx, %d) returns 0x%016llx, expected 0x%016llx\n", func_name, a, b, result, expected);
    return 1;
  }

  return 0;
}

int
test_i64_variable_shift(const char *func_name, int64_t (*func)(int64_t, int), int64_t a, unsigned int b, int64_t expected) {
  int64_t result = (*func)(a, b);

  if (result != expected) {
    printf("%s(0x%016llx, %d) returns 0x%016llx, expected 0x%016llx\n", func_name, a, b, result, expected);
    return 1;
  }

  return 0;
}

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

int64_t i64_mul(int64_t a, int64_t b) {
  return a * b;
}

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */

int
main(void)
{
  int             i, j, failed = 0;
  const char     *something_failed = "  %d tests failed.\n";
  const char     *all_tests_passed = "  All tests passed.\n";

  printf("tval_a = %20lld (0x%016llx)\n", tval_a, tval_a);
  printf("tval_b = %20lld (0x%016llx)\n", tval_b, tval_b);
  printf("tval_c = %20lld (0x%016llx)\n", tval_c, tval_c);
  printf("tval_d = %20lld (0x%016llx)\n", tval_d, tval_d);
  printf("tval_e = %20lld (0x%016llx)\n", tval_e, tval_e);
  printf("tval_f = %20llu (0x%016llx)\n", tval_f, tval_f);
  printf("tval_g = %20llu (0x%016llx)\n", tval_g, tval_g);
  printf("----------------------------------------\n");

  for (i = 0; i < ARR_SIZE(int64_preds); ++i) {
    printf("%s series:\n", int64_preds[i].name);
    if ((failed = compare_expect_int64(int64_preds + i)) > 0) {
      printf(something_failed, failed);
    } else {
      printf(all_tests_passed);
    }

    printf("----------------------------------------\n");
  }

  for (i = 0; i < ARR_SIZE(uint64_preds); ++i) {
    printf("%s series:\n", uint64_preds[i].name);
    if ((failed = compare_expect_uint64(uint64_preds + i)) > 0) {
      printf(something_failed, failed);
    } else {
      printf(all_tests_passed);
    }

    printf("----------------------------------------\n");
  }

  /*----------------------------------------------------------------------*/

  puts("signed/zero-extend tests:");

  failed = 0;
  failed += test_i64_sext_i32(-1, -1LL);
  failed += test_i64_sext_i32(10, 10LL);
  failed += test_i64_sext_i32(0x7fffffff, 0x7fffffffLL);
  failed += test_i64_sext_i16(-1, -1LL);
  failed += test_i64_sext_i16(10, 10LL);
  failed += test_i64_sext_i16(0x7fff, 0x7fffLL);
  failed += test_i64_sext_i8(-1, -1LL);
  failed += test_i64_sext_i8(10, 10LL);
  failed += test_i64_sext_i8(0x7f, 0x7fLL);

  failed += test_i64_zext_i32(0xffffffff, 0x00000000ffffffffLLU);
  failed += test_i64_zext_i32(0x01234567, 0x0000000001234567LLU);
  failed += test_i64_zext_i16(0xffff,     0x000000000000ffffLLU);
  failed += test_i64_zext_i16(0x569a,     0x000000000000569aLLU);
  failed += test_i64_zext_i8(0xff,        0x00000000000000ffLLU);
  failed += test_i64_zext_i8(0xa0,        0x00000000000000a0LLU);

  if (failed > 0) {
    printf("  %d tests failed.\n", failed);
  } else {
    printf("  All tests passed.\n");
  }

  printf("----------------------------------------\n");

  failed = 0;
  puts("signed left/right shift tests:");
  failed += test_i64_constant_shift("i64_shl_const", i64_shl_const, tval_a,     0x00047dc7ec114c00LL);
  failed += test_i64_variable_shift("i64_shl",       i64_shl,       tval_a, 10, 0x00047dc7ec114c00LL);
  failed += test_i64_constant_shift("i64_srl_const", i64_srl_const, tval_a,     0x0000000047dc7ec1LL);
  failed += test_i64_variable_shift("i64_srl",       i64_srl,       tval_a, 10, 0x0000000047dc7ec1LL);
  failed += test_i64_constant_shift("i64_sra_const", i64_sra_const, tval_a,     0x0000000047dc7ec1LL);
  failed += test_i64_variable_shift("i64_sra",       i64_sra,       tval_a, 10, 0x0000000047dc7ec1LL);

  if (failed > 0) {
    printf("  %d tests ailed.\n", failed);
  } else {
    printf("  All tests passed.\n");
  }

  printf("----------------------------------------\n");

  failed = 0;
  puts("unsigned left/right shift tests:");
  failed += test_u64_constant_shift("u64_shl_const", u64_shl_const,  tval_f,     0xfffc1d404d7ae400LL);
  failed += test_u64_variable_shift("u64_shl",       u64_shl,        tval_f, 10, 0xfffc1d404d7ae400LL);
  failed += test_u64_constant_shift("u64_srl_const", u64_srl_const,  tval_f,     0x003fffffc1d404d7LL);
  failed += test_u64_variable_shift("u64_srl",       u64_srl,        tval_f, 10, 0x003fffffc1d404d7LL);
  failed += test_i64_constant_shift("i64_sra_const", i64_sra_const,  tval_f,     0xffffffffc1d404d7LL);
  failed += test_i64_variable_shift("i64_sra",       i64_sra,        tval_f, 10, 0xffffffffc1d404d7LL);
  failed += test_u64_constant_shift("u64_sra_const", u64_sra_const,  tval_f,     0x003fffffc1d404d7LL);
  failed += test_u64_variable_shift("u64_sra",       u64_sra,        tval_f, 10, 0x003fffffc1d404d7LL);

  if (failed > 0) {
    printf("  %d tests ailed.\n", failed);
  } else {
    printf("  All tests passed.\n");
  }

  printf("----------------------------------------\n");

  int64_t result;
  
  result = i64_mul(tval_g, tval_g);
  printf("%20lld * %20lld = %20lld (0x%016llx)\n", tval_g, tval_g, result, result);
  result = i64_mul(tval_d, tval_e);
  printf("%20lld * %20lld = %20lld (0x%016llx)\n", tval_d, tval_e, result, result);
  /* 0xba7a664f13077c9 */
  result = i64_mul(tval_a, tval_b);
  printf("%20lld * %20lld = %20lld (0x%016llx)\n", tval_a, tval_b, result, result);

  printf("----------------------------------------\n");

  return 0;
}
