#include <stdio.h>
#include <gmp.h>
#include <mpfr.h>
#include <math.h>

// Inputs
#define X 0.000001
#define Y 1.0

int main() {
  mpfr_t xl, yl, t1l, t2l, t3l;
  mpfr_t xh, yh, t1h, t2h, t3h;
  mpfr_t x_abs_err, y_abs_err, t1_abs_err, t2_abs_err, t3_abs_err;
  mpfr_t x_rel_err, y_rel_err, t1_rel_err, t2_rel_err, t3_rel_err;
  mpfr_t t1_cond_x;

  mpfr_init2 (xl, 32);
  mpfr_init2 (yl, 32);
  mpfr_init2 (t1l, 32);
  mpfr_init2 (t2l, 32);
  mpfr_init2 (t3l, 32);

  mpfr_init2 (xh, 200);
  mpfr_init2 (yh, 200);
  mpfr_init2 (t1h, 200);
  mpfr_init2 (t2h, 200);
  mpfr_init2 (t3h, 200);

  mpfr_init2 (x_abs_err, 200);
  mpfr_init2 (y_abs_err, 200);
  mpfr_init2 (t1_abs_err, 200);
  mpfr_init2 (t2_abs_err, 200);
  mpfr_init2 (t3_abs_err, 200);

  mpfr_init2 (x_rel_err, 200);
  mpfr_init2 (y_rel_err, 200);
  mpfr_init2 (t1_rel_err, 200);
  mpfr_init2 (t2_rel_err, 200);
  mpfr_init2 (t3_rel_err, 200);

  mpfr_init2 (t1_cond_x, 200);

  mpfr_set_flt (xl, X, MPFR_RNDN);
  mpfr_set_flt (yl, Y, MPFR_RNDN);

  mpfr_set_d (xh, X, MPFR_RNDN);
  mpfr_set_d (yh, Y, MPFR_RNDN);

  mpfr_printf("xl = %.21Rf\n", xl);
  mpfr_printf("xh = %.21Rf\n", xh);
  mpfr_sub(x_abs_err, xl, xh, MPFR_RNDN);
  mpfr_div(x_rel_err, x_abs_err, xh, MPFR_RNDN);
  mpfr_printf("Relative Error in x = %.21Rf\n\n", x_rel_err);

  mpfr_printf("yl = %.21Rf\n", yl);
  mpfr_printf("yh = %.21Rf\n", yh);
  mpfr_sub(y_abs_err, yl, yh, MPFR_RNDN);
  mpfr_div(y_rel_err, y_abs_err, yh, MPFR_RNDN);
  mpfr_printf("Relative Error in y = %.21Rf\n\n", y_rel_err);

  mpfr_add(t1l, xl, yl, MPFR_RNDN);
  mpfr_add(t1h, xh, yh, MPFR_RNDN);

  mpfr_printf("t1l = %.21Rf\n", t1l);
  mpfr_printf("t1h = %.21Rf\n", t1h);
  mpfr_sub(t1_abs_err, t1l, t1h, MPFR_RNDN);
  mpfr_div(t1_rel_err, t1_abs_err, t1h, MPFR_RNDN);
  mpfr_printf("Relative Error in t1 = %.21Rf\n\n", t1_rel_err);

  mpfr_div(t1_cond_x, xh, t1h, MPFR_RNDN);
  mpfr_printf("t1_cond_x = %.21Rf\n\n", t1_cond_x);

  mpfr_sqrt(t2l, t1l, MPFR_RNDN);
  mpfr_sqrt(t2h, t1h, MPFR_RNDN);

  mpfr_printf("t2l = %.21Rf\n", t2l);
  mpfr_printf("t2h = %.21Rf\n", t2h);
  mpfr_sub(t2_abs_err, t2l, t2h, MPFR_RNDN);
  mpfr_div(t2_rel_err, t2_abs_err, t2h, MPFR_RNDN);
  mpfr_printf("Relative Error in t2 = %.21Rf\n\n", t2_rel_err);

  mpfr_sub(t3l, t2l, yl, MPFR_RNDN);
  mpfr_sub(t3h, t2h, yh, MPFR_RNDN);

  mpfr_printf("t3l = %.21Rf\n", t3l);
  mpfr_printf("t3h = %.21Rf\n", t3h);
  mpfr_sub(t3_abs_err, t3l, t3h, MPFR_RNDN);
  mpfr_div(t3_rel_err, t3_abs_err, t3h, MPFR_RNDN);
  mpfr_printf("Relative Error in t3 = %.21Rf\n\n", t3_rel_err);

  mpfr_clear (xl);
  mpfr_clear (yl);
  mpfr_clear (t1l);
  mpfr_clear (t2l);
  mpfr_clear (t3l);

  mpfr_clear (xh);
  mpfr_clear (yh);
  mpfr_clear (t1h);
  mpfr_clear (t2h);
  mpfr_clear (t3h);

  mpfr_clear (t1_cond_x);

  mpfr_clear (x_abs_err);
  mpfr_clear (y_abs_err);
  mpfr_clear (t3_abs_err);

  mpfr_clear (x_rel_err);
  mpfr_clear (y_rel_err);
  mpfr_clear (t3_rel_err);

  mpfr_free_cache ();
  return 0;
}