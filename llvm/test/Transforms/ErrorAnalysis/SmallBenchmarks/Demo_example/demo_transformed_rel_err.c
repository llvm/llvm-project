#include <stdio.h>
#include <gmp.h>
#include <mpfr.h>
#include <math.h>

// Inputs
#define X 0.000001
#define Y 1.0

int main() {
  mpfr_t xl, yl, t1l, t2l, t3l, t4l;
  mpfr_t xh, yh, t1h, t2h, t3h, t4h;
  mpfr_t x_abs_err, y_abs_err, t4_abs_err;
  mpfr_t x_rel_err, y_rel_err, t4_rel_err;

  mpfr_init2 (xl, 32);
  mpfr_init2 (yl, 32);
  mpfr_init2 (t1l, 32);
  mpfr_init2 (t2l, 32);
  mpfr_init2 (t3l, 32);
  mpfr_init2 (t4l, 32);

  mpfr_init2 (xh, 64);
  mpfr_init2 (yh, 64);
  mpfr_init2 (t1h, 64);
  mpfr_init2 (t2h, 64);
  mpfr_init2 (t3h, 64);
  mpfr_init2 (t4h, 64);

  mpfr_init2 (x_abs_err, 52);
  mpfr_init2 (y_abs_err, 52);
  mpfr_init2 (t4_abs_err, 52);

  mpfr_init2 (x_rel_err, 52);
  mpfr_init2 (y_rel_err, 52);
  mpfr_init2 (t4_rel_err, 52);


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
  mpfr_sqrt(t2l, t1l, MPFR_RNDN);
  mpfr_add(t3l, t2l, yl, MPFR_RNDN);
  mpfr_div(t4l, xl, t3l, MPFR_RNDN);

  mpfr_add(t1h, xh, yh, MPFR_RNDN);
  mpfr_sqrt(t2h, t1h, MPFR_RNDN);
  mpfr_add(t3h, t2h, yh, MPFR_RNDN);
  mpfr_div(t4h, xh, t3h, MPFR_RNDN);

  mpfr_printf("t4l = %.21Rf\n", t4l);
  mpfr_printf("t4h = %.21Rf\n", t4h);
  mpfr_sub(t4_abs_err, t4l, t4h, MPFR_RNDN);
  mpfr_div(t4_rel_err, t4_abs_err, t4h, MPFR_RNDN);
  mpfr_printf("Relative Error in t4 = %.21Rf\n\n", t4_rel_err);

  mpfr_clear (xl);
  mpfr_clear (yl);
  mpfr_clear (t1l);
  mpfr_clear (t2l);
  mpfr_clear (t3l);
  mpfr_clear (t4l);

  mpfr_clear (xh);
  mpfr_clear (yh);
  mpfr_clear (t1h);
  mpfr_clear (t2h);
  mpfr_clear (t3h);
  mpfr_clear (t4h);

  mpfr_clear (x_abs_err);
  mpfr_clear (y_abs_err);
  mpfr_clear (t4_abs_err);

  mpfr_clear (x_rel_err);
  mpfr_clear (y_rel_err);
  mpfr_clear (t4_rel_err);

  mpfr_free_cache ();
  return 0;
}