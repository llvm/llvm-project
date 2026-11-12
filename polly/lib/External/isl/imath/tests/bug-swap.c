/* Regression test for mp_int_swap() bug on self-stored values. */
#include <stdio.h>
#include "imath.h"

int main(int argc, char* argv[]) {
  mpz_t a, b;
  int result;

  mp_int_init_value(&a, 1);
  mp_int_init_value(&b, 16);

  mp_int_swap(&a, &b);
  result = (a.digits == &(a.single) && b.digits == &(b.single) &&
            a.digits[0] == 16 && b.digits[0] == 1);

  printf("REGRESSION: mp_int_swap() on self-stored values: %s\n",
         result ? "OK" : "FAILED");

  mp_int_clear(&b);
  mp_int_clear(&a);
  return !result;
}
