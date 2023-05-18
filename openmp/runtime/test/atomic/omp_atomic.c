// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

#define DOUBLE_DIGITS 20  /* dt^DOUBLE_DIGITS */
#define MAX_FACTOR 10
#define KNOWN_PRODUCT 3628800  /* 10! */

int test_omp_atomic()
{
  int sum;
  int diff;
  double dsum = 0;
  double dt = 0.5;  /* base of geometric row for + and - test*/
  double ddiff;
  int product;
  int x;
  int *logics;
  int bit_and = 1;
  int bit_or = 0;
  int exclusiv_bit_or = 0;
  int j;
  int known_sum;
  int known_diff;
  int known_product;
  int result = 0;
  int logic_and = 1;
  int logic_or = 0;
  double dknown_sum;
  double rounding_error = 1.E-9;
  double dpt, div;
  int logicsArray[LOOPCOUNT];
  logics = logicsArray;

  sum = 0;
  diff = 0;
  product = 1;

  // sum of integers test
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 1; i <= LOOPCOUNT; i++) {
      #pragma omp atomic
      sum += i;
    }

  }
  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
  if (known_sum != sum)
  {
    fprintf(stderr,
      "Error in sum with integers: Result was %d instead of %d.\n",
      sum, known_sum);
    result++;
  }

  // difference of integers test
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 0; i < LOOPCOUNT; i++) {
      #pragma omp atomic
      diff -= i;
    }
  }
  known_diff = ((LOOPCOUNT - 1) * LOOPCOUNT) / 2 * -1;
  if (diff != known_diff)
  {
    fprintf (stderr,
      "Error in difference with integers: Result was %d instead of 0.\n",
      diff);
    result++;
  }

  // sum of doubles test
  dsum = 0;
  dpt = 1;
  for (j = 0; j < DOUBLE_DIGITS; ++j) {
    dpt *= dt;
  }
  dknown_sum = (1 - dpt) / (1 -dt);
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 0; i < DOUBLE_DIGITS; ++i) {
      #pragma omp atomic
      dsum += pow (dt, i);
    }
  }
  if (dsum != dknown_sum && (fabs (dsum - dknown_sum) > rounding_error)) {
    fprintf (stderr, "Error in sum with doubles: Result was %f"
      " instead of: %f (Difference: %E)\n",
      dsum, dknown_sum, dsum - dknown_sum);
    result++;
  }

  // difference of doubles test
  dpt = 1;
  for (j = 0; j < DOUBLE_DIGITS; ++j) {
    dpt *= dt;
  }
  ddiff = (1 - dpt) / (1 - dt);
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 0; i < DOUBLE_DIGITS; ++i) {
      #pragma omp atomic
      ddiff -= pow (dt, i);
    }
  }
  if (fabs (ddiff) > rounding_error) {
    fprintf (stderr,
      "Error in difference with doubles: Result was %E instead of 0.0\n",
      ddiff);
    result++;
  }

  // product of integers test
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 1; i <= MAX_FACTOR; i++) {
      #pragma omp atomic
      product *= i;
    }
  }
  known_product = KNOWN_PRODUCT;
  if (known_product != product) {
    fprintf (stderr,
      "Error in product with integers: Result was %d instead of %d\n",
      product, known_product);
    result++;
  }

  // division of integers test
  product = KNOWN_PRODUCT;
  #pragma omp parallel
  {
     int i;
    #pragma omp for
    for (i = 1; i <= MAX_FACTOR; ++i) {
      #pragma omp atomic
      product /= i;
    }
  }
  if (product != 1) {
    fprintf (stderr,
      "Error in product division with integers: Result was %d"
      " instead of 1\n",
      product);
    result++;
  }

  // division of doubles test
  div = 5.0E+5;
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 1; i <= MAX_FACTOR; i++) {
      #pragma omp atomic
      div /= i;
    }
  }
  if (fabs(div-0.137787) >= 1.0E-4 ) {
    result++;
    fprintf (stderr, "Error in division with double: Result was %f"
      " instead of 0.137787\n", div);
  }

  // ++ test
  x = 0;
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 0; i < LOOPCOUNT; ++i) {
      #pragma omp atomic
      x++;
    }
  }
  if (x != LOOPCOUNT) {
    result++;
    fprintf (stderr, "Error in ++\n");
  }

  // -- test
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 0; i < LOOPCOUNT; ++i) {
      #pragma omp atomic
      x--;
    }
  }
  if (x != 0) {
    result++;
    fprintf (stderr, "Error in --\n");
  }

  // bit-and test part 1
  for (j = 0; j < LOOPCOUNT; ++j) {
    logics[j] = 1;
  }
  bit_and = 1;
  #pragma omp parallel
  {
     int i;
    #pragma omp for
    for (i = 0; i < LOOPCOUNT; ++i) {
      #pragma omp atomic
      bit_and &= logics[i];
    }
  }
  if (!bit_and) {
    result++;
    fprintf (stderr, "Error in BIT AND part 1\n");
  }

  // bit-and test part 2
  bit_and = 1;
  logics[LOOPCOUNT / 2] = 0;
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 0; i < LOOPCOUNT; ++i) {
      #pragma omp atomic
      bit_and &= logics[i];
    }
  }
  if (bit_and) {
    result++;
    fprintf (stderr, "Error in BIT AND part 2\n");
  }

  // bit-or test part 1
  for (j = 0; j < LOOPCOUNT; j++) {
    logics[j] = 0;
  }
  bit_or = 0;
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 0; i < LOOPCOUNT; ++i) {
      #pragma omp atomic
      bit_or |= logics[i];
    }
  }
  if (bit_or) {
    result++;
    fprintf (stderr, "Error in BIT OR part 1\n");
  }

  // bit-or test part 2
  bit_or = 0;
  logics[LOOPCOUNT / 2] = 1;
  #pragma omp parallel
  {

    int i;
    #pragma omp for
    for (i = 0; i < LOOPCOUNT; ++i) {
      #pragma omp atomic
      bit_or |= logics[i];
    }
  }
  if (!bit_or) {
    result++;
    fprintf (stderr, "Error in BIT OR part 2\n");
  }

  // bit-xor test part 1
  for (j = 0; j < LOOPCOUNT; j++) {
    logics[j] = 0;
  }
  exclusiv_bit_or = 0;
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 0; i < LOOPCOUNT; ++i) {
      #pragma omp atomic
      exclusiv_bit_or ^= logics[i];
    }
  }
  if (exclusiv_bit_or) {
    result++;
    fprintf (stderr, "Error in EXCLUSIV BIT OR part 1\n");
  }

  // bit-xor test part 2
  exclusiv_bit_or = 0;
  logics[LOOPCOUNT / 2] = 1;
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 0; i < LOOPCOUNT; ++i) {
      #pragma omp atomic
      exclusiv_bit_or ^= logics[i];
    }

  }
  if (!exclusiv_bit_or) {
    result++;
    fprintf (stderr, "Error in EXCLUSIV BIT OR part 2\n");
  }

  // left shift test
  x = 1;
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 0; i < 10; ++i) {
      #pragma omp atomic
      x <<= 1;
    }

  }
  if ( x != 1024) {
    result++;
    fprintf (stderr, "Error in <<\n");
    x = 1024;
  }

  // right shift test
  #pragma omp parallel
  {
    int i;
    #pragma omp for
    for (i = 0; i < 10; ++i) {
      #pragma omp atomic
      x >>= 1;
    }
  }
  if (x != 1) {
    result++;
    fprintf (stderr, "Error in >>\n");
  }

  return (result == 0);
} // test_omp_atomic()

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_atomic()) {
      num_failed++;
    }
  }
  return num_failed;
}
