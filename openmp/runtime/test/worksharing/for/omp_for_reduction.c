// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp_testsuite.h"

#define DOUBLE_DIGITS 20  /* dt^DOUBLE_DIGITS */
#define MAX_FACTOR 10
#define KNOWN_PRODUCT 3628800  /* 10! */

int test_omp_for_reduction ()
{
  double dt;
  int sum;
  int diff;
  int product = 1;
  double dsum;
  double dknown_sum;
  double ddiff;
  int logic_and;
  int logic_or;
  int bit_and;
  int bit_or;
  int exclusiv_bit_or;
  int *logics;
  int i;
  int known_sum;
  int known_product;
  double rounding_error = 1.E-9; /* over all rounding error to be
                    ignored in the double tests */
  double dpt;
  int result = 0;
  int logicsArray[LOOPCOUNT];

  /* Variables for integer tests */
  sum = 0;
  product = 1;
  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
  /* variabels for double tests */
  dt = 1. / 3.;  /* base of geometric row for + and - test*/
  dsum = 0.;
  /* Variabeles for logic  tests */
  logics = logicsArray;
  logic_and = 1;
  logic_or = 0;
  /* Variabeles for bit operators tests */
  bit_and = 1;
  bit_or = 0;
  /* Variables for exclusiv bit or */
  exclusiv_bit_or = 0;

  /************************************************************************/
  /** Tests for integers                         **/
  /************************************************************************/

  /**** Testing integer addition ****/
  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(+:sum)
    for (j = 1; j <= LOOPCOUNT; j++) {
      sum = sum + j;
    }
  }
  if (known_sum != sum) {
    result++;
    fprintf (stderr, "Error in sum with integers: Result was %d"
      " instead of %d.\n", sum, known_sum);
  }

  /**** Testing integer subtracton ****/
  diff = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(-:diff)
    for (j = 1; j <= LOOPCOUNT; j++) {
      diff = diff - j;
    }
  }
  if (diff != 0) {
    result++;
    fprintf (stderr, "Error in difference with integers: Result was %d"
      " instead of 0.\n", diff);
  }

  /**** Testing integer multiplication ****/
  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(*:product)
    for (j = 1; j <= MAX_FACTOR; j++) {
      product *= j;
    }
  }
  known_product = KNOWN_PRODUCT;
  if(known_product != product) {
    result++;
    fprintf (stderr,"Error in Product with integers: Result was %d"
      " instead of %d\n",product,known_product);
  }

  /************************************************************************/
  /** Tests for doubles                          **/
  /************************************************************************/

  /**** Testing double addition ****/
  dsum = 0.;
  dpt = 1.;
  for (i = 0; i < DOUBLE_DIGITS; ++i) {
    dpt *= dt;
  }
  dknown_sum = (1 - dpt) / (1 - dt);
  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(+:dsum)
    for (j = 0; j < DOUBLE_DIGITS; j++) {
      dsum += pow (dt, j);
    }
  }
  if (fabs (dsum - dknown_sum) > rounding_error) {
    result++;
    fprintf (stderr, "\nError in sum with doubles: Result was %f"
      " instead of: %f (Difference: %E)\n",
      dsum, dknown_sum, dsum-dknown_sum);
  }

  /**** Testing double subtraction ****/
  ddiff = (1 - dpt) / (1 - dt);
  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(-:ddiff)
    for (j = 0; j < DOUBLE_DIGITS; ++j) {
      ddiff -= pow (dt, j);
    }
  }
  if (fabs (ddiff) > rounding_error) {
    result++;
    fprintf (stderr, "Error in Difference with doubles: Result was %E"
      " instead of 0.0\n", ddiff);
  }


  /************************************************************************/
  /** Tests for logical values                       **/
  /************************************************************************/

  /**** Testing logic and ****/
  for (i = 0; i < LOOPCOUNT; i++) {
    logics[i] = 1;
  }

  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(&&:logic_and)
    for (j = 0; j < LOOPCOUNT; ++j) {
      logic_and = (logic_and && logics[j]);
    }
  }
  if(!logic_and) {
    result++;
    fprintf (stderr, "Error in logic AND part 1\n");
  }

  logic_and = 1;
  logics[LOOPCOUNT / 2] = 0;

  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(&&:logic_and)
    for (j = 0; j < LOOPCOUNT; ++j) {
      logic_and = logic_and && logics[j];
    }
  }
  if(logic_and) {
    result++;
    fprintf (stderr, "Error in logic AND part 2\n");
  }

  /**** Testing logic or ****/
  for (i = 0; i < LOOPCOUNT; i++) {
    logics[i] = 0;
  }

  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(||:logic_or)
    for (j = 0; j < LOOPCOUNT; ++j) {
      logic_or = logic_or || logics[j];
    }
  }
  if (logic_or) {
    result++;
    fprintf (stderr, "Error in logic OR part 1\n");
  }

  logic_or = 0;
  logics[LOOPCOUNT / 2] = 1;

  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(||:logic_or)
    for (j = 0; j < LOOPCOUNT; ++j) {
      logic_or = logic_or || logics[j];
    }
  }
  if(!logic_or) {
    result++;
    fprintf (stderr, "Error in logic OR part 2\n");
  }

  /************************************************************************/
  /** Tests for bit values                         **/
  /************************************************************************/

  /**** Testing bit and ****/
  for (i = 0; i < LOOPCOUNT; ++i) {
    logics[i] = 1;
  }

  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(&:bit_and)
    for (j = 0; j < LOOPCOUNT; ++j) {
      bit_and = (bit_and & logics[j]);
    }
  }
  if (!bit_and) {
    result++;
    fprintf (stderr, "Error in BIT AND part 1\n");
  }

  bit_and = 1;
  logics[LOOPCOUNT / 2] = 0;

  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(&:bit_and)
    for (j = 0; j < LOOPCOUNT; ++j) {
      bit_and = bit_and & logics[j];
    }
  }
  if (bit_and) {
    result++;
    fprintf (stderr, "Error in BIT AND part 2\n");
  }

  /**** Testing bit or ****/
  for (i = 0; i < LOOPCOUNT; i++) {
    logics[i] = 0;
  }

  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(|:bit_or)
    for (j = 0; j < LOOPCOUNT; ++j) {
      bit_or = bit_or | logics[j];
    }
  }
  if (bit_or) {
    result++;
    fprintf (stderr, "Error in BIT OR part 1\n");
  }

  bit_or = 0;
  logics[LOOPCOUNT / 2] = 1;

  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(|:bit_or)
    for (j = 0; j < LOOPCOUNT; ++j) {
      bit_or = bit_or | logics[j];
    }
  }
  if (!bit_or) {
    result++;
    fprintf (stderr, "Error in BIT OR part 2\n");
  }

  /**** Testing exclusive bit or ****/
  for (i = 0; i < LOOPCOUNT; i++) {
    logics[i] = 0;
  }

  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(^:exclusiv_bit_or)
    for (j = 0; j < LOOPCOUNT; ++j) {
      exclusiv_bit_or = exclusiv_bit_or ^ logics[j];
    }
  }
  if (exclusiv_bit_or) {
    result++;
    fprintf (stderr, "Error in EXCLUSIV BIT OR part 1\n");
  }

  exclusiv_bit_or = 0;
  logics[LOOPCOUNT / 2] = 1;

  #pragma omp parallel
  {
    int j;
    #pragma omp for schedule(dynamic,1) reduction(^:exclusiv_bit_or)
    for (j = 0; j < LOOPCOUNT; ++j) {
      exclusiv_bit_or = exclusiv_bit_or ^ logics[j];
    }
  }
  if (!exclusiv_bit_or) {
    result++;
    fprintf (stderr, "Error in EXCLUSIV BIT OR part 2\n");
  }

  return (result == 0);
  free (logics);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_reduction()) {
      num_failed++;
    }
  }
  return num_failed;
}
