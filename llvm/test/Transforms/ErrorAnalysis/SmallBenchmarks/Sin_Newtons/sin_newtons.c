#include <math.h>
#include <stdio.h>
#include <time.h>

#define TYPE double
#define PRINT_PRECISION_FORMAT "%0.15lf"
#define FABS fabs

#define ANGLE_IN_RADIANS 3*M_PI/2
//#define TOLERANCE 0.0000001 // For float to reach result with 0 error. Requiores 8 iterations
#define TOLERANCE 0.00000000001 // For double to reach result with 0 error. Requiores 7 iterations
#define N 10


__attribute__((noinline))
TYPE newtons_sin(TYPE x) {
  TYPE sin_x = x;
  TYPE old_sin_x;
  int I = 0;

  do {
    // Compute Powers of approximations of sin_x to be used for the Taylor expansion of sin and cos.
    TYPE pow_2 = sin_x * sin_x;
    TYPE pow_3 = pow_2 * sin_x;
    TYPE pow_4 = pow_3 * sin_x;
    TYPE pow_5 = pow_4 * sin_x;
    TYPE pow_6 = pow_5 * sin_x;
    TYPE pow_7 = pow_6 * sin_x;

    // Compute the terms in the Taylor expansion of sin.
    TYPE sin_second_taylor_term = pow_3 / 6.0;
    TYPE sin_third_taylor_term = pow_5 / 120.0;
    TYPE sin_fourth_taylor_term = pow_7 / 5040.0;

    // Compute the terms in the Taylor expansion of cos.
    TYPE cos_second_taylor_term = pow_2 / 2.0;
    TYPE cos_third_taylor_term = pow_4 / 24.0;
    TYPE cos_fourth_taylor_term = pow_6 / 720.0;

    old_sin_x = sin_x;
    // Use Newtons method to compute the next approximation of sin(x).
    // x-(Taylor expansion of sin(x))/Taylor expansion of cos(x))
    sin_x = sin_x - ((sin_x - sin_second_taylor_term + sin_third_taylor_term - sin_fourth_taylor_term) /
             (1.0 - cos_second_taylor_term + cos_third_taylor_term - cos_fourth_taylor_term));

//    printf("sin(x) = "PRINT_PRECISION_FORMAT"\n", sin_x);
    I++;
  } while(FABS(sin_x - old_sin_x) > TOLERANCE && I<N);

  printf("Iterations: %d\n", I);

  return sin_x;
}

int main() {
  // Calculate Time
  clock_t t;
  t = clock();

  TYPE sin_x = newtons_sin(ANGLE_IN_RADIANS);

  t = clock() - t;
  printf("Time: %f\n", ((double)t)/CLOCKS_PER_SEC);
  printf("sin(x) = "PRINT_PRECISION_FORMAT"\n", sin_x);

  return 0;
}