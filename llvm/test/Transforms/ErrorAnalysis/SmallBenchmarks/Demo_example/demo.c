#include <stdio.h>
#include <math.h>
#include <time.h>

#define TYPE float
#define PRINT_PRECISION_FORMAT "%0.7f"
#define SQRT sqrtf

// Inputs
#define X 0.000001
#define Y 1.0

__attribute__((noinline))
TYPE demo(TYPE x, TYPE y) {
  TYPE res = SQRT(x + y)-y;
  return res;
}

int main() {
  // Calculate Time
  clock_t t;
  t = clock();

  TYPE res = demo(X, Y);
  t = clock() - t;
  printf("res = "PRINT_PRECISION_FORMAT"\n", res);
  printf("Time: %f\n", ((double)t)/CLOCKS_PER_SEC);

  return 0;
}