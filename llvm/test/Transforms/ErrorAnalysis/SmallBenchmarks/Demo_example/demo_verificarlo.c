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

  double mean = 0.0;
  TYPE std_dev = 0.0;

  TYPE res[10000] = {0.0};

  for (int i = 0; i < 10000; ++i) {
    res[i] = demo(X, Y);
//    printf("res = "PRINT_PRECISION_FORMAT"\n", res[i]);
  }

  t = clock() - t;

  for (int i = 0; i < 10000; ++i) {
    mean += res[i];
  }
  mean = mean / 10000;
  for (int i = 0; i < 10000; ++i) {
    std_dev += pow(res[i] - mean, 2);
  }
  printf("mean = %0.15lf\n", mean);
  printf("std_dev = %0.15lf\n", std_dev);
  printf("Significant Bits = %0.15lf\n", -log(std_dev/mean));
  printf("Time: %f\n", ((double)t)/CLOCKS_PER_SEC);

  return 0;
}