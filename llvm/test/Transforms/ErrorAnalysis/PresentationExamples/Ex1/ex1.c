//
// Created by tanmay on 7/20/22.
//

#include<stdio.h>
#include<math.h>

#define TYPE double
#define PRINT_PRECISION_FORMAT "%0.15lf"

__attribute__((noinline))
TYPE atomUExample(TYPE x) {
  TYPE v1 = cosf(x);
  TYPE v2 = 1.0-v1;
  TYPE v3 = x*x;
  TYPE v4 = v2/v3;
  return v4;
}

int main() {
  TYPE x = 0.0;
  printf("x = "PRINT_PRECISION_FORMAT"\n", atomUExample(x));
  return 0;
}

