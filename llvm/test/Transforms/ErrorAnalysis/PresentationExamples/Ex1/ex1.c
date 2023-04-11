//
// Created by tanmay on 7/20/22.
//

#include<stdio.h>
#include<math.h>

#define TYPE double
#define PRINT_PRECISION_FORMAT "%0.15lf"
#define COS cos

__attribute__((noinline))
TYPE atomUExample(TYPE x) {
  return (1.0-COS(x))/(x*x);;
}

int main() {
  TYPE x = 0.0000001;
  printf("x = "PRINT_PRECISION_FORMAT"\n", atomUExample(x));
  return 0;
}