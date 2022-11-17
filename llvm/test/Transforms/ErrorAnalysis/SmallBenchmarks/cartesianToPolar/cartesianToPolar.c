#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include<stdio.h>
#define TRUE 1
#define FALSE 0

double ex0(double x, double y) {
	return sqrt(((x * x) + (y * y)));
}

int main() {
  double x, y;

  printf("Enter value of x: ");
  scanf("%lf", &x);
  printf("Enter value of y: ");
  scanf("%lf", &y);

  double res = ex0(x, y);

  printf("Result = %lf\n", res);

  return 0;
}