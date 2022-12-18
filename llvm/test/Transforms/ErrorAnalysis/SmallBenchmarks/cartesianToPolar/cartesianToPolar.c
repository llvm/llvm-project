#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include<stdio.h>
#define TRUE 1
#define FALSE 0

int main() {
  double x, y;

  printf("Enter value of x: ");
  scanf("%lf", &x);
//  printf("Enter value of y: ");
//  scanf("%lf", &y);
  y = 10;

  double x_squared = x*x;
  double y_squared = y*y;
  double sum = x_squared+y_squared;
  double res = sqrt(sum);
  fAFfp64markForResult(sum);

  printf("Result = %lf\n", res);

  return 0;
}