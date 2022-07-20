//
// Created by tanmay on 7/20/22.
//

#include<stdio.h>
#include<math.h>

int main() {
  double x, v4;

  printf("First Example from Detecting Floating-Point Errors via Atomic Conditions\n");
  printf("Enter value of x: ");
  scanf("%lf", &x);

  v4 = (1.0-cos(x))/(x*x);

  printf("Function Result = %lf\n", v4);
  fAFfp64markForResult(v4);

  return 0;
}

