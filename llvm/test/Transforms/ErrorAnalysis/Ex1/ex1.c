//
// Created by tanmay on 7/20/22.
//

#include<stdio.h>
#include<math.h>

int main() {
  double x, v1, v2, v3, v4;

  printf("First Example from Detecting Floating-Point Errors via Atomic Conditions\n");
  printf("Enter value of x: ");
  scanf("%lf", &x);

  v1 = cos(x);
  v2 = 1.0-v1;
  v3 = x*x;
  v4 = v2/v3;
//  v4 = (1.0-cos(x))/(x*x);

  printf("v1 = %0.18lf\n", v1);
  printf("v2 = %0.18lf\n", v2);
  printf("v3 = %0.18lf\n", v3);
  printf("v4 = %0.18lf\n\n", v4);

  printf("v1 = %0.18e\n", v1);
  printf("v2 = %0.18e\n", v2);
  printf("v3 = %0.18e\n", v3);
  printf("v4 = %0.18e\n", v4);
  fAFfp64markForResult(v4);

  return 0;
}

