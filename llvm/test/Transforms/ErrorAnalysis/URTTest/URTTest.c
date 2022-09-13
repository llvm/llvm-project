//
// Created by tanmay on 8/29/22.
//

#include<stdio.h>

float sum(float a, float b) {
  float c = a+b;
  fAFfp32markForResult(c);
  return c;
}

int main() {
  float a, b, c;

  printf("Add operation\n");
  printf("Enter value of a: ");
  scanf("%f", &a);
  printf("Enter value of b: ");
  scanf("%f", &b);

  c = sum(a, b);
  printf("Sum: %f\n", c);
  fURTff(0.1, sum, a-1, a+1, b-1, b+1);

  return 0;
}
