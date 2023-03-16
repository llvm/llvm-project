//
// Created by tanmay on 6/27/22.
//

#include<stdio.h>

__attribute__((noinline))
float temp(float a, float res) {
  for(int i = 0; i < 3; i++) {
    res -= a;
    printf("Result = %0.15f\n", res);
  }

  return res;
}

int main() {
  float a, res;
  int b;
  res=0.1;

  printf("Multiply Accumulator\n");
  printf("Enter value of a: ");
  scanf("%f", &a);
  printf("Enter value of b: ");
  scanf("%d", &b);

  res = temp(a, res);


  printf("Product = %0.15f\n", res);
//  fAFfp32markForResult(res);

  return 0;
}

