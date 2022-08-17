//
// Created by tanmay on 6/27/22.
//

#include<stdio.h>

int main() {
  float a, b, res;
  res=0;

  printf("Multiply Accumulator\n");
  printf("Enter value of a: ");
  scanf("%f", &a);
  printf("Enter value of b: ");
  scanf("%f", &b);

  for(int i = 0; i < b; i++)
    res+=a;

  printf("Product = %f\n", res);
  fAFfp32markForResult(res);

  return 0;
}

