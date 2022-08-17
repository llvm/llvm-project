//
// Created by tanmay on 7/5/22.
//

#include<stdio.h>

int main() {
  double a, b, res;
  res=0;

  printf("Multiply Accumulator\n");
  printf("Enter value of a: ");
  scanf("%lf", &a);
  printf("Enter value of b: ");
  scanf("%lf", &b);

  for(int i = 0; i < b/2; i++)
    res+=a;

  for(int i = 0; i < b/2; i++)
    res+=a;

  printf("Product = %f\n", res);
  fAFfp64markForResult(res);

  return 0;
}

