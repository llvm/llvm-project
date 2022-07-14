//
// Created by tanmay on 7/5/22.
//

#include<stdio.h>
#include "/home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h"

int main() {
  float a, b, res;
  res=0;

  printf("Multiply Accumulator\n");
  printf("Enter value of a: ");
  scanf("%f", &a);
  printf("Enter value of b: ");
  scanf("%f", &b);

  for(int i = 0; i < b/2; i++)
    res+=a;

  for(int i = 0; i < b/2; i++)
    res+=a;

  printf("Product = %f\n", res);
  fAFfp32markForResult(res);

  return 0;
}

