//
// Created by tanmay on 6/27/22.
//

#include<stdio.h>

int main() {
  int c;
  float a, b;

  printf("Simple Calculator\n");
  printf("Enter value of c: ");
  scanf("%d", &c);
  printf("Enter value of a: ");
  scanf("%f", &a);
  printf("Enter value of b: ");
  scanf("%f", &b);

  if(c == 0)
    printf("Sum = %.2f\n", a+b);
  else if(c == 1)
    printf("Difference = %.2f\n", a-b);
  else if(c == 2)
    printf("Product = %.2f\n", a*b);
  else if(c == 3)
    printf("Quotient = %.2f\n", a/b);
  else
    printf("Invalid Option.\n");

  return 0;
}

