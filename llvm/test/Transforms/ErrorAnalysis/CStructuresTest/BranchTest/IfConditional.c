#include<stdio.h>

#define TYPE double
#define PRINT_PRECISION_FORMAT "%0.15lf"
#define SCAN_PRECISION_FORMAT "%lf"

TYPE calculate_using_operands(TYPE a, TYPE b) {
  int c;
  TYPE d;

  printf("Simple Calculator\n");
  printf("Enter value of c: ");
  scanf("%d", &c);

  if(c == 0) {
    printf("Sum = "PRINT_PRECISION_FORMAT"\n", a + b);
    d = a + b;
  }
  else if(c == 1) {
    printf("Difference = \"PRINT_PRECISION_FORMAT\"\n", a - b);
    d = a - b;
  }
  else if(c == 2) {
    printf("Product = \"PRINT_PRECISION_FORMAT\"\n", a * b);
    d = a * b;
  }
  else if(c == 3) {
    printf("Quotient = \"PRINT_PRECISION_FORMAT\"\n", a / b);
    d = a / b;
  }
  else
    printf("Invalid Option.\n");

  return d;
}

int main() {
  int c;
  float a, b;

  printf("Simple Calculator\n");
  printf("Enter value of c: ");
  scanf("%d", &c);
  printf("Enter value of a: ");
  scanf(SCAN_PRECISION_FORMAT, &a);
  printf("Enter value of b: ");
  scanf(SCAN_PRECISION_FORMAT, &b);

  calculate_using_operands(a, b);

  return 0;
}

