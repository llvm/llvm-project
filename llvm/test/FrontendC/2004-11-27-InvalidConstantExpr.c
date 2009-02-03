// RUN: %llvmgcc %s -S -o - | not grep {foo\\* sub}
// This should not produce a subtrace constantexpr of a pointer
struct foo {
  int Y;
  char X[100];
} F;

int test(char *Y) {
   return Y - F.X;
} 
