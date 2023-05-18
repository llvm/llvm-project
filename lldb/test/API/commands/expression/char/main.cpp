#include <stdio.h>

char g = 0;
signed char gs = 0;
unsigned char gu = 0;

int foo(char c) { return 1; }
int foo(signed char c) { return 2; }
int foo(unsigned char c) { return 3; }

int main() {
  char c = 0;
  signed char sc = 0;
  unsigned char uc = 0;
  printf("%d %d %d\n", foo(c), foo(sc), foo(uc));
  return 0; // Break here
}
