#include <stdio.h>

struct Thing {
  int zero;
  int one;
};

int main() {
  struct Thing x;
  x.zero = 1;
  x.one = 2;
  printf("break here\n");
  return 0;
}
