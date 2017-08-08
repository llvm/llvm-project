#include <time.h>
#include <stdio.h>

int main() {
  int c = 0;
  int its = 100;
  clock_t start = clock(), diff;
  #pragma unroll
  for (int i = 0; i < its; i++) {
    c += i;
  }
  diff = clock() - start;
  int msec = (diff * 1000000) / CLOCKS_PER_SEC;
  printf("%d\n", msec);
  return c;
}
